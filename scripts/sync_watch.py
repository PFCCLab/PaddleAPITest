"""Local file watcher that syncs .py changes to a remote HTTP server.

Watches the local PaddleAPITest directory for .py file changes and:
1. Uploads each changed file via POST /admin/upload_file
2. Triggers a server restart via POST /admin/restart
3. Polls /health until the server is back up

Usage:
    pip install watchdog          # one-time setup
    python scripts/sync_watch.py \\
        --host 10.78.119.13 --port 8089 --token <admin_token> \\
        [--watch_dir /workspace/PaddleAPITest] \\
        [--debounce 1.5]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("watchdog is required. Install it with: pip install watchdog")
    raise

# Patterns matching paths that should NOT be synced (mirrors .gitignore)
SKIP_PATTERNS = [
    "__pycache__",
    ".mypy_cache",
    ".vscode",
    "test_log",
    "api_config/api_config",
    "api_config/output",
    ".ipynb_checkpoints",
    "trace_output",
    ".huggingface",
]


class SyncHandler(FileSystemEventHandler):
    def __init__(self, args: argparse.Namespace, watch_dir: str) -> None:
        super().__init__()
        self.args = args
        self.watch_dir = watch_dir
        self._pending: set[str] = set()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # watchdog callbacks                                                   #
    # ------------------------------------------------------------------ #

    def on_modified(self, event):
        self._enqueue(event)

    def on_created(self, event):
        self._enqueue(event)

    def _enqueue(self, event):
        if event.is_directory:
            return
        path: str = event.src_path
        if not path.endswith(".py"):
            return
        if any(p in path for p in SKIP_PATTERNS):
            return
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.args.debounce, self._flush)
            self._timer.start()

    # ------------------------------------------------------------------ #
    # sync + restart                                                       #
    # ------------------------------------------------------------------ #

    def _flush(self):
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
            self._timer = None

        uploaded = []
        for path in paths:
            if self._upload_file(path):
                rel = os.path.relpath(path, self.watch_dir)
                uploaded.append(rel)

        if uploaded:
            print(f"[sync] Uploaded {len(uploaded)} file(s): {', '.join(uploaded)}", flush=True)
            self._trigger_restart()

    def _upload_file(self, abs_path: str) -> bool:
        rel = os.path.relpath(abs_path, self.watch_dir)
        try:
            content = Path(abs_path).read_text(encoding="utf-8")
        except Exception as e:
            print(f"[sync] Read error for {rel}: {e}", flush=True)
            return False

        body = json.dumps({"path": rel, "content": content}).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.args.host}:{self.args.port}/admin/upload_file",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Admin-Token": self.args.token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                if result.get("status") != "ok":
                    print(f"[sync] Upload failed for {rel}: {result}", flush=True)
                    return False
            return True
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            print(f"[sync] Upload HTTP error {e.code} for {rel}: {body_text}", flush=True)
            return False
        except Exception as e:
            print(f"[sync] Upload error for {rel}: {e}", flush=True)
            return False

    def _trigger_restart(self):
        req = urllib.request.Request(
            f"http://{self.args.host}:{self.args.port}/admin/restart",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Admin-Token": self.args.token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                print(f"[sync] Restart triggered: {result}", flush=True)
        except Exception as e:
            print(f"[sync] Restart request failed: {e}", flush=True)
            return

        # Poll /health until the server comes back up (max 60 seconds)
        health_url = f"http://{self.args.host}:{self.args.port}/health"
        print("[sync] Waiting for server to come back up...", flush=True)
        for i in range(60):
            time.sleep(1)
            try:
                with urllib.request.urlopen(health_url, timeout=3) as r:
                    data = json.loads(r.read().decode())
                    if data.get("status") == "ok":
                        print(f"[sync] Server ready (attempt {i + 1}).", flush=True)
                        return
            except Exception:
                pass
        print("[sync] Warning: server did not come back up within 60 seconds.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Watch local .py files and sync changes to the remote HTTP server."
    )
    parser.add_argument("--host", required=True, help="Remote server host, e.g. 10.78.119.13")
    parser.add_argument("--port", type=int, default=8089, help="Remote server port")
    parser.add_argument("--token", required=True, help="Admin token (--admin_token on server)")
    parser.add_argument(
        "--watch_dir",
        default=str(Path(__file__).resolve().parent.parent),
        help="Local directory to watch (default: repo root)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=1.5,
        help="Seconds to wait after last change before syncing (default: 1.5)",
    )
    args = parser.parse_args()

    watch_dir = os.path.abspath(args.watch_dir)
    print(f"[sync] Watching {watch_dir}", flush=True)
    print(f"[sync] Remote: http://{args.host}:{args.port}", flush=True)
    print(f"[sync] Debounce: {args.debounce}s", flush=True)

    # Verify connectivity before starting the observer
    health_url = f"http://{args.host}:{args.port}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            data = json.loads(r.read().decode())
            print(f"[sync] Server OK: {data}", flush=True)
    except Exception as e:
        print(f"[sync] Warning: cannot reach server at {health_url}: {e}", flush=True)

    event_handler = SyncHandler(args, watch_dir)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=True)
    observer.start()
    print("[sync] Observer started. Press Ctrl+C to stop.", flush=True)

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        print("[sync] Stopping...", flush=True)
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
