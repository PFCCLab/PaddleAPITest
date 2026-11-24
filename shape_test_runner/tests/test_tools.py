#!/usr/bin/env python3
"""shape_test_runner 的离线自测：不依赖 GPU / paddle / torch。

用合成的 engine 日志目录验证进度统计、单 case 日志拆分、OOM 缺日志标注和
复现脚本生成，避免每次都要真实跑一轮才能验证工具本身。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

import campaign  # noqa: E402
import debug_bundle  # noqa: E402
import lib  # noqa: E402
import report  # noqa: E402

C_ADD = 'paddle.add(Tensor(paddle.Size([0, 4]),"float32"), Tensor(paddle.Size([0, 4]),"float32"))'
C_RELU = 'paddle.relu(Tensor(paddle.Size([0, 8]),"float32"))'
C_MM = 'paddle.matmul(Tensor(paddle.Size([1024, 1024]),"float32"), Tensor(paddle.Size([1024, 1024]),"float32"))'
C_OOM = 'paddle.matmul(Tensor(paddle.Size([8192, 8192]),"float32"), Tensor(paddle.Size([8192, 8192]),"float32"))'
C_BIG = 'paddle.matmul(Tensor(paddle.Size([4096, 4096]),"float32"), Tensor(paddle.Size([4096, 4096]),"float32"))'


def _case_block(uid: str, config: str, status: str, body: str = "") -> str:
    return f">>> CASE {uid}\n{config}\n{body}\n<<< CASE {uid} | {status}\n\n"


def make_fake_run(root: Path) -> Path:
    """构造一个含 pass / 精度失败 / 运行报错 / OOM 无日志的 run 目录。"""
    run_dir = root / "run_demo"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "inputs" / "0size.txt").write_text(
        "\n".join([C_ADD, C_RELU, C_MM, C_OOM, C_BIG]) + "\n"
    )
    # 日志目录名固定 <shape>_<mode>，与 run.json 的 log_dir 字段和 slot 命名保持一致。
    log_dir = run_dir / "logs" / "0size_accuracy"
    (log_dir / ".tmp").mkdir(parents=True)
    (log_dir / "checkpoint.txt").write_text("\n".join([C_ADD, C_RELU, C_MM]) + "\n")
    (log_dir / "api_config_pass.txt").write_text(C_ADD + "\n")
    (log_dir / "api_config_paddle_accuracy.txt").write_text(C_RELU + "\n")
    (log_dir / "api_config_paddle_error.txt").write_text(C_MM + "\n")
    (log_dir / "api_config_oom.txt").write_text(C_OOM + "\n")
    # 只写 .tmp 下的 worker 片段、不写 log_inorder.log，专门覆盖日志来源的回退分支。
    worker = (
        _case_block("a1", C_ADD, "PASS")
        + _case_block(
            "b2", C_RELU, "FAIL", "[paddle_accuracy] Compare forward\nmax_abs_diff=1.2e-3"
        )
        + _case_block(
            "c3",
            C_MM,
            "FAIL",
            "Traceback (most recent call last):\nRuntimeError: matmul failed at 0x7f00",
        )
        + _case_block("d4", C_BIG, "FAIL", "RuntimeError: matmul failed at 0x7f01")
    )
    (log_dir / ".tmp" / "log_1001.log").write_text(worker)
    # total 故意写成 4（源文件 5 行），用于验证进度按 job 声明值而非配置行数计算。
    run = {
        "run_id": "run_demo",
        "python": "python",
        "auto_report": True,
        "model": "demo",
        "card": "H",
        "inputs": {"0size": {"path": "inputs/0size.txt", "total": 5}},
        "modes": ["accuracy"],
        "jobs": [
            {
                "key": "0size_accuracy",
                "shape": "0size",
                "mode": "accuracy",
                "input": "inputs/0size.txt",
                "log_dir": "logs/0size_accuracy",
                "total": 4,
                "engine_args": ["--accuracy=True", "--num_workers_per_gpu=4"],
                "env": {},
                "merged_from": ["generic_configs/run_0size_accuracy.yaml"],
            }
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False))
    (run_dir / "state.json").write_text(json.dumps({"jobs": {"0size_accuracy": {"exit_code": 1}}}))
    return run_dir


def make_campaign(root: Path) -> Path:
    """构造两 part 的 campaign：各有一条不同的 paddle_error。"""
    camp = root / "camp"
    configs = (C_RELU, C_MM)
    bodies = ("RuntimeError: alpha failed at 0x1111", "RuntimeError: beta failed at 0x2222")
    parts = []
    for index, config in enumerate(configs):
        part = f"part-{index:02d}"
        pdir = camp / "parts" / part
        log_dir = pdir / "logs" / "0size_accuracy"
        (log_dir / ".tmp").mkdir(parents=True)
        (log_dir / "checkpoint.txt").write_text(config + "\n")
        (log_dir / "api_config_paddle_error.txt").write_text(config + "\n")
        (log_dir / ".tmp" / f"log_{1000 + index}.log").write_text(
            _case_block(str(2000 + index), config, "FAIL", bodies[index])
        )
        # 两个 part 使用同名 run 日志，验证合并时加 part 前缀而不会互相覆盖。
        (log_dir / f"log_20260101_00000{index}_1.log").write_text("run log\n")
        run = {
            "run_id": f"camp/{part}",
            "model": "demo",
            "card": "H",
            "jobs": [
                {
                    "key": "0size_accuracy",
                    "shape": "0size",
                    "mode": "accuracy",
                    "input": "inputs/0size.txt",
                    "log_dir": "logs/0size_accuracy",
                    "total": 1,
                    "engine_args": ["--accuracy=True"],
                    "env": {},
                }
            ],
        }
        (pdir / "run.json").write_text(json.dumps(run))
        (pdir / "state.json").write_text(
            json.dumps(
                {
                    "host": lib.hostname(),
                    "jobs": {"0size_accuracy": {"exit_code": 3, "host": lib.hostname()}},
                }
            )
        )
        (pdir / "heartbeat.json").write_text(
            json.dumps(
                {
                    "host": lib.hostname(),
                    "status": "done",
                    "heartbeat_ts": time.time(),
                }
            )
        )
        shard = camp / "shards" / part / "0size.txt"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(config + "\n")
        parts.append(
            {
                "id": part,
                "shapes": {"0size": f"shards/{part}/0size.txt"},
                "totals": {"0size": 1},
            }
        )
    # dedup=False：campaign 不去重，人工分片里重复的配置按 part 各自记账。
    info = {
        "run_id": "camp",
        "kind": "campaign",
        "modes": ["accuracy"],
        "model": "demo",
        "card": "H",
        "stale_seconds": 120,
        "dedup": False,
        "inputs": {"0size": {"source": "0size.txt", "total": 2}},
        "parts": parts,
    }
    (camp / "campaign.json").write_text(json.dumps(info))
    return camp


class TestConfigValidation(unittest.TestCase):
    # 防的回归：把 1024 里的字符 0 当成 0 维，导致 1M 配置被误报成含 0size case。
    # total 仍按非空非注释行计数：把注释行算成配置会让覆盖率分母虚高。
    def test_zero_dim_detection_ignores_1024(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1M.txt"
            path.write_text(C_MM + "\n" + C_ADD + "\n")
            info = lib.validate_config_file(path)
            self.assertEqual(info["total"], 2)
            self.assertEqual(info["zero_dim_cases"], 1)
            self.assertEqual(info["parse_errors"], [])

    # 防的回归：语法错误被静默吞掉，直到 engine 启动才崩，掩盖配置生成阶段的问题。
    # 只断言上报条数：SyntaxError 文案随 Python 版本变化，不适合钉死字符串。
    def test_parse_error_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("paddle.add(not closed\n")
            info = lib.validate_config_file(path)
            self.assertEqual(len(info["parse_errors"]), 1)


class TestReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = make_fake_run(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    # 防的回归：pass 率分母口径漂移（把 skip 计入或漏掉 oom），与流水线报表对不上。
    # done 取自 checkpoint 行数而非日志块数量，同一配置重跑不会让进度超过 total。
    def test_status_counts(self):
        payload = report.build_status("run_demo", self.run_dir)
        job = payload["jobs"][0]
        self.assertEqual(job["done"], 3)
        self.assertEqual(job["total"], 4)
        self.assertEqual(job["pass"], 1)
        self.assertEqual(job["fail"], 2)
        self.assertEqual(job["ref"], 1)
        # pass 率沿用流水线口径：oom 计入分母，skip 不计入。
        self.assertEqual(job["pass_rate"], 25.0)
        self.assertEqual(payload["totals"]["progress"], 75.0)
        self.assertTrue((self.run_dir / "status.md").exists())

    # 防的回归：OOM 打挂进程时没有 case 块，导出日志会静默漏掉这条配置。
    # 签名聚合先抹掉地址：同一句 RuntimeError 仅地址不同必须并成一组。
    def test_collect_splits_logs_and_marks_missing(self):
        payload = report.collect_issues("run_demo", self.run_dir)
        cases = self.run_dir / "issues" / "cases" / "0size_accuracy"
        self.assertTrue((cases / "paddle_accuracy" / "paddle_accuracy__b2.log").exists())
        self.assertTrue((cases / "paddle_error" / "paddle_error__c3.log").exists())
        missing = (cases / "oom" / "missing_log.txt").read_text()
        self.assertIn(C_OOM, missing)
        # 同一句 RuntimeError（仅地址不同）应聚合成一组签名。
        sigs = {(item["category"], item["signature"]) for item in payload["signatures"]}
        self.assertEqual(len(sigs), 2)
        self.assertTrue((self.run_dir / "issues" / "INDEX.md").exists())

    # 防的回归：参考分类（oom）被写进真实失败清单，复现批次被无意义放大。
    # 每个脚本都过 bash -n：模板拼接出错时在这里先暴露，不必等到真实环境。
    def test_repro_bundle(self):
        report.build_repro("run_demo", self.run_dir)
        repro = self.run_dir / "repro"
        cfg = repro / "configs" / "0size_accuracy_failures.txt"
        # 只有 paddle_accuracy + paddle_error 两类进入真实失败清单。
        self.assertEqual(len(cfg.read_text().strip().splitlines()), 2)
        self.assertTrue((repro / "refs" / "0size_accuracy_oom.txt").exists())
        for script in ("run_0size_accuracy.sh", "run_case.sh"):
            path = repro / script
            self.assertTrue(path.exists())
            # bash -n 只做语法检查，不需要真实环境。
            proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)

    # 防的回归：进程被杀导致缺少 <<< CASE 尾行时整块被丢弃，crash 现场无从定位。
    # 断言状态是 INCOMPLETE 而非 PASS，避免复现包把这条当成正常用例跳过。
    def test_incomplete_case_is_kept(self):
        log_dir = self.run_dir / "logs" / "0size_accuracy"
        unique = 'paddle.flatten(Tensor(paddle.Size([0, 2, 3]),"float32"))'
        (log_dir / ".tmp" / "log_1002.log").write_text(
            ">>> CASE e5\n" + unique + "\nRuntimeError: killed\n"
        )
        blocks = lib.parse_case_blocks(log_dir)
        picked = lib.pick_case_block(blocks[unique])
        self.assertEqual(picked["status"], "INCOMPLETE")

    # 端到端回归：目录形态、清单口径、单 case 日志、README/INDEX 和脚本一次覆盖。
    # 真实失败只应含 accuracy + error；oom 只能出现在 oom_*.txt 参考清单里。
    def test_debug_bundle(self):
        bundle = debug_bundle.build_debug_bundle("run_demo", self.run_dir)
        self.assertIsNotNone(bundle)
        root = Path(bundle["path"])
        self.assertEqual(root.name, "debugH")
        model_dir = root / "demo"
        # 真实失败清单只收 paddle_accuracy + paddle_error
        failures = (model_dir / "accuracy_0size.txt").read_text().strip().splitlines()
        self.assertEqual(len(failures), 2)
        self.assertTrue((model_dir / "oom_accuracy_0size.txt").exists())

        pe = model_dir / "pipeline_errors" / "accuracy_0size"
        self.assertTrue((pe / "paddle_accuracy__b2.log").exists())
        self.assertTrue((pe / "paddle_error__c3.log").exists())
        self.assertIn(C_OOM, (pe / "missing_log.txt").read_text())
        self.assertTrue((pe / "INDEX.md").exists())
        # 顶层 README/INDEX 与可执行脚本
        self.assertTrue((root / "README.md").exists())
        self.assertTrue((root / "INDEX.md").exists())
        for script in ("run_accuracy.sh", "run_case.sh"):
            path = root / script
            self.assertTrue(path.exists())
            proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)

    # 防的回归：没有真实失败时仍生成空 debug<卡型>/，让用户误以为存在失败。
    # 退化路径要求返回 None 而不是抛异常，调用方据此决定是否提示。
    def test_debug_requires_failures(self):
        log_dir = self.run_dir / "logs" / "0size_accuracy"
        for name in (
            "api_config_paddle_accuracy.txt",
            "api_config_paddle_error.txt",
            "api_config_oom.txt",
        ):
            (log_dir / name).unlink()
        self.assertIsNone(debug_bundle.build_debug_bundle("run_demo", self.run_dir))


class TestCampaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # 防的回归：切分丢行或重复行，多机跑完后覆盖率校验永远差几条。
    # 按 shape 分别求和：不同 shape 的分片文件彼此独立，混在一起求和会掩盖单边丢失。
    def test_split_conserves_lines(self):
        src0 = self.root / "0size.txt"
        src1 = self.root / "1M.txt"
        src0.write_text(
            "\n".join(f'paddle.relu(Tensor(paddle.Size([0, {i + 1}]),"float32"))' for i in range(5))
            + "\n"
        )
        src1.write_text(
            "\n".join(
                f'paddle.matmul(Tensor(paddle.Size([{1024 + i}, 1024]),"float32"), '
                f'Tensor(paddle.Size([1024, 1024]),"float32"))'
                for i in range(3)
            )
            + "\n"
        )
        with mock.patch.object(lib, "RUNS_DIR", self.root / "runs"):
            info = campaign.create_campaign(
                "camp_split",
                {"0size": src0, "1M": src1},
                3,
                modes=["accuracy"],
                model="demo",
                card="H",
                python="python",
                gpus="-1",
                sanitizer="off",
                accuracy_mode="auto",
                workers=None,
                timeout=None,
                engine_extra=[],
            )
        self.assertEqual(len(info["parts"]), 3)
        for shape, expected in (("0size", 5), ("1M", 3)):
            total = sum(part["totals"].get(shape, 0) for part in info["parts"])
            self.assertEqual(total, expected)
        # 每个 part 下都应同时有 0size/1M 分片（行数足够时）。
        for part in info["parts"]:
            self.assertEqual(set(part["shapes"]), {"0size", "1M"})

    # 防的回归：contiguous 切法打乱行序，分片内容与源文件顺序不一致会影响定位。
    # 10 行分 3 份按 4/3/3 分配：前几份多一行是刻意约定，保证行数守恒。
    def test_split_lines_contiguous(self):
        lines = [str(i) for i in range(10)]
        chunks = campaign.split_lines(lines, 3, "contiguous")
        self.assertEqual([len(chunk) for chunk in chunks], [4, 3, 3])
        self.assertEqual([line for chunk in chunks for line in chunk], lines)

    # 防的回归：未启动 part 被漏出表格，总进度分母缩水，看起来像已全部完成。
    # 未启动 part 不允许伪造日志目录，但仍必须计入 total。
    def test_campaign_status_counts_unstarted_parts(self):
        camp = make_campaign(self.root)
        info = campaign.load_campaign(camp)
        payload = campaign.campaign_status(camp, info)
        # part-00 有 1 条，part-01 还有 1 条待跑。
        self.assertEqual(payload["totals"]["total"], 2)
        self.assertTrue(payload["campaign"])
        statuses = {row["id"]: row["status"] for row in payload["parts"]}
        self.assertIn("结束(exit=3)", statuses["part-00"])

        # 未启动 part 不伪造日志，但应出现在分 part 表里，且仍计入总进度。
        shutil.rmtree(camp / "parts" / "part-01")
        payload = campaign.campaign_status(camp, info)
        statuses = {row["id"]: row["status"] for row in payload["parts"]}
        self.assertEqual(statuses["part-01"], "未启动")
        self.assertEqual(payload["totals"]["total"], 2)
        self.assertEqual(payload["totals"]["done"], 1)

    # 防的回归：远端 part 心跳过期后仍显示运行中，掩盖机器掉线。
    # 故意用 another-host，确保走心跳分支而不是本机 PID 分支。
    def test_campaign_stale_heartbeat(self):
        camp = make_campaign(self.root)
        info = campaign.load_campaign(camp)
        hb = camp / "parts" / "part-00" / "heartbeat.json"
        hb.write_text(
            json.dumps(
                {
                    "host": "another-host",
                    "status": "running",
                    "heartbeat_ts": time.time() - 9999,
                }
            )
        )
        payload = campaign.campaign_status(camp, info)
        row = next(item for item in payload["parts"] if item["id"] == "part-00")
        self.assertEqual(row["status"], "失联")

    # 防的回归：跨 part 合并时同名 run 日志互相覆盖，只剩最后一台机器的现场。
    # 两个 part 各一条不同配置，合并后失败清单应去重为 2 条。
    # INDEX 需同时提供签名聚合与按 API 聚合两块，便于按算子分派排查。
    def test_campaign_collect_and_debug_merge_parts(self):
        camp = make_campaign(self.root)
        info = campaign.load_campaign(camp)
        issues = campaign.campaign_collect(camp, info)
        # 两个 part 的失败日志都要导出。
        cases = camp / "issues" / "cases"
        self.assertTrue((cases / "part-00" / "0size_accuracy" / "paddle_error").is_dir())
        self.assertTrue((cases / "part-01" / "0size_accuracy" / "paddle_error").is_dir())
        self.assertEqual(len(issues["signatures"]), 2)
        apis = {entry["api"] for entry in issues["apis"]}
        self.assertIn("paddle.relu", apis)
        self.assertIn("paddle.matmul", apis)
        index = (camp / "issues" / "INDEX.md").read_text()
        self.assertIn("按 API 聚合", index)
        self.assertIn("paddle.relu", index)

        bundle = campaign.campaign_debug(camp, info)
        self.assertIsNotNone(bundle)
        root = Path(bundle["path"])
        failures = (root / "demo" / "accuracy_0size.txt").read_text().strip().splitlines()
        self.assertEqual(len(failures), 2)
        pe = root / "demo" / "pipeline_errors" / "accuracy_0size"
        self.assertTrue((pe / "paddle_error__2000.log").exists())
        self.assertTrue((pe / "paddle_error__2001.log").exists())
        # 多个 part 的同名 run 日志必须加前缀，不能互相覆盖。
        self.assertTrue((pe / "part-00__log_20260101_000000_1.log").exists())
        self.assertTrue((pe / "part-01__log_20260101_000001_1.log").exists())
        for script in ("run_accuracy.sh", "run_case.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(root / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    # 防的回归：_run_custom_op 被截成裸名，丢失真正需要排查的算子名。
    # 普通 API 保留带前缀全名，分组时不再做二次截断。
    def test_config_api(self):
        self.assertEqual(
            report.config_api('paddle.add(Tensor(paddle.Size([0, 4]),"float32"))'), "paddle.add"
        )
        self.assertEqual(
            report.config_api(
                'paddle._C_ops._run_custom_op("fuse_weighted_swiglu_fp8_quant", True)'
            ),
            "_run_custom_op/fuse_weighted_swiglu_fp8_quant",
        )

    # 防的回归：漏跑配置因分类文件缺失被当成通过，覆盖率出现 100% 的假象。
    # 缺失清单文件名固定 <part>__<shape>__<mode>.txt，下游脚本依赖这个约定。
    def test_campaign_verify_complete_and_missing(self):
        camp = make_campaign(self.root)
        info = campaign.load_campaign(camp)
        payload = campaign.verify_campaign(camp, info)
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["missing_total"], 0)
        self.assertTrue((camp / "verify" / "verify.md").exists())

        # 删掉一个 checkpoint，应该能定位到缺失配置。
        log_dir = camp / "parts" / "part-00" / "logs" / "0size_accuracy"
        (log_dir / "checkpoint.txt").unlink()
        (log_dir / "api_config_paddle_error.txt").unlink()
        payload = campaign.verify_campaign(camp, info)
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["missing_total"], 1)
        missing_file = camp / "verify" / "missing" / "part-00__0size__accuracy.txt"
        self.assertTrue(missing_file.exists())
        self.assertIn(C_RELU, missing_file.read_text())

    # 防的回归：已结束的 part 被重复认领，两台机器同时跑同一分片。
    # exclude 代表本轮已认领集合，避免同一台机器重复拿到同一个 part。
    def test_parts_table_and_next_available(self):
        camp = make_campaign(self.root)
        info = campaign.load_campaign(camp)
        rows = campaign.parts_table(camp, info)
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["available"])  # 已结束的 part 不自动重跑
        # 删掉 part-01 后应变成可认领
        shutil.rmtree(camp / "parts" / "part-01")
        rows = campaign.parts_table(camp, info)
        self.assertTrue(rows[1]["available"])
        self.assertEqual(campaign.next_available_part(camp, info), "part-01")
        self.assertIsNone(campaign.next_available_part(camp, info, exclude={"part-01"}))

    # 防的回归：两个聚合进程同时写 issues/debug 目录，产物互相覆盖。
    # 锁必须在 with 结束时删除，否则下一次聚合会被永久拒绝。
    def test_aggregate_lock_is_exclusive(self):
        camp = make_campaign(self.root)
        with campaign.aggregate_lock(camp):
            with self.assertRaises(SystemExit):
                with campaign.aggregate_lock(camp):
                    pass
        self.assertFalse((camp / ".aggregate.lock").exists())

    # 防的回归：force reset 后残留旧锁，part 再也无法被重新认领。
    # 心跳状态要落成 reset 而不是 done，否则会被判定成已完成而跳过。
    def test_reset_part_clears_stale_lock(self):
        camp = make_campaign(self.root)
        info = campaign.load_campaign(camp)
        pdir = camp / "parts" / "part-00"
        (pdir / campaign.LOCK_FILE).write_text(json.dumps({"host": "dead-host", "pid": 1}))
        (pdir / campaign.HEARTBEAT_FILE).write_text(
            json.dumps(
                {
                    "host": "dead-host",
                    "status": "running",
                    "heartbeat_ts": time.time() - 9999,
                }
            )
        )
        campaign.reset_part(camp, info, "part-00", force=True)
        self.assertFalse((pdir / campaign.LOCK_FILE).exists())
        hb = json.loads((pdir / campaign.HEARTBEAT_FILE).read_text())
        self.assertEqual(hb["status"], "reset")


if __name__ == "__main__":
    unittest.main(verbosity=2)
