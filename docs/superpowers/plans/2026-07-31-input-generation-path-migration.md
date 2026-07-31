# Input Generation Path Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `tester/api_config/input_generation` to `tester/input_generation` and update every repository reference without retaining the old import path.

**Architecture:** Move the complete package as one unit so its internal relative imports remain unchanged. Rewrite consumers according to their package location: `tester` modules use `.input_generation`, `tester.api_config` modules use `..input_generation`, and external tools use `tester.input_generation`. Parser script mode gets an explicit repository-root import path.

**Tech Stack:** Python packages, Git rename detection, `rg`, `pytest`, `compileall`, Ruff, and pre-commit.

---

### Task 1: Move The Package

**Files:**
- Move: `tester/api_config/input_generation/` to `tester/input_generation/`

- [ ] **Step 1: Move all package files atomically**

```bash
git mv tester/api_config/input_generation tester/input_generation
```

Expected: all ten package files appear under `tester/input_generation/`, and `tester/api_config/input_generation/` no longer exists.

- [ ] **Step 2: Confirm internal imports remain package-relative**

```bash
rg -n '^(from \.|import )' tester/input_generation --glob '*.py'
```

Expected: internal imports refer only to sibling modules and do not mention `tester.api_config.input_generation`.

### Task 2: Rewrite Runtime And Tool References

**Files:**
- Modify: `tester/base.py`
- Modify: `tester/paddle_gpu_performance.py`
- Modify: `tester/paddle_torch_gpu_performance.py`
- Modify: `tester/torch_gpu_performance.py`
- Modify: `tester/test_file_generator.py`
- Modify: `tester/api_config/__init__.py`
- Modify: `tester/api_config/parser.py`
- Modify: `tester/api_config/bittensor_config_filter.py`
- Modify: `tester/api_config/performance_numel_stat.py`
- Modify: `tester/api_config/performance_numel_stat2.py`
- Modify: `tester/api_config/to_0_size_config.py`
- Modify: `tester/api_config/to_big_size_config.py`
- Modify: `tester/api_config/to_prof_size_config.py`
- Modify: `tester/api_config/big_and_0size/to_0_size_config.py`
- Modify: `tools/qa_test/to_0_size_config.py`
- Modify: `tools/prof/paddleapitest_matmul_heatmap.py`
- Modify: `tools/regression/collect_configs.py`
- Modify: `tools/normalize_origin_api_config.py`

- [ ] **Step 1: Rewrite imports by package depth**

Use these exact replacements:

```text
tester/base.py and tester/*_performance.py:
  .api_config.input_generation -> .input_generation

tester/api_config/*.py and tester/api_config/big_and_0size/*.py:
  tester.api_config.input_generation -> tester.input_generation
  .input_generation -> ..input_generation

tools and generated source strings:
  tester.api_config.input_generation -> tester.input_generation
```

- [ ] **Step 2: Update parser package and script-mode imports**

Package mode must use:

```python
from ..input_generation.tensor_config import TensorConfig
```

Script mode must add the repository root to `sys.path` and import:

```python
from tester.input_generation.tensor_config import TensorConfig
```

- [ ] **Step 3: Update generated test-file templates and documentation strings**

Every emitted import string must use `tester.input_generation.tensor_config` so generated repro files do not reference the removed path.

### Task 3: Remove And Audit The Old Path

**Files:**
- Modify: all files returned by the repository-wide old-path search

- [ ] **Step 1: Search for stale source references**

```bash
rg -n 'tester\.api_config\.input_generation|tester/api_config/input_generation|\.api_config\.input_generation' . --glob '!report/**' --glob '!*.log'
```

Expected: no matches.

- [ ] **Step 2: Check package tree and Git rename status**

```bash
test ! -e tester/api_config/input_generation
test -d tester/input_generation
git status --short
```

Expected: old directory absent, new directory present, and Git recognizes the package as renames where content is unchanged.

### Task 4: Validate Behavior And Tooling

**Files:**
- Test: moved package and all direct consumers

- [ ] **Step 1: Compile moved and dependent modules**

```bash
python -m compileall -q tester/input_generation tester/api_config tester/base.py tester/*performance.py tools
```

Expected: exit code 0.

- [ ] **Step 2: Run import and API input-generation smoke checks**

```bash
PYTHONPATH=. python - <<'PY'
from tester.api_config.parser import APIConfig
from tester.base import APITestBase
from tester.input_generation import registry

config = APIConfig('paddle.add(Tensor([2], "float32"), Tensor([2], "float32"))')
base = APITestBase(config)
assert base.ana_api_info()
assert base.gen_numpy_input()
assert base.gen_paddle_input()
assert base.gen_torch_input()
base.clear_tensor()
print("input_generation_path_smoke_ok")
PY
```

Expected: prints `input_generation_path_smoke_ok`.

- [ ] **Step 3: Run targeted tests and static checks**

```bash
pytest -q tester/api_config/test_sanitizer_output.py
ruff check tester/input_generation tester/api_config tester/base.py tester/*performance.py tools
pre-commit run --all-files
```

Expected: targeted tests pass, Ruff passes, and pre-commit passes.

- [ ] **Step 4: Review the final diff and working tree**

```bash
git diff --check
git diff --stat
git status --short --branch
```

Expected: no whitespace errors, only the planned path migration is present, and no unresolved or untracked migration artifacts remain.
