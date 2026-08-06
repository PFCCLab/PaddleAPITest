# API-Aware Generator Pattern

Use this pattern after completing the contract worksheet. Keep API-specific relationships
inside `build_case`; reuse the skill utility for parsing, serialization, and output.

```python
#!/usr/bin/env python3
from __future__ import annotations

import collections
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
    SKILL_SCRIPTS = (
        candidate / ".claude/skills/paddle-apitest-config-generator/scripts"
    )
    if SKILL_SCRIPTS.is_dir():
        APITEST_ROOT = candidate
        break
else:
    raise FileNotFoundError("paddle-apitest-config-generator skill is not installed")
sys.path.insert(0, str(SKILL_SCRIPTS))

from apitest_config_utils import (  # noqa: E402
    APIConfig,
    CaseRecord,
    SPECS,
    TensorConfig,
    case_category,
    write_case_tree,
)

API_NAME = "paddle.some_api"


def make_config(args, kwargs=None):
    config = APIConfig(f"{API_NAME}()")
    config.args = args
    config.kwargs = collections.OrderedDict(kwargs or {})
    return config


def build_case(spec: str, index: int) -> CaseRecord:
    category = case_category(index)
    violations = []

    # Derive linked shapes together. Do not mutate them independently.
    rows = 0 if spec == "0size" else 4096 + index * 128
    hidden = (128, 256, 512, 1024)[index % 4]
    x = TensorConfig([rows, hidden * 2], "bfloat16")
    scale = TensorConfig([rows, 1], "float32")

    if category == "edge":
        hidden = 127 + index
        x.shape[-1] = hidden * 2
        violations.append("non_aligned_hidden")
    elif category == "intentionally_invalid":
        scale.shape[0] = rows + 1
        violations.append("scale_row_mismatch")

    return CaseRecord(
        spec=spec,
        api=API_NAME,
        index=index,
        category=category,
        violations=tuple(violations),
        config=make_config([x, scale]),
    )


records = [
    build_case(spec, index)
    for spec in SPECS
    for index in range(512)
]
write_case_tree(Path("generated_api_configs"), records)
```

For custom ops, create configs with:

```python
config = APIConfig("paddle._C_ops._run_custom_op()")
config.args = [op_name, *registered_inputs, *registered_attrs]
config.kwargs = collections.OrderedDict()
```

Before writing, add a small call-envelope validator that checks API name, arity, Tensor
versus list/optional positions, keyword names, and Python attribute types. Reparse each
serialized config; do not validate by string shape alone.
