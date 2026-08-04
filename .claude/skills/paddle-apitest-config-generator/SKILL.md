---
name: paddle-apitest-config-generator
description: Generate, expand, split, and validate PaddleAPITest API configuration cases from Paddle/PaddleFleet API names or existing seed config files. Use for creating broad 4096, 1M, and 0size case sets; increasing sampled shapes and parameter combinations; modeling public APIs or paddle._C_ops._run_custom_op schemas; preserving linked Tensor constraints; adding edge or intentionally invalid stability probes; and producing per-API config files with manifests.
---

# Generate PaddleAPITest Configs

Build reproducible config generators from API contracts. Treat Python signatures and
custom-op registration schemas as the call-envelope source of truth. Keep CUDA-only
limits as edge labels unless the user explicitly asks to filter by kernel support.

## Resolve The Request

1. Work from the target PaddleAPITest checkout. The bundled tools themselves are
   self-contained and require no generator or parser from that checkout.
2. Record the requested API names, seed config paths, specs, case count, output path,
   and whether invalid probes are wanted.
3. Default to `4096`, `1M`, and `0size`, 512 cases per API/spec, separate API
   directories, and a 2:1:1 valid/edge/invalid split for API-aware generation.
4. Never modify seed files or unrelated generated outputs.

Choose one path:

- **Seed configs supplied:** run the structural seed expander for an initial corpus,
  then add API-aware mutations when broader parameter coverage is needed.
- **Only API names supplied:** discover the contract and existing examples first.
  Use examples as seeds when available; otherwise build `APIConfig` objects directly.

## Discover The Contract

Read [contract-discovery.md](references/contract-discovery.md) before modeling a new
API. Capture:

- positional and keyword arguments, defaults, Python types, optional/Vec inputs;
- accepted dtypes, ranks, enum values, and scalar attributes;
- equality, ratio, numel, and shape-derived relationships across Tensor arguments;
- distinctions between API/schema constraints and infermeta/kernel constraints.

For `_run_custom_op`, model each `op_name` independently. Include the op name as the
first positional argument and match `PD_BUILD_OP(...).Inputs(...).Attrs(...)` exactly.

## Expand Seed Configs

Use the bundled script for deterministic shape expansion:

```bash
python .claude/skills/paddle-apitest-config-generator/scripts/expand_from_seeds.py \
  --api paddle.some_api \
  --seed-file apitest_config/path/to/seeds.txt \
  --output-dir generated_api_configs \
  --cases-per-spec 512
```

Repeat `--api` and `--seed-file` as needed. Omit `--api` to expand every API found in
the seed files. The script:

- parses and clones `APIConfig` objects instead of editing strings;
- changes the selected anchor dimension across all equal linked dimensions;
- produces separate `4096.txt`, `1M.txt`, and `0size.txt` files per API;
- ensures each 0size case contains a zero dimension;
- writes a per-API `manifest.jsonl` and root `index.json`;
- rejects duplicate or non-round-trip output.

Seed expansion preserves call structure but cannot infer every semantic relationship.
Do not label its output contract-valid without checking the API contract. Add an
API-aware builder for ratios such as `[N, H]` versus `[N, 2H]`, derived scale shapes,
Tensor lists, optional inputs, enums, and attributes.

## Build API-Aware Cases

Read [case-design.md](references/case-design.md) and
[generator-pattern.md](references/generator-pattern.md). Implement a task-local
generator that imports `scripts/apitest_config_utils.py` and returns structured
`APIConfig` objects plus category and violation metadata.

Generate dimensions from explicit boundary tables. Cover ranks, zero positions,
dtypes, optional values, Tensor-list lengths, bool combinations, numeric attributes,
enum values, matching relationships, and deliberate mismatches. Guarantee uniqueness
with a dimension or numeric attribute that retains the case index; never rely only on
short modulo cycles.

Classify cases as:

- `contract_valid`: conforms to the documented API/custom-op contract;
- `edge`: preserves the call envelope but probes infermeta/kernel boundaries;
- `intentionally_invalid`: deliberately violates a named dtype, enum, rank, shape,
  or cross-input relationship.

Record every edge/invalid reason in `violations`. When the contract is uncertain, use
`edge` and document the assumption rather than claiming validity.

## Validate Outputs

Run validation after every generation change:

```bash
python .claude/skills/paddle-apitest-config-generator/scripts/validate_case_tree.py \
  generated_api_configs \
  --expected-per-file 512 \
  --require-zero \
  --materialize-zero
```

Require all of the following:

- every line parses and is `APIConfig` round-trip stable;
- every API/spec file has the expected count and no duplicates;
- config API/op name matches its manifest record;
- manifest records correspond to config lines in order;
- each non-empty 0size case contains a zero-dimension Tensor;
- zero-dimension Tensors materialize through the bundled Tensor model when requested.

Do not allocate nonzero 1M tensors during static validation. Do not run CUDA kernels
unless the user requests runtime testing and the environment supports the op.

The generator does not import any task-specific generator or generated artifact. The
validator uses the bundled parser by default; inside PaddleAPITest, also run an
authoritative compatibility pass with:

```bash
python .claude/skills/paddle-apitest-config-generator/scripts/validate_case_tree.py \
  generated_api_configs \
  --official-analyzer \
  --apitest-root .
```

This optional pass uses PaddleAPITest's core `config_analyzer`, not a task-specific
generation script or reference implementation.

## Deliver

Report the generator path, output root, per-API/spec counts, category counts,
validation results, and any runtime validation not performed. Keep old combined files
untouched unless the user explicitly asks to remove them.
