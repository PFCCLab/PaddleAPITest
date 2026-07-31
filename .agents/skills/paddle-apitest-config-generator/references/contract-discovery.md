# Contract Discovery

## Source Priority

Use sources in this order and keep the layers distinct:

1. Python signature or custom-op registration schema: call envelope.
2. Docstring and infermeta: documented dtype, rank, and shape relationships.
3. Kernel implementation: execution limits and boundary probes.
4. Existing PaddleAPITest configs: syntax and realistic seeds, not authoritative constraints.

Do not infer legality from a passing or generated config alone.

## Public Paddle APIs

Search Paddle definitions before implementation details:

```bash
rg -n "^def <api_short_name>\(" ../Paddle/python/paddle
rg -n "<api_short_name>" ../Paddle/paddle/phi/infermeta ../Paddle/paddle/phi/ops/yaml
rg -n "<fully.qualified.api>|<api_short_name>" apitest_config
```

Record every positional/keyword argument, default, accepted Python type, enum, and
optional value. Check whether the wrapper translates strings or booleans before calling
`_C_ops`; invalid wrapper enums should be classified separately from kernel-invalid
shapes.

## PaddleFleet Custom Ops

Search the registration, not only the kernel function:

```bash
rg -n "PD_BUILD_OP\(<op_name>\)" ../PaddleFleet
rg -n "<op_name>" ../PaddleFleet/packages -g '*.cu' -g '*.cc' -g '*.py'
rg -n "_run_custom_op\(\"<op_name>\"" apitest_config
```

Read the complete registration chain:

- `.Inputs(...)`: required, optional, and `paddle::Vec` Tensor inputs;
- `.Attrs(...)`: exact attribute order and C++ types;
- `.Outputs(...)`: useful for runtime expectations, not config arity;
- infer-shape/infer-dtype functions: linked dimensions and supported dtypes.

Represent calls as:

```text
paddle._C_ops._run_custom_op("<op_name>", <registered inputs>, <attrs>)
```

Count the op name in `APIConfig.args`, but not in the registered input arity.

## Contract Worksheet

Write this table in task findings before generating:

| Item | Record |
|---|---|
| API key | Fully qualified API or custom `op_name` |
| Call envelope | Ordered args and kwargs/attrs |
| Tensor kinds | Tensor, optional Tensor, Tensor list |
| Dtypes | Accepted and deliberate invalid dtypes |
| Ranks | Documented ranks and exploratory ranks |
| Shape links | Equality, ratio, broadcast, numel, derived block counts |
| Scalars/enums | Values, defaults, boundary values, invalid values |
| Zero behavior | Allowed positions and known infermeta rejection |
| Kernel-only limits | Alignment, vector width, maximum list size, early return |
| Assumptions | Anything not established by source |

## Existing Config Search

Search all config-like files, not only one model directory:

```bash
rg -l "<api_or_op_name>" apitest_config test_* -g '*.txt'
```

Parse candidates with `APIConfig`. Prefer legal 4096 seeds. Do not mutate original files.
If examples disagree with current signatures, follow current source and label the old
config as stale.
