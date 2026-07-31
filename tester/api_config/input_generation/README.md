# Input Generation Runtime

This package contains runtime code only. Offline inventories, fixed-seed baselines,
and verification commands live in `tools/input_generation_governance/`.

The detailed migration and architecture rationale is documented in
[`DESIGN.zh-CN.md`](DESIGN.zh-CN.md).

## Runtime Flow

```text
APIConfig (tester.api_config.parser)
  -> APITestBase.gen_numpy_input
  -> dispatcher
  -> decorator-registered rule
  -> TensorConfig materialization
```

## Modules

- `tensor_config.py`: tensor configuration, cache, and framework materialization.
- `input_generator.py`: legacy per-op NumPy generation archive; runtime no longer
  dispatches through it.
- `dispatcher.py`: v2 rule dispatch and fail-fast handling for missing or blocked rules.
- `model.py` / `binding.py`: argument identity and signature binding for v2.
- `value_generators.py`: pure API-independent NumPy value generation.
- `registry.py`: `@rules.register` decorator rules, `RuleCase`, API lookup, and
  duplicate detection.
- `strategies.py`: compatibility imports for older tooling only; new code should import
  generators from `value_generators.py` and API mappings from `registry.py`.
- `telemetry.py`: opt-in context-local legacy and dispatch events.

The decorator registry contains only explicit, verified API allowlists; it never
uses a catch-all default rule. The current v2 allowlists cover the default
generation family (`paddle.add`, `paddle.logical_not`, `paddle.concat`), the legacy non-zero
family, `paddle.bernoulli`, `paddle.standard_gamma`, `paddle.poisson`,
fixed-range `sqrt/rsqrt`, and additional single-parameter value-domain rules
for creation, elementwise, shape-tensor, and selected `nn.functional` APIs.
Each migrated API's initialization flow is expressed in its decorated rule
function; shared dtype/shape/RNG details stay in `value_generators.py`.
Rule bodies should keep API-specific behavior local to the decorated function
or a nested helper. They should use `RuleCase` wrappers such as
`case.value_domain()`, `case.random()`, `case.randint()`, and `case.array()`
instead of directly calling NumPy or the legacy RNG; this keeps the rule syntax
stable when RNG ownership is moved off global NumPy. Rule bodies should read
API arguments through `case.arg()`, `case.kwarg()`, and `case.has_kwarg()`
rather than reaching into the raw case object. Rule-local value helpers take
only the current binding and use the enclosing `case` for everything else.
`generate_by_parameter()` is retained for simple parameter dispatch and legacy
traversal order, not as a place to hide one-off API logic behind module-level
helpers.
Current coverage is 114 decorator rules and 209 explicit APIs. Migrated rules
use the case-local `CaseNumpyRNG` facade, which owns a legacy `RandomState`
copy and commits it after successful rule generation. GPU and cached generation
are rule-gated with `allow_gpu` / `allow_cached`; unsupported rules fail before
consuming value-generator RNG. Configuration tools should import `tester.api_config.parser` and
`tester.api_config.input_generation.tensor_config` directly.
