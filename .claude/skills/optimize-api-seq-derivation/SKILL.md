---
name: optimize-api-seq-derivation
description: Improve and validate PaddleAPITest sequence-length configuration derivation across different models. Use when test_pipeline/config_preprocess/derive_api_seq.py produces low accuracy, when hard-coded model constants must be removed, or when regular and MoE configurations need model-adaptive derivation and verification against known origin configs.
---

# Optimize API Seq Derivation

Use this skill to make `test_pipeline/config_preprocess/derive_api_seq.py` generalize from two
origin configurations at different sequence lengths to a larger target sequence
length. The result must be driven by source configuration structure and validated
against a target origin configuration, not tuned to one model's observed output.

## Ground Rules

- Work only with origin configs when measuring derivation accuracy. The target
  origin config is a validation oracle, not a source for hard-coded rules.
- Accuracy means the multiset overlap reported by
  `test_pipeline/config_preprocess/verify_api_seq.py`. Do not replace it with positional
  matching, runtime test logs, or a visual comparison.
- Report overall, deterministic, and MoE-class accuracy separately. The
  deterministic class is the primary optimization target; MoE routing is
  data-dependent and has a lower theoretical ceiling.
- Preserve unrelated worktree changes. Inspect `git status` before editing and
  keep generated outputs in a temporary directory unless the user requests a
  checked-in artifact.
- Never solve a model-specific miss by adding its hidden size, expert count,
  top-k, padding, layer count, or a target sequence length as a literal rule.

## Locate Inputs

First identify the actual implementation and matching files in the current
checkout:

```bash
rg --files | rg '(^|/)(derive_api_seq|verify_api_seq)\.py$'
find <model-origin-dir> -type f -name 'api_config_*.txt' | sort
```

Use two configs from the same model and compatible generation setup as sources:

- `small`: known smaller sequence length;
- `large`: known larger sequence length and the output skeleton;
- `target`: a known origin config used only for verification.

Confirm the sequence values from filenames or configuration content. Pass them
explicitly with `--seq-small` and `--seq-large` when they are not the script
defaults. Do not silently mix models, preprocessed configs, `paddleonly` files,
or configs produced by a different pipeline stage.

## Baseline First

Run an unmodified or current version at every available validation target. Use
absolute paths when the checkout or model directory is outside the current
directory:

```bash
python test_pipeline/config_preprocess/derive_api_seq.py <target> \
  --seq-small <small-seq> --seq-large <large-seq> \
  --small <small-origin> --large <large-origin> \
  -o /tmp/derived_<target>.txt

python test_pipeline/config_preprocess/verify_api_seq.py \
  -d /tmp/derived_<target>.txt -r <target-origin>
```

Record the three percentages and the deterministic Top mismatch lines before
changing code. Validate at least two target lengths when available, for example
`4096` and `8192`; include `1048576` for a 1M request. A rule is useful only if
it improves the intended class without materially regressing another target.

## Read The Derivation Pipeline

Before adding a heuristic, trace the current script in this order:

1. `numbers()` and `signature()` define the line structure and protected dtype
   regions. Do not parse configuration lines with an unscoped global digit
   replacement.
2. `build_value_map()` learns position-specific changes between the two source
   configs. Constants should remain unchanged; only stable, unambiguous mappings
   belong in the deterministic path.
3. `derive_line()` applies the learned mapping to the large skeleton and is the
   right place for generic line-level shape transformations.
4. MoE reconstruction is a separate pass. It may replace dispatch-related values
   after the generic pass, but it must not rewrite unrelated deterministic lines.
5. `verify_api_seq.py` defines the class split and the acceptance metric. Keep
   classifier and optimizer assumptions aligned with it.

## Generalize Deterministic Shapes

Prefer relationships learned from the small/large pair over literal dimensions.
For each candidate value, establish all of the following before rewriting it:

- the same structural signature and numeric position occur in both sources;
- the value changes consistently with sequence length;
- occurrence counts remain compatible, so a data-dependent list is not mistaken
  for a shape transformation;
- the inferred transformation is valid for every source pair available.

Use an affine or ratio mapping only when its direction and rounding are clear.
For common shape families, infer the multiplier or offset from the source values:

- direct sequence dimensions `S`;
- packed dimensions such as `S / block`, `S * block`, or `S + offset`;
- repeated tensor shapes and slice boundaries;
- reshape, cast, concat, zeros/full, transpose, matmul, comparison, and softmax
  metadata that carries one of those dimensions.

When a relationship cannot be inferred from the sources, leave the value alone
and record it as an unresolved mismatch. A narrow postprocessor is acceptable
when it matches an operator family and a source-derived shape relation; it must
not depend on a model's fixed hidden size, head count, expert count, or one
specific target.

Protect values that look like dtypes, enum attributes, vocabulary/model
dimensions, ranks, axis identifiers, and unrelated numeric literals. Preserve
negative shape sentinels such as `-1` unless the source evidence proves they are
sequence-dependent.

## Make MoE Model-Adaptive

MoE can be improved, but exact routing rows are input-data-dependent. Optimize
the structural consequences of routing and keep the uncertainty explicit.

Infer the profile from the source skeleton/configs rather than constants:

- `hidden` from the Tensor shapes around `moe_permute` or `moe_unpermute`;
- `top_k` from the routing/index dimensions and call attributes;
- `num_experts` from the operator attributes or expert-count lists;
- `padding_alignment` from the operator call;
- FP8/BF16 and custom-op variants from their actual call signatures.

For a target ratio `target / large`:

1. Scale token/input counts using the chosen rounding policy, consistently for
   the same operator family.
2. Scale the total `tokens_per_expert` and redistribute it across experts using
   the source distribution as weights. Use deterministic largest-remainder
   rounding so the sum is exact.
3. Recompute each padded expert count with the inferred padding alignment, then
   recompute the padded buffer total. Do not scale an already padded total when
   the per-expert list is available.
4. Propagate the learned old-to-new values only to related dispatch, unpermute,
   fused projection, and sequence-chunk lines. Match exact integer tokens and
   structural context to avoid changing hidden dimensions or attributes that
   merely happen to have the same number.
5. Handle multiple MoE call variants through parsed fields, not one regex that
   assumes a particular dtype, argument order, or number of experts.

Do not claim that MoE routing accuracy can always reach 90%. The actionable
acceptance target is deterministic accuracy >= 90% and no regression in other
targets; MoE accuracy should be reported as a separate diagnostic. If a model's
MoE output is deterministic across source/target generation, use that evidence
to extend the parser, but do not encode the observed expert route list itself.

## Iterate From Mismatches

After each focused change, regenerate and verify the same targets. Use the
deterministic Top mismatch section to classify misses:

- wrong value with the right line shape: fix the inferred transformation or
  rounding policy;
- right value in the wrong count: check list multiplicity and whether the line
  was incorrectly treated as deterministic;
- missing related shape line: add a context-aware operator-family rule;
- mismatch only in dispatch/index data: keep it in the MoE path and avoid
  weakening deterministic mappings.

Add one rule family at a time. Compare before/after counters and line counts,
and retain a rule only when it improves the target metric across at least two
targets or across a held-out model. Never use the target file to manufacture a
mapping that is absent from both source files.

## Regression Checks

Run these checks after editing:

```bash
python -m py_compile test_pipeline/config_preprocess/derive_api_seq.py
python test_pipeline/config_preprocess/verify_api_seq.py \
  -d /tmp/derived_<target>.txt -r <target-origin>
git diff --check
```

Also check that:

- derived output line count equals the large skeleton line count;
- source files are unchanged and output contains no accidental blank records;
- constants inferred from the source model remain stable;
- FP8/BF16 MoE variants and ordinary non-MoE configs both pass;
- at least one target not used while authoring a rule remains non-regressed;
- no rule contains a model-specific literal that could have been inferred from
  the source configuration instead.

## Delivery

Report the modified script, source and target config paths, exact derive and
verify commands, and before/after overall, deterministic, and MoE percentages.
State clearly whether the 90% target applies to overall or deterministic
accuracy. Mention any unavailable target, runtime test, or model variant; do not
substitute test logs for the requested origin-config verification.
