# Case Design And Validation

## Case Layers

Use three explicit layers for API-aware generation:

| Category | Purpose | Default share |
|---|---|---:|
| `contract_valid` | Match the established API/schema contract | 50% |
| `edge` | Preserve the call envelope while probing uncertain infermeta/kernel boundaries | 25% |
| `intentionally_invalid` | Violate one named rule for stability/error handling | 25% |

Never mix an unexplained invalid case into `contract_valid`. Store violation names in
the manifest, for example `scale_row_mismatch`, `unsupported_quant_method`, or
`heterogeneous_tensor_shapes`.

## Shape Profiles

Sample boundary tables rather than a single arithmetic sequence.

Regular/4096 anchors:

```text
1, 2, 3, 7, 15, 31, 63, 127, 128, 129,
255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 16384
```

1M anchors:

```text
999872, 999936, 999999, 1000000, 1000001, 1000064, 1000128,
1048447, 1048448, 1048575, 1048576, 1048577, 1048704
```

Widths:

```text
1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64,
127, 128, 129, 255, 256, 257, 511, 512, 513,
1023, 1024, 1025, 2048, 3072, 4096, 5120, 7168, 8192, 16384
```

Add API-specific aligned sets for 128/256/512 block sizes. Cover 1D through 4D when
the API or stability probe makes those ranks meaningful.

For 0size, place zero in the first, intermediate, and last dimensions. Preserve at
least one nonzero case marker when fixed zero shapes would otherwise repeat. Every
non-empty 0size case must contain a Tensor with a zero dimension.

## Parameter Coverage

Include applicable combinations:

- all meaningful bool combinations;
- scalar, `[N]`, `[N, 1]`, transposed, and derived scale shapes;
- optional Tensor as `None` and each accepted Tensor form;
- Tensor-list counts such as 1, 2, 3, 4, 7, 8, and 16;
- homogeneous and deliberately heterogeneous lists;
- supported dtypes plus named unsupported dtypes;
- default, boundary, negative, zero, and large numeric attributes;
- every supported enum plus non-empty unsupported enum values;
- matching and mismatching equality, ratio, numel, and block-count relationships.

Use non-empty invalid strings because some historical analyzer revisions tokenize empty
string kwargs differently.

## Uniqueness

Do not assume `index % n` combinations remain unique at 512 cases. Encode the index in
at least one unbounded field:

- a nonzero dimension while another dimension remains zero;
- a numeric attribute such as epsilon or clamp value;
- a Tensor-list count or a marker Tensor for list edge cases.

Run duplicate detection across the full requested count before writing final files.

## Output Layout

Keep API outputs separate:

```text
output_root/
  index.json
  api_slug/
    4096.txt
    1M.txt
    0size.txt
    manifest.jsonl
```

Each manifest record should contain `api`, `spec`, `index`, `category`, `violations`,
`source`, and the exact serialized `config`.

## Static Validation

Validate without allocating nonzero 1M arrays:

1. Parse every line with `APIConfig`.
2. Require `str(APIConfig(line)) == line` for generated output.
3. Check exact per-file counts and file-local uniqueness.
4. Check API/op name and manifest correspondence.
5. Check category counts.
6. For 0size, materialize only zero-dimension TensorConfigs and require `array.size == 0`.

Runtime CUDA validation is a separate, optional phase. Report it as not run when the
environment or custom-op registration is unavailable.
