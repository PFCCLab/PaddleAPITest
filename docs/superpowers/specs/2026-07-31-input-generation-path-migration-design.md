# Input Generation Path Migration Design

**Goal:** Move `tester/api_config/input_generation` to `tester/input_generation` without retaining any compatibility layer or changing input-generation behavior.

**Architecture:** The existing input-generation package moves atomically as one directory, preserving its internal relative imports. All repository references are rewritten to import from `tester.input_generation` or the corresponding relative path from `tester`, and the old package path is removed. No forwarding modules, duplicate files, or runtime fallback imports remain.

**Tech Stack:** Python packages, `git mv`, repository-wide `rg` reference checks, `pytest`, `compileall`, Ruff, and pre-commit.

---

## Scope

- Move all files under `tester/api_config/input_generation/` to `tester/input_generation/`.
- Update imports in runtime modules, utilities, documentation, and generated test-file templates.
- Update `tester/api_config/__init__.py` and `tester/api_config/parser.py` so public exports and script-mode parsing resolve the new package.
- Remove every source reference to `tester.api_config.input_generation` and the old directory.
- Preserve package behavior, public symbols, input binding, dispatcher selection, and generated values.

## Explicit Non-Goals

- Do not add a compatibility shim for the old path.
- Do not rename or split input-generation modules.
- Do not change input-generation algorithms, registries, or TensorConfig semantics.

## Validation

- Confirm no old-path references remain outside historical logs.
- Compile all moved and dependent Python modules.
- Import `tester.input_generation`, parser, base, and direct consumers.
- Run the existing input-generation sanitizer tests and a minimal API input-generation smoke test.
- Run targeted Ruff and the repository pre-commit hooks.

