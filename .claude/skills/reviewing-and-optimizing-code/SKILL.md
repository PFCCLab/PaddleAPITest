---
name: reviewing-and-optimizing-code
description: Use when reviewing or simplifying mature code paths, especially when correctness must come before simplification, readability, helper removal, logging ownership, or global-state cleanup.
---

# Reviewing and Optimizing Code

## Core Principle

Correctness first. A cleaner shape that changes behavior, error handling, cleanup, or protocol
contracts is a regression. Once behavior is sound, prefer the simplest direct structure that
makes ownership obvious.

## Review Phases

### 1. Correctness

- Check behavior, exit codes, cleanup, and invariants first.
- Verify failure paths before style changes.
- Look for hidden protocol changes between producer and consumer code.
- Treat tests and call sites as evidence, not assumptions.

### 2. Ownership

- Find the one clear owner for each policy.
- Avoid duplicated policy across main flow, helpers, and logging layers.
- Keep boundaries explicit when a function owns a protocol or state transition.

### 3. Simplification

- Remove helpers only when the caller stays clearer without them.
- Delete wrappers that only forward one call, one condition, or one append.
- Keep helpers that encode real policy, a protocol boundary, or shared behavior.
- Prefer direct code over compatibility scaffolding unless an external contract depends on it.

### 4. Organization

- Group globals by purpose, not by accident of history.
- Prefer short comments that explain intent, not history.
- Keep logging, banner, and report formatting in the layer that already owns output shape.
- Watch for long relay chains where one function only re-exports another decision.

### 5. Verification

- Re-run the narrowest tests that prove the behavior still holds.
- Compile or type-check when structure changes touch imports or signatures.
- If a change removes a layer, verify the live call graph still reaches the same owner.

## Optimization Rules

- Simplify the main path before polishing helpers.
- Inline one-off wrappers unless they form a stable boundary.
- Move shared policy upward when multiple callers need the same rule.
- Do not split files unless the boundary is genuinely hard to understand in one place.
- Do not preserve dead compatibility shims just because they are familiar.

## What To Look For

- over-thin helpers
- duplicated branches with one owner missing
- long pass-through chains
- scattered global state
- misplaced output formatting
- comments that explain old history instead of current intent
- compatibility code without a live consumer

## Review Output

- Findings first, ordered by severity.
- Cite file and line numbers.
- Separate correctness issues from style issues.
- If something is only an optimization preference, say so clearly.
- End with a short judgment on whether the code is ready.
