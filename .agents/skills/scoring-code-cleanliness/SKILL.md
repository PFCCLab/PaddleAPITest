---
name: scoring-code-cleanliness
description: Use when judging code quality, cleanliness, or maintainability and a concise, file-agnostic scoring rubric is needed.
---

# Scoring Code Cleanliness

## Overview

This is a rubric for scoring mature code after correctness has been checked. It is meant for
plain review judgments, not for implementation guidance.

## Scoring Order

Always score in this order:

1. Correctness and safety
2. Ownership and boundaries
3. Simplification and duplication
4. Organization and naming
5. Readability and local polish

If correctness is weak, the overall score should stay low even when the code looks neat.

## Score Bands

Use a 10-point scale:

- 9.0-10.0: Excellent. Clean ownership, minimal noise, clear boundaries, no obvious cleanup left.
- 8.0-8.9: Strong. Good structure with a few remaining simplifications or heavy areas.
- 7.0-7.9: Acceptable. Works and is readable, but still has noticeable clutter or uneven boundaries.
- 6.0-6.9: Fair. Correct enough, but repetitive, tangled, or harder to extend than it should be.
- Below 6.0: Needs work. Correctness, structure, or maintainability is still too shaky.

## Review Checklist

- Is behavior stable and verified?
- Is each policy owned by one clear place?
- Are there thin wrappers, relay chains, or duplicated branches?
- Are globals grouped and named by purpose?
- Are comments current, not historical?
- Does the main path read directly?
- Is logging/report formatting owned by the output layer?
- Are helper functions small because they matter, not because they were left behind?

## Common Deductions

- Hidden protocol coupling
- Over-thin helpers without a real boundary
- Duplicate policy across layers
- Scattered global state
- Overloaded entry points
- Output formatting in the wrong layer
- Comments that explain history instead of intent

## Final Judgment Template

Use one short sentence:

- `Correctness is solid, structure is mostly clean, and the remaining issues are polish-level.`
- `Correctness is acceptable, but the code still carries too much structural noise.`
- `The code is not ready yet because the main ownership or behavior boundary is still unclear.`
