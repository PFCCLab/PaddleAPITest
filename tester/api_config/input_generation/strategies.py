"""Compatibility imports for the pre-simplification strategy module.

New code should import pure generators from ``value_generators`` and API mappings
from ``registry``.
"""

from __future__ import annotations

from .registry import INPUT_GENERATION_RULES
from .value_generators import (
    LEGACY_NUMPY_RNG,
    LegacyNumpyRNG,
)
from .value_generators import (
    generate_default as generate_default_numpy,
)
from .value_generators import (
    generate_nonzero as generate_nonzero_numpy,
)
from .value_generators import (
    generate_uniform as generate_uniform_range_numpy,
)
from .value_generators import (
    generate_unit_interval as generate_probability_numpy,
)
from .value_generators import (
    generate_unit_interval as generate_unit_interval_numpy,
)
from .value_generators import (
    generation_dtype as default_generation_dtype,
)

DEFAULT_STRATEGY_ID = "strategy.default.v1"
NONZERO_STRATEGY_ID = "strategy.nonzero.v1"
PROBABILITY_STRATEGY_ID = "strategy.probability.v1"
UNIT_INTERVAL_STRATEGY_ID = "strategy.unit_interval.v1"
NONNEGATIVE_RANGE_STRATEGY_ID = "strategy.range.nonnegative.v1"
POSITIVE_RANGE_STRATEGY_ID = "strategy.range.positive.v1"

(
    DEFAULT_CASE_RULE,
    NONZERO_CASE_RULE,
    PROBABILITY_CASE_RULE,
    UNIT_INTERVAL_CASE_RULE,
    NONNEGATIVE_RANGE_CASE_RULE,
    POSITIVE_RANGE_CASE_RULE,
) = INPUT_GENERATION_RULES

__all__ = [
    "DEFAULT_CASE_RULE",
    "DEFAULT_STRATEGY_ID",
    "LEGACY_NUMPY_RNG",
    "LegacyNumpyRNG",
    "NONNEGATIVE_RANGE_CASE_RULE",
    "NONNEGATIVE_RANGE_STRATEGY_ID",
    "NONZERO_CASE_RULE",
    "NONZERO_STRATEGY_ID",
    "POSITIVE_RANGE_CASE_RULE",
    "POSITIVE_RANGE_STRATEGY_ID",
    "PROBABILITY_CASE_RULE",
    "PROBABILITY_STRATEGY_ID",
    "UNIT_INTERVAL_CASE_RULE",
    "UNIT_INTERVAL_STRATEGY_ID",
    "default_generation_dtype",
    "generate_default_numpy",
    "generate_nonzero_numpy",
    "generate_probability_numpy",
    "generate_uniform_range_numpy",
    "generate_unit_interval_numpy",
]
