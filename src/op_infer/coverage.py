"""Fixed-tolerance coverage planning through SciPy's binomial probabilities.

This host planning boundary covers an
explicit family that rejects on any coverage violation, with unrestricted
dependence between rules. It neither runs inference nor qualifies other gates.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Annotated, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import __version__ as scipy_version
from scipy.stats import binom

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

__all__ = [
    "CoverageRule",
    "coverage_operating_characteristics",
    "plan_coverage_replicates",
]


class CoverageRule(BaseModel):
    """One inclusive empirical-coverage range, declared before a fixed bank.

    ``name`` identifies an estimand/nominal-level criterion. Its calibrated
    coverage probability is ``nominal``; this assumption needs independent
    replicates, an appropriate PIT/atom rule and sufficient posterior resolution.
    Bounds may be asymmetric but must contain the nominal probability.

    Examples:
        >>> rule = CoverageRule(
        ...     name="theta_50", nominal=0.5, lower=0.45, upper=0.55
        ... )
        >>> rule.lower, rule.upper
        (0.45, 0.55)
    """

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )

    name: Annotated[str, Field(min_length=1, pattern=r"\S")]
    nominal: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    lower: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    upper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if not self.lower <= self.nominal <= self.upper:
            msg = "Coverage bounds must contain their nominal probability"
            raise ValueError(msg)
        return self


def _count(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 1 <= value < 2**53
    ):
        msg = f"{name} must be a positive integer below 2**53"
        raise ValueError(msg)
    return int(value)


def _rules(rules: Sequence[CoverageRule]) -> tuple[CoverageRule, ...]:
    result = tuple(CoverageRule.model_validate(rule) for rule in rules)
    if not result or len({rule.name for rule in result}) != len(result):
        msg = "Supply a nonempty family of distinct named coverage rules"
        raise ValueError(msg)
    return result


def _accepted_counts(
    rule: CoverageRule, counts: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    # Adjust multiplication rounding to match inclusive float comparisons k/n.
    lower = np.ceil(rule.lower * counts).astype(np.int64)
    upper = np.floor(rule.upper * counts).astype(np.int64)
    lower -= (lower > 0) & ((lower - 1) / counts >= rule.lower)
    lower += lower / counts < rule.lower
    upper += (upper < counts) & ((upper + 1) / counts <= rule.upper)
    upper -= upper / counts > rule.upper
    return lower, upper


def _rejection(
    rule: CoverageRule, counts: NDArray[np.int64], probability: float
) -> NDArray[np.float64]:
    lower, upper = _accepted_counts(rule, counts)
    return np.asarray(
        np.clip(
            binom.cdf(lower - 1, counts, probability)
            + binom.sf(upper, counts, probability),
            0,
            1,
        ),
        dtype=np.float64,
    )


def coverage_operating_characteristics(
    rules: Sequence[CoverageRule],
    *,
    n_replicates: int,
    probabilities: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed coverage family using exact binomial tail probabilities.

    All criteria must pass. Replicates are independent
    within each rule; dependence between rules is unrestricted. The family
    rejection probability lies between the largest marginal probability and
    their sum (capped at one). It is not generally equal to either bound.

    Args:
        rules: Complete prespecified criterion family, with unique names.
        n_replicates: Fixed number of usable, independent replicates, not attempts.
        probabilities: Actual coverage probability for every rule under an
            alternative. If absent, use the nominal calibrated probabilities.

    Returns:
        JSON-compatible marginal rejection probabilities and family bounds,
        inclusive accepted counts, criteria, assumptions and implementation version.
        Under calibration these describe false failures; under a specified
        alternative they bound rejection power. No sampler/gate certification.

    Raises:
        ValueError: Counts, rules or complete probability mapping are invalid.

    Examples:
        >>> rule = CoverageRule(name="half", nominal=0.5, lower=0.25, upper=0.75)
        >>> result = coverage_operating_characteristics((rule,), n_replicates=4)
        >>> result["criteria"][0]["rejection_probability"]
        0.125
        >>> result["family_rejection_probability"]
        {'lower': 0.125, 'upper': 0.125}
    """
    family = _rules(rules)
    count = _count(n_replicates, "n_replicates")
    supplied = (
        {rule.name: rule.nominal for rule in family}
        if probabilities is None
        else dict(probabilities)
    )
    if set(supplied) != {rule.name for rule in family} or any(
        isinstance(p, (bool, np.bool_)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in supplied.values()
    ):
        msg = "Supply finite coverage probabilities in [0, 1] for every rule"
        raise ValueError(msg)
    counts = np.array([count], dtype=np.int64)
    rows = []
    rejected = []
    for rule in family:
        lower, upper = _accepted_counts(rule, counts)
        probability = float(_rejection(rule, counts, supplied[rule.name])[0])
        rejected.append(probability)
        rows.append(
            {
                "rule": rule.model_dump(mode="json"),
                "coverage_probability": float(supplied[rule.name]),
                "accepted_count_min": int(lower[0]),
                "accepted_count_max": int(upper[0]),
                "rejection_probability": probability,
            }
        )
    return {
        "schema_version": 1,
        "n_replicates": count,
        "criteria": rows,
        "family_rejection_probability": {
            "lower": max(rejected),
            "upper": min(1.0, math.fsum(rejected)),
        },
        "evaluated_under": "nominal calibration"
        if probabilities is None
        else "specified coverage probabilities",
        "decision_rule": "reject_if_any_coverage_outside_inclusive_bounds",
        "sampling": "fixed independent successful replicate bank",
        "dependence_between_rules": "unrestricted",
        "implementation": "scipy.stats.binom.cdf/sf",
        "scipy_version": scipy_version,
        "adequacy_status": "not_assessed",
    }


def plan_coverage_replicates(
    rules: Sequence[CoverageRule],
    *,
    max_false_fail: float,
    max_replicates: int,
) -> dict[str, Any]:
    """Find the first fixed bank meeting a conservative family false-failure limit.

    Search every integer size, since lattice effects make binomial
    rejection probabilities nonmonotone. A union bound protects the complete
    supplied family without assuming independent estimands or coverage levels.
    The result applies to the exact chosen count, not every larger count.

    Args:
        rules: Complete family of inclusive coverage criteria.
        max_false_fail: Explicit maximum family false-failure probability in (0, 1).
        max_replicates: Explicit upper search budget for successful replicates.

    Returns:
        First qualifying count and its operating-characteristic report, or
        ``available=False`` and no count when the search budget cannot qualify.
        The count sizes only this fixed coverage rule, not failure policies,
        sampler diagnostics, other gate criteria or repeated interim looks.

    Raises:
        ValueError: Rules, probability target or budget are invalid.

    Examples:
        >>> rule = CoverageRule(name="half", nominal=0.5, lower=0.25, upper=0.75)
        >>> plan = plan_coverage_replicates(
        ...     (rule,), max_false_fail=0.125, max_replicates=4
        ... )
        >>> plan["available"], plan["n_replicates"]
        (True, 4)
    """
    family = _rules(rules)
    maximum = _count(max_replicates, "max_replicates")
    if (
        isinstance(max_false_fail, (bool, np.bool_))
        or not math.isfinite(max_false_fail)
        or not 0 < max_false_fail < 1
    ):
        msg = "max_false_fail must be finite and in (0, 1)"
        raise ValueError(msg)
    chosen = None
    # Bounded chunks avoid allocating the entire caller's search budget.
    for start in range(1, maximum + 1, 4096):
        counts = np.arange(start, min(start + 4096, maximum + 1), dtype=np.int64)
        upper = np.zeros(counts.size, dtype=np.float64)
        for rule in family:
            upper += _rejection(rule, counts, rule.nominal)
        passing = np.flatnonzero(upper <= max_false_fail)
        if passing.size:
            chosen = int(counts[passing[0]])
            break
    return {
        "schema_version": 1,
        "available": chosen is not None,
        "n_replicates": chosen,
        "max_false_fail": float(max_false_fail),
        "max_replicates": maximum,
        "rules": [rule.model_dump(mode="json") for rule in family],
        "reason": None if chosen is not None else "no_size_within_search_budget",
        "operating_characteristics": None
        if chosen is None
        else coverage_operating_characteristics(family, n_replicates=chosen),
        "sizing_method": "binomial union bound; exhaustive integer search",
        "adequacy_status": "not_assessed",
    }
