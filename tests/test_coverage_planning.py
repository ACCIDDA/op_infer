"""Fixed coverage families against enumeration and calibrated Gaussian posteriors."""

from __future__ import annotations

import json
import math
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import norm

from op_infer.calibrate import (
    CoverageRule,
    coverage_operating_characteristics,
    pit,
    plan_coverage_replicates,
)


def criterion(name: str = "theta_50") -> CoverageRule:
    """Use an explicit toy range; it is not a package scientific default."""
    return CoverageRule(name=name, nominal=0.5, lower=0.25, upper=0.75)


@pytest.mark.parametrize("probability", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_binomial_probabilities_against_direct_enumeration(probability: float) -> None:
    """Sum independent Bernoulli masses, without the reference tail routines."""
    n = 12
    result = coverage_operating_characteristics(
        (criterion(),), n_replicates=n, probabilities={"theta_50": probability}
    )
    expected = sum(
        math.comb(n, k) * probability**k * (1 - probability) ** (n - k)
        for k in range(n + 1)
        if not 0.25 <= k / n <= 0.75
    )
    row = result["criteria"][0]
    assert row["accepted_count_min"] == 3
    assert row["accepted_count_max"] == 9
    assert row["rejection_probability"] == pytest.approx(expected)
    assert result["family_rejection_probability"]["lower"] == pytest.approx(expected)
    assert result["family_rejection_probability"]["upper"] == pytest.approx(expected)
    assert result["evaluated_under"] == "specified coverage probabilities"
    assert result["adequacy_status"] == "not_assessed"
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_inclusive_float_boundaries_match_actual_proportion_comparison() -> None:
    """Rounding a threshold product cannot discard an accepted boundary count."""
    rule = CoverageRule(name="boundary", nominal=0.57, lower=0.56, upper=0.58)
    result = coverage_operating_characteristics((rule,), n_replicates=50)
    row = result["criteria"][0]
    assert (row["accepted_count_min"], row["accepted_count_max"]) == (28, 29)
    accepted = [k for k in range(51) if rule.lower <= k / 50 <= rule.upper]
    assert accepted == [28, 29]


def test_empty_acceptance_set_and_full_range() -> None:
    """An empty integer acceptance set always rejects; the full range never does."""
    narrow = CoverageRule(name="exact_half", nominal=0.5, lower=0.5, upper=0.5)
    wide = CoverageRule(name="all", nominal=0.5, lower=0.0, upper=1.0)
    rows = coverage_operating_characteristics((narrow, wide), n_replicates=1)[
        "criteria"
    ]
    assert rows[0]["accepted_count_min"] > rows[0]["accepted_count_max"]
    assert rows[0]["rejection_probability"] == 1
    assert rows[1]["rejection_probability"] == 0


def test_family_bounds_do_not_assume_independent_estimands() -> None:
    """Perfect dependence and independence both lie inside the returned bounds."""
    result = coverage_operating_characteristics(
        (criterion("a"), criterion("b"), criterion("c")), n_replicates=12
    )
    marginal = result["criteria"][0]["rejection_probability"]
    bounds = result["family_rejection_probability"]
    assert bounds["lower"] == marginal  # All three indicators could be identical.
    assert bounds["upper"] == pytest.approx(3 * marginal)
    independent = 1 - (1 - marginal) ** 3
    assert bounds["lower"] < independent < bounds["upper"]
    assert result["dependence_between_rules"] == "unrestricted"


def test_search_checks_lattice_sizes_and_retains_insufficient_budget() -> None:
    """N=4 qualifies but N=5 fails; a monotone search could miss the first size."""
    first = plan_coverage_replicates(
        (criterion(),), max_false_fail=0.125, max_replicates=5
    )
    assert first["available"]
    assert first["n_replicates"] == 4
    assert (
        first["operating_characteristics"]["family_rejection_probability"]["upper"]
        == 0.125
    )
    later = coverage_operating_characteristics((criterion(),), n_replicates=5)
    assert later["family_rejection_probability"]["upper"] > 0.125
    short = plan_coverage_replicates(
        (criterion(),), max_false_fail=0.125, max_replicates=3
    )
    assert not short["available"]
    assert short["n_replicates"] is None
    assert short["operating_characteristics"] is None
    assert short["reason"] == "no_size_within_search_budget"
    assert short["rules"] == [criterion().model_dump(mode="json")]


def test_whole_family_requires_more_replicates_than_one_rule() -> None:
    """The supplied family affects sizing even when every nominal level is 50%."""
    family = tuple(
        CoverageRule(name=f"q{i}", nominal=0.5, lower=0.45, upper=0.55)
        for i in range(20)
    )
    single = plan_coverage_replicates(
        family[:1], max_false_fail=0.01, max_replicates=4000
    )
    combined = plan_coverage_replicates(
        family, max_false_fail=0.01, max_replicates=4000
    )
    assert single["n_replicates"] < combined["n_replicates"]
    assert (
        combined["operating_characteristics"]["family_rejection_probability"]["upper"]
        <= 0.01
    )


def test_calibrated_and_biased_gaussian_posterior_banks() -> None:
    """A complete gate family is checked on exact conjugate and biased posteriors."""
    family = tuple(
        CoverageRule(name=f"q{i}", nominal=0.5, lower=0.45, upper=0.55)
        for i in range(3)
    )
    plan = plan_coverage_replicates(family, max_false_fail=0.05, max_replicates=3000)
    n = plan["n_replicates"]
    rng = np.random.default_rng(518)
    banks = 300
    truth = rng.normal(size=(banks, n, len(family)))
    observed = truth + rng.normal(size=truth.shape)
    # theta~N(0,1), y|theta~N(theta,1): exact posterior N(y/2, 1/2).
    residual = (truth - observed / 2) / np.sqrt(0.5)
    rejected = []
    for shift in (0.0, 1.0):
        pits = norm.cdf(residual - shift / np.sqrt(0.5))
        coverage = ((pits >= 0.25) & (pits <= 0.75)).mean(axis=1)
        rejected.append(np.any((coverage < 0.45) | (coverage > 0.55), axis=1))
    assert rejected[0].sum() < banks * 0.05 + 5 * np.sqrt(banks * 0.05 * 0.95)
    assert rejected[1].mean() > 0.95
    # The alternative's analytical central-coverage probability is known.
    z = norm.ppf(0.75)
    probability = float(norm.cdf(z + np.sqrt(2)) - norm.cdf(-z + np.sqrt(2)))
    alternative = coverage_operating_characteristics(
        family,
        n_replicates=n,
        probabilities={rule.name: probability for rule in family},
    )
    assert alternative["family_rejection_probability"]["lower"] > 0.99


def test_calibrated_point_mass_uses_randomized_pits() -> None:
    """The planner's nominal model also holds for the atom-aware PIT convention."""
    rule = CoverageRule(name="atom", nominal=0.5, lower=0.45, upper=0.55)
    plan = plan_coverage_replicates((rule,), max_false_fail=0.05, max_replicates=2000)
    n = plan["n_replicates"]
    banks = 128
    rng = np.random.default_rng(211)
    values = np.array(
        [pit([0, 0], 0, atoms=True, rng=rng) for _ in range(banks * n)]
    ).reshape(banks, n)
    coverage = ((values >= 0.25) & (values <= 0.75)).mean(axis=1)
    failures = np.count_nonzero((coverage < 0.45) | (coverage > 0.55))
    assert failures < banks * 0.05 + 5 * np.sqrt(banks * 0.05 * 0.95)
    # Midranks concentrate at 0.5, giving 100% central coverage for this atom.
    assert pit([0, 0], 0, atoms=False) == 0.5


@pytest.mark.parametrize("value", [0, -1, True, 2.5, float("nan"), 2**53])
def test_invalid_counts_cannot_supply_a_plan(value: object) -> None:
    """Counts must be actual positive integers in the supported numerical range."""
    with pytest.raises(ValueError, match="n_replicates"):
        coverage_operating_characteristics(
            (criterion(),), n_replicates=cast("int", value)
        )
    with pytest.raises(ValueError, match="max_replicates"):
        plan_coverage_replicates(
            (criterion(),), max_false_fail=0.05, max_replicates=cast("int", value)
        )


@pytest.mark.parametrize("probability", [True, np.bool_(0), -0.1, 1.1, float("nan")])
def test_invalid_probability_mapping(probability: object) -> None:
    """Alternative probabilities must be finite probabilities, not truth values."""
    with pytest.raises(ValueError, match="every rule"):
        coverage_operating_characteristics(
            (criterion(),),
            n_replicates=12,
            probabilities={"theta_50": cast("float", probability)},
        )


def test_rule_family_and_specification_validation() -> None:
    """Missing/duplicate rules and invalid bounds cannot silently change a gate."""
    for family in ((), (criterion(), criterion())):
        with pytest.raises(ValueError, match="family"):
            coverage_operating_characteristics(family, n_replicates=4)
    for probabilities in ({}, {"theta_50": 0.5, "extra": 0.5}):
        with pytest.raises(ValueError, match="every rule"):
            coverage_operating_characteristics(
                (criterion(),), n_replicates=4, probabilities=probabilities
            )
    with pytest.raises(ValidationError, match="contain"):
        CoverageRule(name="bad", nominal=0.5, lower=0.7, upper=0.9)
    invalid = criterion().model_copy(update={"lower": 0.9})
    with pytest.raises(ValidationError, match="contain"):
        coverage_operating_characteristics((invalid,), n_replicates=4)
    for maximum in (True, np.bool_(0), 0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="max_false_fail"):
            plan_coverage_replicates(
                (criterion(),), max_false_fail=cast("float", maximum), max_replicates=4
            )
