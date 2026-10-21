"""Fixed-bank posterior coverage, failure accounting and independent references."""

from __future__ import annotations

import doctest
import json
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from op_infer import calibrate
from op_infer.calibrate import coverage_summary, pit

pytest.importorskip("trade_study")


def test_coverage_gaussian_reference_and_overconfidence() -> None:
    """Exact normal-normal posteriors cover; narrow/biased posteriors under-cover."""
    rng = np.random.default_rng(125)
    truth = rng.normal(size=10000)
    observed = truth + rng.normal(size=truth.size)
    residual = truth - observed / 2
    ranks = {
        "calibrated": norm.cdf(residual / np.sqrt(0.5)),
        "overconfident": norm.cdf(residual / 0.1),
        "biased": norm.cdf((residual - 2) / np.sqrt(0.5)),
    }
    report = coverage_summary(ranks, confidence=0.99, n_failed=7)
    for row in report["estimands"]["calibrated"]:
        assert row["low"] < row["nominal"] < row["high"]
    for name in ("overconfident", "biased"):
        assert all(row["high"] < row["nominal"] for row in report["estimands"][name])
    assert report["n_total"] == 10007
    assert report["failed_fraction"] == pytest.approx(7 / 10007)
    assert report["conditioning"] == "successful fits"
    assert json.loads(json.dumps(report)) == report


def test_atom_coverage_uses_randomized_pits() -> None:
    """An exact point-mass posterior needs randomization for central coverage."""
    rng = np.random.default_rng(211)
    ranks = [pit([0, 0, 0], 0, atoms=True, rng=rng) for _ in range(5000)]
    report = coverage_summary({"atom": ranks}, confidence=0.999)
    assert all(
        row["low"] < row["nominal"] < row["high"] for row in report["estimands"]["atom"]
    )


def test_coverage_matches_independent_score_interval() -> None:
    """Endpoint inclusion and the Gaussian score-test inversion agree."""
    values = np.array([0, 0.25, 0.5, 0.75, 1])
    row = coverage_summary({"theta": values}, levels=[0.5])["estimands"]["theta"][0]
    assert (row["k"], row["n"], row["estimate"]) == (3, 5, 0.6)
    z = norm.ppf(0.975)
    # Solve n(p_hat-p)^2 = z^2 p(1-p), the score-test boundary.
    roots = np.roots([5 + z**2, -(2 * 3 + z**2), 3**2 / 5])
    np.testing.assert_allclose([row["low"], row["high"]], sorted(roots))


def test_all_failed_bank_is_unavailable() -> None:
    """No successful fit supplies no coverage interval, while all attempts count."""
    result = coverage_summary({}, n_failed=9)
    assert result["available"] is False
    assert result["estimands"] == {}
    assert result["n_total"] == 9
    assert result["failed_fraction"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pits": {}},
        {"pits": {"a": []}},
        {"pits": {"a": [0.5], "b": [0.2, 0.8]}},
        {"pits": {"a": [float("nan")]}},
        {"pits": {"a": [1.1]}},
        {"levels": []},
        {"levels": [0.5, 0.5]},
        {"levels": [0]},
        {"levels": [1]},
        {"levels": [float("nan")]},
        {"levels": [True]},
        {"n_failed": -1},
        {"n_failed": 0.5},
        {"n_failed": True},
        {"confidence": 1},
    ],
)
def test_invalid_bank_cannot_produce_coverage(kwargs: dict[str, Any]) -> None:
    """Malformed values and incomparable denominators are rejected."""
    with pytest.raises(ValueError, match=r"replicate|PIT|levels|n_failed|confidence"):
        coverage_summary(**({"pits": {"theta": [0.5]}} | kwargs))


def test_coverage_docstring() -> None:
    """The public example executes with the required study extra installed."""
    runner = doctest.DocTestRunner()
    finder = doctest.DocTestFinder()
    for test in finder.find(coverage_summary, globs=vars(calibrate)):
        for example in test.examples:
            example.options.pop(doctest.SKIP, None)
        runner.run(test)
    assert runner.failures == 0
