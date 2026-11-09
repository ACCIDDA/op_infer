"""Independent references and failure accounting for per-unit importance integration."""

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import logsumexp

from op_infer.integration import (
    ImportanceProposal,
    UnitIntegrationResult,
    integrate_unit_importance,
)


def _normal_logpdf(points: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return -0.5 * ((points[:, 0] - mean) / sd) ** 2 - math.log(
        sd * math.sqrt(2 * math.pi)
    )


def _normal_proposal(
    identity: str, mean: float, sd: float, *, mode_identity: str | None = None
) -> ImportanceProposal:
    def sample(rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.normal(mean, sd, size=(count, 1))

    def logpdf(points: np.ndarray) -> np.ndarray:
        return _normal_logpdf(points, mean, sd)

    return ImportanceProposal(identity, sample, logpdf, mode_identity)


def _integrate(  # noqa: PLR0913 - independent budget and stream controls
    target: Callable[[np.ndarray], np.ndarray],
    proposals: tuple[ImportanceProposal, ...],
    *,
    weights: tuple[float, ...] | None = None,
    draws: int = 1024,
    seed: int = 41,
    draw_identity: str = "independent-stream-41",
) -> UnitIntegrationResult:
    """Run one explicitly identified synthetic unit attempt."""
    return integrate_unit_importance(
        target,
        proposals,
        weights or tuple(1 / len(proposals) for _ in proposals),
        defensive_component=proposals[-1].identity,
        dimension=1,
        draw_count=draws,
        rng=np.random.default_rng(seed),
        unit_id="synthetic-unit",
        parameter_id="theta-0",
        attempt_id=f"attempt-{seed}",
        draw_identity=draw_identity,
        target_identity="analytic-reference-target-v1",
        proposal_identity="synthetic-proposal-mixture-v1",
    )


def test_normal_integral_matches_analytic_value_with_exact_proposal() -> None:
    """Matching normal target/proposal yields exact unit evidence and Kish ESS."""
    proposal = _normal_proposal("normal", mean=1.2, sd=0.7)
    result = _integrate(
        lambda points: _normal_logpdf(points, 1.2, 0.7),
        (proposal,),
        draws=257,
    )

    assert result.status == "estimated"
    assert result.log_estimate == pytest.approx(0.0, abs=1e-14)
    assert result.ess == pytest.approx(257.0)
    assert result.finite_weight_count == 257
    assert result.zero_weight_count == result.invalid_weight_count == 0
    assert result.draw_digest is not None
    assert result.defensive_component == "normal"


def test_multimodal_integral_matches_analytic_mixture_across_budgets() -> None:
    """Independent draw sets recover a two-mode analytic integral as budget grows."""
    left = _normal_proposal("left-proposal", -2.0, 1.0, mode_identity="mode-left")
    right = _normal_proposal("right-proposal", 2.4, 1.0, mode_identity="mode-right")

    def target(points: np.ndarray) -> np.ndarray:
        return np.asarray(
            logsumexp(
                np.stack(
                    [
                        math.log(0.35) + _normal_logpdf(points, -2.0, 0.5),
                        math.log(0.65) + _normal_logpdf(points, 2.4, 0.7),
                    ]
                ),
                axis=0,
            )
        )

    low, high = [], []
    digests = set()
    for replicate in range(10):
        low_result = _integrate(
            target,
            (left, right),
            draws=200,
            seed=100 + replicate,
            draw_identity=f"low-budget-{replicate}",
        )
        high_result = _integrate(
            target,
            (left, right),
            draws=4000,
            seed=1000 + replicate,
            draw_identity=f"high-budget-{replicate}",
        )
        assert low_result.status == high_result.status == "estimated"
        assert low_result.proposal_components == ("left-proposal", "right-proposal")
        assert low_result.mode_search_status == "not_used"
        assert low_result.finite_weight_count == 200
        assert high_result.finite_weight_count == 4000
        low.append(low_result.log_estimate)
        high.append(high_result.log_estimate)
        digests.add(low_result.draw_digest)
        digests.add(high_result.draw_digest)

    low_error = np.asarray(low, dtype=float)
    high_error = np.asarray(high, dtype=float)
    assert abs(float(high_error.mean())) < 0.025
    assert float(np.mean(high_error**2)) < float(np.mean(low_error**2))
    assert len(digests) == 20


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_invalid_log_weights_are_counted_and_never_dropped(bad_value: float) -> None:
    """An invalid sampled target makes the whole attempt unreadable."""
    proposal = _normal_proposal("normal", 0.0, 1.0)

    def target(points: np.ndarray) -> np.ndarray:
        values = _normal_logpdf(points, 0.0, 1.0)
        values[0] = bad_value
        return values

    result = _integrate(target, (proposal,), draws=50)
    assert result.status == "unreadable"
    assert result.log_estimate is None
    assert result.invalid_weight_count == 1
    assert result.finite_weight_count == 49
    assert result.ess is None
    assert "no weights were dropped" in (result.reason or "")


def test_all_zero_weights_are_unreadable_not_a_zero_likelihood_claim() -> None:
    """A proposal can miss support; an all-zero draw is not evidence of zero target."""
    proposal = _normal_proposal("normal", 0.0, 1.0)
    result = _integrate(lambda points: np.full(len(points), -math.inf), (proposal,))

    assert result.status == "unreadable"
    assert result.log_estimate is None
    assert result.finite_weight_count == 0
    assert result.zero_weight_count == result.draw_count
    assert result.ess == 0.0
    assert "missed mode" in (result.reason or "")


def test_callback_failure_is_a_retained_failed_attempt() -> None:
    """A proposal exception surfaces without pretending to have drawn samples."""

    def fail(_rng: np.random.Generator, _count: int) -> np.ndarray:
        msg = "synthetic proposal failure"
        raise RuntimeError(msg)

    proposal = ImportanceProposal("failing-proposal", fail, lambda points: points[:, 0])
    result = _integrate(lambda points: points[:, 0], (proposal,), draws=32)

    assert result.status == "failed"
    assert result.draw_count == 32
    assert result.completed_draw_count == 0
    assert result.log_estimate is None
    assert "synthetic proposal failure" in (result.reason or "")


def test_validation_requires_positive_defensive_mixture_mass() -> None:
    """Invalid caller-owned proposal mixture controls fail before sampling."""
    proposal = _normal_proposal("normal", 0.0, 1.0)
    with pytest.raises(ValueError, match="positive and sum to one"):
        _integrate(lambda points: points[:, 0], (proposal,), weights=(0.0,))


def test_mode_search_metadata_is_caller_owned_and_kept() -> None:
    """A supplied mode-search record is retained, not run or requalified here."""
    proposal = _normal_proposal("mode-proposal", 0.0, 1.0, mode_identity="mode-1")
    result = integrate_unit_importance(
        lambda points: _normal_logpdf(points, 0.0, 1.0),
        (proposal,),
        (1.0,),
        defensive_component="mode-proposal",
        dimension=1,
        draw_count=10,
        rng=np.random.default_rng(10),
        unit_id="unit-1",
        parameter_id="theta-1",
        attempt_id="attempt-1",
        draw_identity="mode-supplied-stream-1",
        target_identity="target-1",
        proposal_identity="proposal-1",
        mode_search_identity="external-search-settings-v1",
        mode_search_status="succeeded",
    )
    assert result.mode_search_identity == "external-search-settings-v1"
    assert result.mode_search_status == "succeeded"


def test_valid_zero_weights_keep_the_planned_denominator() -> None:
    """A half-normal integral is one half; zero draws remain in its denominator."""
    proposal = _normal_proposal("normal", 0.0, 1.0)

    def target(points: np.ndarray) -> np.ndarray:
        return np.where(points[:, 0] >= 0, _normal_logpdf(points, 0.0, 1.0), -math.inf)

    result = _integrate(target, (proposal,), draws=4096)
    assert result.status == "estimated"
    assert result.log_estimate is not None
    assert math.exp(result.log_estimate) == pytest.approx(0.5, abs=0.025)
    assert result.finite_weight_count + result.zero_weight_count == result.draw_count
    assert result.zero_weight_count > 0
    assert result.ess == pytest.approx(result.finite_weight_count)


@pytest.mark.parametrize("density", [math.nan, math.inf, -math.inf])
def test_invalid_mixture_density_makes_all_weights_unreadable(density: float) -> None:
    """A sampled point must have a finite actual mixture density."""
    normal = _normal_proposal("normal", 0.0, 1.0)
    proposal = ImportanceProposal(
        "bad-density", normal.sample, lambda points: np.full(len(points), density)
    )
    result = _integrate(lambda points: _normal_logpdf(points, 0.0, 1.0), (proposal,))
    assert result.status == "unreadable"
    assert result.invalid_weight_count == result.draw_count
    assert result.log_estimate is result.ess is None


def test_malformed_callback_shape_is_explicit_failed_attempt() -> None:
    """A scalar instead of per-draw density is not silently broadcast."""
    normal = _normal_proposal("normal", 0.0, 1.0)
    result = _integrate(lambda _points: np.asarray(0.0), (normal,))
    assert result.status == "failed"
    assert result.completed_draw_count == result.draw_count
    assert result.draw_digest is not None
    assert "one value per draw" in (result.reason or "")


def test_unexpected_callback_error_propagates_to_execution_ledger() -> None:
    """Programming faults are not obscured by an unrestricted callback catch."""
    normal = _normal_proposal("normal", 0.0, 1.0)

    def target(_points: np.ndarray) -> np.ndarray:
        msg = "missing programmer-owned state"
        raise KeyError(msg)

    with pytest.raises(KeyError, match="programmer-owned"):
        _integrate(target, (normal,))
