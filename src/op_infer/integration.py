"""Raw defensive-mixture importance integration with retained unit attempts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

type Points = NDArray[np.float64]
type LogTarget = Callable[[Points], Points]

__all__ = ["ImportanceProposal", "UnitIntegrationResult", "integrate_unit_importance"]


@dataclass(frozen=True)
class ImportanceProposal:
    """Caller-qualified normalized sampler/density pair in integration coordinates."""

    identity: str
    sample: Callable[[np.random.Generator, int], Points]
    logpdf: LogTarget
    mode_identity: str | None = None


@dataclass(frozen=True)
class UnitIntegrationResult:
    """One planned attempt; an estimate and raw ESS never establish adequacy."""

    status: Literal["estimated", "unreadable", "failed"]
    unit_id: str
    parameter_id: str
    attempt_id: str
    draw_identity: str
    target_identity: str
    proposal_identity: str
    defensive_component: str
    proposal_components: tuple[str, ...]
    proposal_weights: tuple[float, ...]
    mode_identities: tuple[str | None, ...]
    mode_search_identity: str | None
    mode_search_status: Literal["not_used", "succeeded", "failed"]
    draw_count: int
    completed_draw_count: int
    draw_digest: str | None
    log_estimate: float | None
    ess: float | None
    finite_weight_count: int
    zero_weight_count: int
    invalid_weight_count: int
    reason: str | None


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a nonempty identity"
        raise ValueError(msg)


def _validate_proposals(
    components: tuple[ImportanceProposal, ...],
    weights: tuple[float, ...],
    defensive_component: str,
) -> Points:
    if not components or any(not isinstance(p, ImportanceProposal) for p in components):
        msg = "At least one ImportanceProposal is required"
        raise ValueError(msg)
    ids = tuple(p.identity for p in components)
    for proposal in components:
        _text(proposal.identity, "component identity")
        if proposal.mode_identity is not None:
            _text(proposal.mode_identity, "mode identity")
        if not callable(proposal.sample) or not callable(proposal.logpdf):
            msg = "Proposal sampler and log density must be callable"
            raise TypeError(msg)
    probability = np.asarray(weights, dtype=np.float64)
    if (
        probability.shape != (len(components),)
        or not np.all(np.isfinite(probability))
        or np.any(probability <= 0)
        or not np.isclose(probability.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        msg = "Mixture weights must be positive and sum to one"
        raise ValueError(msg)
    if len(set(ids)) != len(ids) or defensive_component not in ids:
        msg = "Unique component identities must include the defensive component"
        raise ValueError(msg)
    return probability


def _sample_component(
    proposal: ImportanceProposal,
    rng: np.random.Generator,
    count: int,
    dimension: int,
) -> Points:
    values = np.asarray(proposal.sample(rng, count))
    if (
        values.shape != (count, dimension)
        or values.dtype.kind != "f"
        or not np.all(np.isfinite(values))
    ):
        msg = f"Invalid sampled points from component {proposal.identity}"
        raise ValueError(msg)
    return np.asarray(values, dtype=np.float64)


def _log_weights(
    target: LogTarget,
    components: tuple[ImportanceProposal, ...],
    probability: Points,
    points: Points,
) -> tuple[Points, NDArray[np.bool_]]:
    logp = np.asarray(target(points), dtype=np.float64)
    densities = [np.asarray(p.logpdf(points), dtype=np.float64) for p in components]
    if logp.shape != (len(points),) or any(d.shape != logp.shape for d in densities):
        msg = "Log target and component densities require one value per draw"
        raise ValueError(msg)
    stack = np.stack(densities)
    bad_density = np.any(np.isnan(stack) | np.isposinf(stack), axis=0)
    logq = logsumexp(np.log(probability)[:, None] + stack, axis=0)
    with np.errstate(invalid="ignore"):
        logw = logp - logq
    invalid = bad_density | np.isnan(logw) | np.isposinf(logw) | ~np.isfinite(logq)
    return logw, invalid


def _validate_metadata(
    identities: tuple[tuple[str, str], ...],
    counts: tuple[tuple[str, int], ...],
    mode_identity: str | None,
    mode_status: str,
) -> None:
    for name, value in identities:
        _text(value, name)
    for name, count in counts:
        if type(count) is not int or count < 1:
            msg = f"{name} must be a positive integer"
            raise ValueError(msg)
    if mode_status not in {"not_used", "succeeded", "failed"}:
        msg = "Invalid caller-owned mode-search status"
        raise ValueError(msg)
    if mode_identity is not None:
        _text(mode_identity, "mode-search identity")
    if (mode_status == "not_used") != (mode_identity is None):
        msg = "Used mode search requires an identity; unused search has none"
        raise ValueError(msg)


def integrate_unit_importance(  # noqa: PLR0913 - explicit target/attempt identities
    target: LogTarget,
    proposals: tuple[ImportanceProposal, ...],
    weights: tuple[float, ...],
    *,
    defensive_component: str,
    dimension: int,
    draw_count: int,
    rng: np.random.Generator,
    unit_id: str,
    parameter_id: str,
    attempt_id: str,
    draw_identity: str,
    target_identity: str,
    proposal_identity: str,
    mode_search_identity: str | None = None,
    mode_search_status: Literal["not_used", "succeeded", "failed"] = "not_used",
) -> UnitIntegrationResult:
    """Estimate an integral using the actual whole-mixture proposal density.

    Caller owns the target measure, bounds, latent prior and Jacobian inclusion,
    normalized proposal laws, tail/support qualification and RNG state. No mode
    search, Laplace construction or adequacy gate occurs here. Expected numerical
    callback failures retain a failed attempt, with no invented estimate or dropped
    weights. Unexpected exceptions propagate to the caller's attempt ledger.

    Returns:
        Attempt record including raw weight counts and Kish ESS.

    Raises:
        ValueError: Metadata, dimensions or mixture controls are invalid.
        TypeError: RNG or callbacks do not satisfy the declared interface.
    """
    names = (
        ("unit_id", unit_id),
        ("parameter_id", parameter_id),
        ("attempt_id", attempt_id),
        ("draw_identity", draw_identity),
        ("target_identity", target_identity),
        ("proposal_identity", proposal_identity),
        ("defensive_component", defensive_component),
    )
    _validate_metadata(
        names,
        (("dimension", dimension), ("draw_count", draw_count)),
        mode_search_identity,
        mode_search_status,
    )
    if not isinstance(rng, np.random.Generator) or not callable(target):
        msg = "Explicit Generator and callable log target required"
        raise TypeError(msg)
    components = tuple(proposals)
    probability = _validate_proposals(components, weights, defensive_component)
    ids = tuple(p.identity for p in components)

    def result(  # noqa: PLR0913, PLR0917 - separate measured attempt outcomes
        status: Literal["estimated", "unreadable", "failed"],
        completed: int,
        digest: str | None,
        estimate: float | None,
        ess: float | None,
        finite: int,
        zero: int,
        invalid: int,
        reason: str | None,
    ) -> UnitIntegrationResult:
        return UnitIntegrationResult(
            status,
            unit_id,
            parameter_id,
            attempt_id,
            draw_identity,
            target_identity,
            proposal_identity,
            defensive_component,
            ids,
            tuple(float(w) for w in probability),
            tuple(p.mode_identity for p in components),
            mode_search_identity,
            mode_search_status,
            draw_count,
            completed,
            digest,
            estimate,
            ess,
            finite,
            zero,
            invalid,
            reason,
        )

    if mode_search_status == "failed":
        return result(
            "failed", 0, None, None, None, 0, 0, 0, "Caller reported failed mode search"
        )
    points = np.empty((draw_count, dimension), dtype=np.float64)
    completed = 0
    digest = None
    stage = "sampling"
    try:
        labels = rng.choice(len(components), size=draw_count, p=probability)
        for index, proposal in enumerate(components):
            selected = labels == index
            count = int(selected.sum())
            if not count:
                continue
            points[selected] = _sample_component(proposal, rng, count, dimension)
            completed += count
        digest = hashlib.sha256(
            f"{points.shape}:float64:little".encode() + points.astype("<f8").tobytes()
        ).hexdigest()
        stage = "log-density evaluation"
        logw, invalid_mask = _log_weights(target, components, probability, points)
        finite = int((np.isfinite(logw) & ~invalid_mask).sum())
        zero = int((np.isneginf(logw) & ~invalid_mask).sum())
        invalid = int(invalid_mask.sum())
        if invalid:
            return result(
                "unreadable",
                completed,
                digest,
                None,
                None,
                finite,
                zero,
                invalid,
                "Invalid weights; no weights were dropped",
            )
        if not finite:
            return result(
                "unreadable",
                completed,
                digest,
                None,
                0.0,
                0,
                zero,
                0,
                "All weights zero: proposal may have missed mode or support",
            )
        total = float(logsumexp(logw))
        normalized = np.exp(logw - total)
        ess = float(1.0 / np.square(normalized).sum())
        return result(
            "estimated",
            completed,
            digest,
            total - np.log(draw_count),
            ess,
            finite,
            zero,
            0,
            None,
        )
    except (ValueError, TypeError, ArithmeticError, RuntimeError) as error:
        return result(
            "failed",
            completed,
            digest,
            None,
            None,
            0,
            0,
            0,
            f"{stage}: {type(error).__name__}: {error}",
        )
