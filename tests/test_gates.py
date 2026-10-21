"""Seeded gates, unreadable outcomes, provenance refusal and amendments."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import norm

from op_infer.calibrate import pit
from op_infer.gates import Gate, GateAmendment, GateOutcome, GateRecord, GateSet
from op_infer.rng import RandomKey
from tests.test_provenance import reference

if TYPE_CHECKING:
    from pathlib import Path


def specification() -> GateSet:
    """Declare synthetic thresholds; none is a scientific calibration claim."""
    return GateSet(
        name="conjugate-fixture",
        stage="validation",
        context=reference("normal-normal toy design and estimator"),
        key=RandomKey(seed=198, stream="gates.validation"),
        gates=(
            Gate(name="adequate", metric="ess", role="adequacy", lower=100.0),
            Gate(name="powered", metric="n", role="adequacy", lower=1000.0),
            Gate(name="bias", metric="bias", role="decision", lower=-0.05, upper=0.05),
        ),
    )


def test_seeded_evaluation_replays_and_round_trips(tmp_path: Path) -> None:
    """Fresh native generators replay; the JSON record retains its criteria."""
    spec = specification()
    path = tmp_path / "validation.json"
    assert spec.write(path) == spec.write(path) == spec.reference()

    def evaluate(key: RandomKey) -> dict[str, float]:
        return {
            "ess": 1000.0,
            "n": 1000.0,
            "bias": key.numpy_generator().normal() / 1000,
        }

    first = spec.evaluate(
        path, evaluate, evidence=reference("inputs"), context=spec.context
    )
    replay = spec.evaluate(
        path, evaluate, evidence=reference("inputs"), context=spec.context
    )
    assert first == replay
    assert first.readable
    assert first.passed
    assert first.statuses() == ("pass", "pass", "pass")
    assert GateRecord.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize(
    ("metrics", "readable", "passed", "statuses"),
    [
        (
            {"ess": 50.0, "n": 1000.0, "bias": 0.0},
            False,
            False,
            ("fail", "pass", "pass"),
        ),
        (
            {"ess": 200.0, "n": 20.0, "bias": 0.0},
            False,
            False,
            ("pass", "fail", "pass"),
        ),
        (
            {"ess": 200.0, "n": 1000.0, "bias": 0.2},
            True,
            False,
            ("pass", "pass", "fail"),
        ),
        ({"ess": 200.0, "n": 1000.0}, True, False, ("pass", "pass", "unavailable")),
        ({"n": 1000.0, "bias": 0.0}, False, False, ("unavailable", "pass", "pass")),
        (
            {"ess": float("nan"), "n": 1000.0, "bias": 0.0},
            False,
            False,
            ("unavailable", "pass", "pass"),
        ),
    ],
)
def test_missing_inadequate_and_underpowered_results_cannot_pass(
    tmp_path: Path,
    metrics: dict[str, float],
    *,
    readable: bool,
    passed: bool,
    statuses: tuple[str, ...],
) -> None:
    """Decision failure remains distinct from unreadable/underpowered evidence."""
    spec = specification()
    path = tmp_path / "spec.json"
    spec.write(path)
    result = spec.evaluate(
        path, lambda _key: metrics, evidence=reference("inputs"), context=spec.context
    )
    assert result.readable is readable
    assert result.passed is passed
    assert result.statuses() == statuses
    loaded = GateRecord.model_validate_json(result.model_dump_json())
    assert loaded.passed is passed


def test_saved_specification_and_context_checked_before_evaluation(
    tmp_path: Path,
) -> None:
    """Mismatch, missing registration and conflicting writes cannot execute code."""
    spec = specification()
    path = tmp_path / "spec.json"

    def forbidden(_key: RandomKey) -> dict[str, float]:
        pytest.fail("Evaluator ran before contract validation")

    with pytest.raises(FileNotFoundError):
        spec.evaluate(
            path, forbidden, evidence=reference("inputs"), context=spec.context
        )
    spec.write(path)
    other = GateSet(**(spec.model_dump() | {"stage": "pilot"}))
    with pytest.raises(ValueError, match="specification differs"):
        other.write(path)
    with pytest.raises(ValueError, match="context differs"):
        spec.evaluate(
            path,
            forbidden,
            evidence=reference("inputs"),
            context=reference("other design"),
        )
    with pytest.raises(ValueError, match="specification or evidence context differs"):
        other.evaluate(
            path, forbidden, evidence=reference("inputs"), context=spec.context
        )


def test_amendment_retains_seen_evidence_and_prior_identity(tmp_path: Path) -> None:
    """A revision creates a distinct artifact and preserves the original bytes."""
    spec = specification()
    path = tmp_path / "original.json"
    spec.write(path)
    original = path.read_bytes()
    seen = reference("original calibration run")
    disclosure = GateAmendment(
        supersedes=spec.reference(),
        recorded_at=datetime(2026, 10, 19, tzinfo=UTC),
        reason="Prospective fixture revision after inspecting a pilot",
        seen_before_amendment=(seen,),
    )
    revised = spec.amend(
        gates=spec.gates,
        key=RandomKey(seed=199, stream="gates.validation"),
        disclosure=disclosure,
    )
    assert revised.reference() != spec.reference()
    revised.write(tmp_path / "revision.json")
    assert path.read_bytes() == original
    restored = GateSet.model_validate_json(revised.model_dump_json())
    assert restored.amendment is not None
    assert restored.amendment.seen_before_amendment == (seen,)
    assert restored.amendment.supersedes == spec.reference()


def test_invalid_criteria_and_tampered_records_refused() -> None:
    """No empty/reversed bounds, missing criteria, forged outcomes or mutation."""
    with pytest.raises(ValidationError, match="at least one bound"):
        Gate(name="x", metric="x", role="decision")
    with pytest.raises(ValidationError, match="exceeds"):
        Gate(name="x", metric="x", role="decision", lower=2.0, upper=1.0)
    with pytest.raises(ValidationError, match="finite"):
        Gate(name="x", metric="x", role="decision", lower=float("inf"))
    spec = specification()
    with pytest.raises(ValidationError, match="adequacy and decision"):
        GateSet(**(spec.model_dump() | {"gates": ()}))
    with pytest.raises(ValidationError, match="unique"):
        GateSet(**(spec.model_dump() | {"gates": spec.gates + spec.gates}))
    with pytest.raises(ValidationError, match="every criterion"):
        GateRecord(
            specification=spec,
            evidence=reference("inputs"),
            outcomes=(GateOutcome(name="bias", value=0.0),),
        )
    with pytest.raises(ValidationError, match="frozen"):
        spec.stage = "changed"


@pytest.mark.parametrize("atoms", [False, True])
def test_gate_on_calibrated_and_biased_posteriors(
    tmp_path: Path, *, atoms: bool
) -> None:
    """A seeded exact normal/point-mass posterior passes; biased PITs fail."""
    base = specification()
    spec = GateSet(
        name=base.name,
        stage=base.stage,
        context=base.context,
        key=base.key,
        gates=(
            *base.gates,
            Gate(
                name="coverage",
                metric="coverage90",
                role="decision",
                lower=0.87,
                upper=0.93,
            ),
        ),
    )
    path = tmp_path / "criteria.json"
    spec.write(path)

    def metrics(key: RandomKey) -> dict[str, float]:
        rng = key.numpy_generator()
        if atoms:
            ranks = np.array(
                [pit([0, 0, 0], 0, atoms=True, rng=rng) for _ in range(5000)]
            )
        else:
            theta = rng.normal(size=5000)
            observed = theta + rng.normal(size=theta.size)
            ranks = norm.cdf((theta - observed / 2) / np.sqrt(0.5))
        return {
            "bias": float(ranks.mean() - 0.5),
            "coverage90": float(np.mean((ranks >= 0.05) & (ranks <= 0.95))),
            "n": float(ranks.size),
            "ess": 1000.0,
        }

    exact = spec.evaluate(
        path, metrics, evidence=reference("exact posterior"), context=spec.context
    )
    assert exact.passed

    def biased(key: RandomKey) -> dict[str, float]:
        rng = key.numpy_generator()
        if atoms:
            ranks = np.array(
                [pit([1, 1, 1], 0, atoms=True, rng=rng) for _ in range(5000)]
            )
        else:
            theta = rng.normal(size=5000)
            observed = theta + rng.normal(size=theta.size)
            ranks = norm.cdf((theta - observed / 2 - 2) / np.sqrt(0.5))
        return {
            "bias": float(ranks.mean() - 0.5),
            "coverage90": float(np.mean((ranks >= 0.05) & (ranks <= 0.95))),
            "n": float(ranks.size),
            "ess": 1000.0,
        }

    failed = spec.evaluate(
        path, biased, evidence=reference("biased posterior"), context=spec.context
    )
    assert failed.readable
    assert not failed.passed


def test_amendment_disclosure_and_schema_validation() -> None:
    """Unknown versions, false integer versions and naive timestamps are refused."""
    spec = specification()
    for version in (True, 2):
        with pytest.raises(ValidationError, match="schema_version"):
            GateSet(**(spec.model_dump() | {"schema_version": version}))
    with pytest.raises(ValidationError, match="timezone-aware"):
        GateAmendment(
            supersedes=spec.reference(),
            recorded_at=datetime(2026, 10, 19),  # noqa: DTZ001 - invalid timestamp fixture
            reason="revision",
            seen_before_amendment=(),
        )
    wrong = GateAmendment(
        supersedes=reference("other spec"),
        recorded_at=datetime(2026, 10, 19, tzinfo=UTC),
        reason="revision",
        seen_before_amendment=(),
    )
    with pytest.raises(ValueError, match="supersedes"):
        spec.amend(gates=spec.gates, key=spec.key, disclosure=wrong)
