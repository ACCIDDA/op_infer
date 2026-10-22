"""Independent finite-sample references and saved calibration reading contracts."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import binom, norm

from op_infer.gates import GateAmendment
from op_infer.posterior import ArtifactReference
from op_infer.readings import (
    CalibrationEvidence,
    CalibrationPolicy,
    CalibrationRecord,
    CalibrationRule,
    MetricEstimate,
    ReplicateFailure,
    wilson_coverage,
    wilson_method,
)

if TYPE_CHECKING:
    from pathlib import Path


def ref(value: str) -> ArtifactReference:
    """Identify public synthetic fixture inputs."""
    return ArtifactReference.from_bytes(value.encode(), kind="toy-v1")


def policy(*, trials: int = 80, minimum: int | None = None) -> CalibrationPolicy:
    """Declare arbitrary public toy criteria before constructing any counts."""
    return CalibrationPolicy(
        name="independent-binomial-toy",
        context=ref("independent-bernoulli-cohort"),
        governing="interval",
        rationale="Illustrate compatibility and finite-sample resolution",
        replicate_ids=tuple(str(i) for i in range(trials)),
        minimum_usable=trials if minimum is None else minimum,
        max_failure_fraction=0.1,
        max_failed_quantities=0,
        must_pass=("quantity-a",),
        rules=(
            CalibrationRule(
                name="coverage",
                quantity="quantity-a",
                metric="coverage",
                method=wilson_method(confidence=0.95),
                lower=0.4,
                upper=0.6,
            ),
        ),
    )


def evidence(
    spec: CalibrationPolicy, successes: int, *, failures: int = 0
) -> CalibrationEvidence:
    """Keep all planned IDs and explicit failures in the original denominator."""
    n = len(spec.replicate_ids) - failures
    return CalibrationEvidence(
        context=spec.context,
        source=ref(f"toy-counts:{successes}/{n};failures={failures}"),
        usable=spec.replicate_ids[:n],
        failures=tuple(
            ReplicateFailure(replicate=i, reason="synthetic timeout", evidence=ref(i))
            for i in spec.replicate_ids[n:]
        ),
        metrics=(
            wilson_coverage(
                quantity="quantity-a",
                metric="coverage",
                successes=successes,
                trials=n,
                confidence=0.95,
            ),
        ),
    )


def test_exact_binomial_operating_characteristics(tmp_path: Path) -> None:
    """Enumerate all counts against an independent closed-form Wilson reference.

    Exact binomial probabilities qualify this toy, not a universal calibration
    rule. Deliberate misspecification at p=.8 has high detection probability.
    """
    spec = policy()
    path = tmp_path / "policy.json"
    spec.write(path)
    n = len(spec.replicate_ids)
    z = norm.ppf(0.975)
    failures = np.zeros((n + 1, 2), dtype=bool)
    for k in range(n + 1):
        p = k / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        record = spec.evaluate(path, evidence(spec, k))
        point, interval = record.readings()
        expected = (not 0.4 <= p <= 0.6, center + half < 0.4 or center - half > 0.6)
        failures[k] = point.state == "fail", interval.state == "fail"
        assert tuple(failures[k]) == expected
        metric = record.evidence.metrics[0]
        assert metric.lower is not None
        assert metric.upper is not None
        np.testing.assert_allclose(
            [metric.lower, metric.upper], [center - half, center + half], atol=1e-14
        )
    counts = np.arange(n + 1)
    calibrated = binom.pmf(counts, n, 0.5) @ failures
    misspecified = binom.pmf(counts, n, 0.8) @ failures
    assert 0 < calibrated[1] < calibrated[0] < 0.1
    assert 0.97 < misspecified[1] < misspecified[0] <= 1.0
    rng = np.random.default_rng(82397)
    for probability in (0.5, 0.8):
        sample = rng.binomial(n, probability, size=20_000)
        observed = failures[sample].mean(axis=0)
        exact = binom.pmf(counts, n, probability) @ failures
        np.testing.assert_allclose(observed, exact, atol=0.004)


def test_prospective_registration_and_governing_reading(tmp_path: Path) -> None:
    """A point failure and interval compatibility remain visible on one bank."""
    spec = policy()
    path = tmp_path / "policy.json"
    with pytest.raises(FileNotFoundError):
        spec.evaluate(path, evidence(spec, 51))
    assert spec.write(path) == spec.write(path) == spec.reference
    original = path.read_bytes()
    record = spec.evaluate(path, evidence(spec, 51))
    point, interval = record.readings()
    assert point.state == "fail"
    assert interval.state == record.governing_reading.state == "pass"
    loaded = CalibrationRecord.model_validate_json(record.model_dump_json())
    assert loaded == record
    assert loaded.readings() == record.readings()
    replacement = CalibrationPolicy.model_validate(
        spec.model_dump() | {"governing": "point"}
    )
    with pytest.raises(ValueError, match="policy differs"):
        replacement.write(path)
    with pytest.raises(ValueError, match="policy differs"):
        replacement.evaluate(path, record.evidence)
    amended = CalibrationPolicy.model_validate(
        replacement.model_dump()
        | {
            "amendment": GateAmendment(
                supersedes=spec.reference,
                recorded_at=datetime(2026, 10, 21, tzinfo=UTC),
                reason="Toy prospective change after the pilot",
                seen_before_amendment=(record.reference,),
            )
        }
    )
    amended.write(tmp_path / "amendment.json")
    assert path.read_bytes() == original
    assert amended.amendment is not None
    assert amended.amendment.seen_before_amendment == (record.reference,)


@pytest.mark.parametrize(
    ("failures", "reason"),
    [(1, "below_minimum_usable"), (9, "failure_budget_exceeded")],
)
def test_inadequate_evidence_preserves_failures(
    tmp_path: Path, failures: int, reason: str
) -> None:
    """Neither reading can pass below its floor or beyond its failure budget."""
    spec = policy(minimum=80 if failures == 1 else 60)
    path = tmp_path / "policy.json"
    spec.write(path)
    result = spec.evaluate(path, evidence(spec, 0, failures=failures))
    for panel in result.readings():
        assert panel.state == "insufficient_evidence"
        assert reason in panel.reasons
        assert panel.failed_quantities == ("quantity-a",)
    assert len(result.evidence.failures) == failures
    assert len(result.evidence.usable) + failures == 80


def test_complete_cohort_method_and_context_guards(tmp_path: Path) -> None:
    """Dropping a failed trial, changing a method or replacing context is refused."""
    spec = policy()
    path = tmp_path / "policy.json"
    spec.write(path)
    data = evidence(spec, 40)
    for changes in (
        {"context": ref("other")},
        {"usable": spec.replicate_ids[:-1], "metrics": ()},
        {"metrics": (data.metrics[0].model_copy(update={"method": ref("other")}),)},
    ):
        with pytest.raises(ValidationError, match="differs"):
            spec.evaluate(
                path, CalibrationEvidence.model_validate(data.model_dump() | changes)
            )
    with pytest.raises(ValidationError, match="unique replicates"):
        CalibrationEvidence.model_validate(
            data.model_dump() | {"usable": (*data.usable, "0")}
        )
    with pytest.raises(ValidationError, match="one usable cohort"):
        CalibrationEvidence.model_validate(
            data.model_dump()
            | {"metrics": (data.metrics[0].model_copy(update={"n_usable": 79}),)}
        )
    missing = CalibrationEvidence.model_validate(data.model_dump() | {"metrics": ()})
    result = spec.evaluate(path, missing)
    assert all(r.state == "insufficient_evidence" for r in result.readings())


def test_distinct_quantity_budget_must_pass_and_fixed_endpoint_cap(
    tmp_path: Path,
) -> None:
    """Two violations of one quantity consume one slot; must-pass overrides it.

    A practical cap on a supplied metric's lower endpoint is independent of
    coverage compatibility and never compares the length of a Wilson interval.
    """
    base = policy()
    generic = ref("public-generic-metric-and-interval-definition")
    quantities = ("quantity-a", "quantity-b", "quantity-c")
    rules = tuple(
        CalibrationRule(
            name=f"coverage-{q}",
            quantity=q,
            metric="coverage",
            method=wilson_method(confidence=0.95),
            lower=0.4,
            upper=0.6,
        )
        for q in quantities
    )
    rules += (
        CalibrationRule(
            name="practical-cap",
            quantity="quantity-b",
            metric="spread-index",
            method=generic,
            upper=2.0,
            view="lower",
        ),
    )
    spec = CalibrationPolicy.model_validate(
        base.model_dump() | {"rules": rules, "max_failed_quantities": 1}
    )
    path = tmp_path / "policy.json"
    spec.write(path)
    metrics = tuple(
        wilson_coverage(
            quantity=q,
            metric="coverage",
            successes=80 if q == "quantity-b" else 40,
            trials=80,
            confidence=0.95,
        )
        for q in quantities
    )
    metrics += (
        MetricEstimate(
            quantity="quantity-b",
            metric="spread-index",
            method=generic,
            n_usable=80,
            point=3.0,
            lower=2.5,
            upper=4.0,
        ),
    )
    data = CalibrationEvidence(
        context=spec.context,
        source=ref("toy-inputs"),
        usable=spec.replicate_ids,
        failures=(),
        metrics=metrics,
    )
    result = spec.evaluate(path, data)
    assert all(
        r.failed_quantities == ("quantity-b",) and r.state == "pass"
        for r in result.readings()
    )
    required = CalibrationPolicy.model_validate(
        spec.model_dump() | {"must_pass": ("quantity-b",)}
    )
    required_path = tmp_path / "required.json"
    required.write(required_path)
    assert all(
        r.state == "fail" for r in required.evaluate(required_path, data).readings()
    )
    compatible = (
        tuple(
            wilson_coverage(
                quantity=q, metric="coverage", successes=40, trials=80, confidence=0.95
            )
            for q in quantities
        )
        + metrics[-1:]
    )
    compatible_data = CalibrationEvidence.model_validate(
        data.model_dump() | {"metrics": compatible}
    )
    panel = required.evaluate(required_path, compatible_data).readings()[1]
    assert panel.state == "fail"
    assert panel.outcomes[-1].lower == panel.outcomes[-1].upper == 2.5
    assert all(o.status == "pass" for o in panel.outcomes[:-1])


def test_unavailable_intervals_and_schema_guards(tmp_path: Path) -> None:
    """Unavailable interval evidence cannot masquerade as compatible coverage."""
    spec = policy()
    path = tmp_path / "policy.json"
    spec.write(path)
    data = evidence(spec, 40)
    metric = MetricEstimate.model_validate(
        data.metrics[0].model_dump() | {"lower": None, "upper": None}
    )
    record = spec.evaluate(
        path,
        CalibrationEvidence.model_validate(data.model_dump() | {"metrics": (metric,)}),
    )
    assert record.readings()[0].state == "pass"
    assert record.readings()[1].state == "insufficient_evidence"
    assert (
        wilson_coverage(
            quantity="a", metric="coverage", successes=0, trials=0, confidence=0.95
        ).point
        is None
    )
    for changes in (
        {"schema_version": True},
        {"must_pass": ("absent",)},
        {"replicate_ids": ("0", "0")},
        {"governing": "choose-afterwards"},
    ):
        with pytest.raises(ValidationError):
            CalibrationPolicy.model_validate(spec.model_dump() | changes)
    for counts in ((True, 80), (-1, 80), (81, 80), (0, -1)):
        with pytest.raises(ValueError, match="integer counts"):
            wilson_coverage(
                quantity="a",
                metric="coverage",
                successes=counts[0],
                trials=counts[1],
                confidence=0.95,
            )
    for confidence in (0.0, 1.0, float("nan"), True):
        with pytest.raises(ValueError, match="confidence"):
            wilson_method(confidence=confidence)
