"""Saved point and interval readings of a fixed calibration replicate bank.

Consumers supply metric definitions and qualify their statistical assumptions.
An interval reading reports compatibility, not equivalence or calibration.
Application-specific statistics and thresholds are supplied by the consumer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.stats import binomtest

from op_infer._registration import write_immutable_json
from op_infer.gates import GateAmendment  # noqa: TC001 - Pydantic resolves schema
from op_infer.posterior import ArtifactReference

__all__ = [
    "CalibrationEvidence",
    "CalibrationPolicy",
    "CalibrationRecord",
    "CalibrationRule",
    "MetricEstimate",
    "Reading",
    "ReplicateFailure",
    "RuleOutcome",
    "wilson_coverage",
    "wilson_method",
]

type _Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
type _Finite = Annotated[float, Field(allow_inf_nan=False)]
type _Count = Annotated[int, Field(ge=0)]
type _ReadingName = Literal["point", "interval"]


class _Record(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "Reading schema version must be an integer"
            raise ValueError(msg)
        return value

    @property
    def reference(self) -> ArtifactReference:
        """Identify complete typed content, including all failure records."""
        return ArtifactReference.from_bytes(
            self.model_dump_json().encode(), kind=f"op-infer-{type(self).__name__}-v1"
        )


class MetricEstimate(_Record):
    """A consumer-defined scalar and optional interval on the usable cohort.

    ``method`` identifies the formula, assumptions, interval family/level and
    any clipping or other options. None explicitly denotes unavailable evidence.
    """

    quantity: _Text
    metric: _Text
    method: ArtifactReference
    n_usable: _Count
    point: _Finite | None
    lower: _Finite | None = None
    upper: _Finite | None = None

    @model_validator(mode="after")
    def _interval(self) -> Self:
        if (self.lower is None) != (self.upper is None) or (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            msg = "Metric intervals require both ordered finite endpoints"
            raise ValueError(msg)
        if self.n_usable == 0 and any(
            value is not None for value in (self.point, self.lower, self.upper)
        ):
            msg = "An empty cohort cannot supply a metric estimate"
            raise ValueError(msg)
        return self


class CalibrationRule(_Record):
    """Inclusive bounds with an explicit view of a named metric.

    ``band`` uses the point in the point reading and interval overlap in the
    interval reading. ``point``, ``lower`` and ``upper`` always use that fixed
    metric view, for example a practical cap on a supplied lower endpoint.
    An interval's length is never substituted for a consumer's width statistic.
    """

    name: _Text
    quantity: _Text
    metric: _Text
    method: ArtifactReference
    lower: _Finite | None = None
    upper: _Finite | None = None
    view: Literal["band", "point", "lower", "upper"] = "band"
    applies_to: tuple[_ReadingName, ...] = ("point", "interval")

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if (
            (self.lower is None and self.upper is None)
            or (
                self.lower is not None
                and self.upper is not None
                and self.lower > self.upper
            )
            or not self.applies_to
            or len(set(self.applies_to)) != len(self.applies_to)
        ):
            msg = "Rules require ordered bounds and unique nonempty readings"
            raise ValueError(msg)
        return self


class ReplicateFailure(_Record):
    """Retain a planned replicate's failure and its diagnostic reference."""

    replicate: _Text
    reason: _Text
    evidence: ArtifactReference


class CalibrationEvidence(_Record):
    """Exact usable and failed identities, with no omitted planned replicates.

    Each metric must describe the same complete usable cohort. The reader checks
    identities/counts; the producer owns numerical correctness and independence.
    """

    context: ArtifactReference
    source: ArtifactReference
    usable: tuple[_Text, ...]
    failures: tuple[ReplicateFailure, ...]
    metrics: tuple[MetricEstimate, ...]

    @model_validator(mode="after")
    def _cohort(self) -> Self:
        identities = self.usable + tuple(f.replicate for f in self.failures)
        keys = tuple((m.quantity, m.metric) for m in self.metrics)
        if (
            len(set(identities)) != len(identities)
            or len(set(keys)) != len(keys)
            or any(m.n_usable != len(self.usable) for m in self.metrics)
        ):
            msg = "Evidence requires unique replicates/metrics and one usable cohort"
            raise ValueError(msg)
        return self


class CalibrationPolicy(_Record):
    """Prespecify both readings, the governing choice and complete trial bank.

    Failure allowance counts distinct quantities, not individual criteria. Every
    reading must cover the same quantity set. A missing metric, exceeded failure
    budget or replicate floor prevents a pass. External records establish when
    results became visible; local registration cannot authenticate chronology.
    """

    name: _Text
    context: ArtifactReference
    governing: _ReadingName
    rationale: _Text
    replicate_ids: tuple[_Text, ...]
    minimum_usable: Annotated[int, Field(ge=1)]
    max_failure_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    max_failed_quantities: _Count
    must_pass: tuple[_Text, ...]
    rules: tuple[CalibrationRule, ...]
    amendment: GateAmendment | None = None

    @model_validator(mode="after")
    def _design(self) -> Self:
        quantities = {r.quantity for r in self.rules}
        if (
            not self.replicate_ids
            or len(set(self.replicate_ids)) != len(self.replicate_ids)
            or self.minimum_usable > len(self.replicate_ids)
            or not self.rules
            or len({r.name for r in self.rules}) != len(self.rules)
            or len(set(self.must_pass)) != len(self.must_pass)
            or not set(self.must_pass) <= quantities
            or self.max_failed_quantities > len(quantities)
            or any(
                {r.quantity for r in self.rules if reading in r.applies_to}
                != quantities
                for reading in ("point", "interval")
            )
        ):
            msg = (
                "Policy requires a fixed bank, complete readings "
                "and declared quantities"
            )
            raise ValueError(msg)
        methods: dict[tuple[str, str], ArtifactReference] = {}
        for rule in self.rules:
            key = (rule.quantity, rule.metric)
            if key in methods and methods[key] != rule.method:
                msg = "A named metric must have one declared method"
                raise ValueError(msg)
            methods[key] = rule.method
        return self

    def write(self, path: str | Path) -> ArtifactReference:
        """Atomically register this policy; a different policy needs a new file."""
        write_immutable_json(
            path,
            self.model_dump_json().encode(),
            conflict="Saved calibration policy differs; write an amendment separately",
        )
        return self.reference

    def evaluate(
        self, path: str | Path, evidence: CalibrationEvidence
    ) -> CalibrationRecord:
        """Require exact saved criteria, then retain evidence and both readings.

        Returns:
            A record whose decisions are derived again on every read.

        Raises:
            ValueError: Saved policy, context, methods or planned cohort differ.
            FileNotFoundError: No policy is registered at the supplied path.
        """
        saved = CalibrationPolicy.model_validate_json(Path(path).read_bytes())
        if saved != self:
            msg = "Saved calibration policy differs"
            raise ValueError(msg)
        return CalibrationRecord(policy=self, evidence=evidence)


class RuleOutcome(_Record):
    """One criterion's complete scalar or interval input and verdict."""

    rule: str
    lower: _Finite | None
    upper: _Finite | None
    status: Literal["pass", "fail", "unavailable"]


class Reading(_Record):
    """One alternative panel, including adequacy reasons and failed quantities."""

    name: _ReadingName
    state: Literal["pass", "fail", "insufficient_evidence"]
    reasons: tuple[str, ...]
    failed_quantities: tuple[str, ...]
    outcomes: tuple[RuleOutcome, ...]


def _outcome(
    rule: CalibrationRule, metric: MetricEstimate | None, reading: str
) -> RuleOutcome:
    if metric is None:
        low = high = None
    elif rule.view == "band" and reading == "interval":
        low, high = metric.lower, metric.upper
    else:
        value = getattr(metric, "point" if rule.view == "band" else rule.view)
        low = high = value
    status: Literal["pass", "fail", "unavailable"] = "unavailable"
    if low is not None and high is not None:
        passed = (rule.lower is None or high >= rule.lower) and (
            rule.upper is None or low <= rule.upper
        )
        status = "pass" if passed else "fail"
    return RuleOutcome(rule=rule.name, lower=low, upper=high, status=status)


class CalibrationRecord(_Record):
    """Retain full policy/evidence; derive both panels without mutable verdicts."""

    policy: CalibrationPolicy
    evidence: CalibrationEvidence

    @model_validator(mode="after")
    def _match(self) -> Self:
        spec, data = self.policy, self.evidence
        identities = data.usable + tuple(f.replicate for f in data.failures)
        methods = {(r.quantity, r.metric): r.method for r in spec.rules}
        if data.context != spec.context or set(identities) != set(spec.replicate_ids):
            msg = "Evidence context or complete planned replicate bank differs"
            raise ValueError(msg)
        for metric in data.metrics:
            key = (metric.quantity, metric.metric)
            if key not in methods or metric.method != methods[key]:
                msg = "Evidence metric or declared method differs"
                raise ValueError(msg)
        return self

    def readings(self) -> tuple[Reading, Reading]:
        """Compute both panels, preserving failures even when evidence is inadequate."""
        spec, data = self.policy, self.evidence
        metrics = {(m.quantity, m.metric): m for m in data.metrics}
        common = []
        if len(data.usable) < spec.minimum_usable:
            common.append("below_minimum_usable")
        if len(data.failures) > spec.max_failure_fraction * len(spec.replicate_ids):
            common.append("failure_budget_exceeded")
        panels = []
        for name in ("point", "interval"):
            rules = tuple(r for r in spec.rules if name in r.applies_to)
            outcomes = tuple(
                _outcome(r, metrics.get((r.quantity, r.metric)), name) for r in rules
            )
            failed = tuple(
                sorted(
                    {
                        r.quantity
                        for r, o in zip(rules, outcomes, strict=True)
                        if o.status == "fail"
                    }
                )
            )
            reasons = tuple(common) + tuple(
                f"unavailable:{o.rule}" for o in outcomes if o.status == "unavailable"
            )
            state: Literal["pass", "fail", "insufficient_evidence"] = (
                "insufficient_evidence"
                if reasons
                else (
                    "fail"
                    if len(failed) > spec.max_failed_quantities
                    or set(failed) & set(spec.must_pass)
                    else "pass"
                )
            )
            panels.append(
                Reading(
                    name=name,
                    state=state,
                    reasons=reasons,
                    failed_quantities=failed,
                    outcomes=outcomes,
                )
            )
        return panels[0], panels[1]

    @property
    def governing_reading(self) -> Reading:
        """Return the prospectively selected panel, preserving the alternative."""
        return self.readings()[0 if self.policy.governing == "point" else 1]


def wilson_method(*, confidence: float) -> ArtifactReference:
    """Identify the ordinary independent-binomial Wilson coverage procedure.

    Returns:
        A method reference including confidence level and cohort assumptions.

    Raises:
        ValueError: Confidence is not a finite float strictly between zero and one.
    """
    if type(confidence) is not float or not 0 < confidence < 1:
        msg = "Wilson confidence must be a float strictly between zero and one"
        raise ValueError(msg)
    return ArtifactReference.from_bytes(
        json.dumps(
            {
                "family": "binomial-proportion",
                "interval": "wilson",
                "confidence": confidence,
                "cohort": "fixed-independent-usable-trials",
            },
            sort_keys=True,
        ).encode(),
        kind="op-infer-coverage-method-v1",
    )


def wilson_coverage(
    *, quantity: str, metric: str, successes: int, trials: int, confidence: float
) -> MetricEstimate:
    """Wrap SciPy's Wilson interval without substituting a new statistical kernel.

    Returns:
        Explicitly unavailable evidence at zero trials, otherwise a point and CI.

    Raises:
        ValueError: Counts are nonintegers or outside their binomial support.
    """
    method = wilson_method(confidence=confidence)
    if (
        type(successes) is not int
        or type(trials) is not int
        or not 0 <= successes <= trials
    ):
        msg = "Coverage requires integer counts with 0 <= successes <= trials"
        raise ValueError(msg)
    if trials == 0:
        return MetricEstimate(
            quantity=quantity, metric=metric, method=method, n_usable=0, point=None
        )
    interval = binomtest(successes, trials).proportion_ci(
        confidence_level=confidence, method="wilson"
    )
    return MetricEstimate(
        quantity=quantity,
        metric=metric,
        method=method,
        n_usable=trials,
        point=successes / trials,
        lower=float(interval.low),
        upper=float(interval.high),
    )
