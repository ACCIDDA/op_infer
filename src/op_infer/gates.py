"""Prespecified, seeded gate records with sampler adequacy and amendments.

This module records scalar criteria and their
application. Consumers supply diagnostics and qualify statistical operating
characteristics; neither a saved specification nor a pass certifies calibration.
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime  # noqa: TC003 - Pydantic resolves the schema at runtime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from op_infer.posterior import ArtifactReference
from op_infer.rng import RandomKey  # noqa: TC001 - Pydantic resolves the schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["Gate", "GateAmendment", "GateOutcome", "GateRecord", "GateSet"]

_Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
_Finite = Annotated[float, Field(allow_inf_nan=False)]


class _Record(BaseModel):
    """Keep specifications and evidence immutable through JSON round trips."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Gate(_Record):
    """One inclusive scalar criterion, with a separately named input metric.

    ``adequacy`` criteria govern whether results are readable, including sampler
    diagnostics, replicate floors and failure limits. ``decision`` criteria
    govern the substantive reading. At least one finite bound is required.
    Missing/nonfinite values are unavailable and cannot pass.

    Examples:
        >>> criterion = Gate(
        ...     name="bias",
        ...     metric="mean_pit_error",
        ...     role="decision",
        ...     lower=-0.05,
        ...     upper=0.05,
        ... )
        >>> criterion.lower, criterion.upper
        (-0.05, 0.05)
    """

    name: _Text
    metric: _Text
    role: Literal["adequacy", "decision"]
    lower: _Finite | None = None
    upper: _Finite | None = None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.lower is None and self.upper is None:
            msg = "A gate requires at least one bound"
            raise ValueError(msg)
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            msg = "Gate lower bound exceeds upper bound"
            raise ValueError(msg)
        return self


class GateAmendment(_Record):
    """Disclose a prospective revision and the evidence already seen.

    ``supersedes`` identifies the prior saved GateSet. An empty
    ``seen_before_amendment`` explicitly records that no result artifacts were
    seen. The timestamp and disclosure are supplied by the author; the package
    cannot authenticate chronology or establish independence of fresh evidence.
    """

    supersedes: ArtifactReference
    recorded_at: datetime
    reason: _Text
    seen_before_amendment: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def _timezone(self) -> Self:
        if self.recorded_at.utcoffset() is None:
            msg = "Amendments require a timezone-aware timestamp"
            raise ValueError(msg)
        return self


class GateSet(_Record):
    """Stage-specific criteria saved before invoking a seeded evaluator.

    The context reference identifies the estimator/design/prior contract;
    evidence identifies exact diagnostic inputs. The callback receives a fresh
    immutable RandomKey on each call. It owns native RNG lowering and must use
    that key for all stochastic diagnostics. Replaying external mutable state
    or ignoring the key is outside this bookkeeping contract.

    ``write`` refuses a different existing specification. ``evaluate`` reads
    and checks that saved specification before calling the evaluator. External
    version control establishes whether results were unseen at registration.

    Examples:
        >>> key = RandomKey(seed=7, stream="gates.validation")
        >>> context = ArtifactReference.from_bytes(b"toy", kind="design-v1")
        >>> criteria = GateSet(
        ...     name="toy",
        ...     stage="validation",
        ...     context=context,
        ...     key=key,
        ...     gates=(
        ...         Gate(
        ...             name="adequate", metric="ess", role="adequacy", lower=100.0
        ...         ),
        ...         Gate(
        ...             name="error",
        ...             metric="error",
        ...             role="decision",
        ...             lower=-0.05,
        ...             upper=0.05,
        ...         ),
        ...     ),
        ... )
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "criteria.json"
        ...     _ = criteria.write(path)
        ...     result = criteria.evaluate(
        ...         path,
        ...         lambda key: {"ess": 1000.0, "error": 0.01},
        ...         evidence=context,
        ...         context=context,
        ...     )
        >>> result.readable, result.passed
        (True, True)
        >>> restored = GateRecord.model_validate_json(result.model_dump_json())
        >>> restored.readable, restored.passed
        (True, True)
    """

    schema_version: Literal[1] = 1
    name: _Text
    stage: _Text
    context: ArtifactReference
    key: RandomKey
    gates: tuple[Gate, ...]
    amendment: GateAmendment | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "schema_version must be an integer"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _criteria(self) -> Self:
        if len({g.name for g in self.gates}) != len(self.gates):
            msg = "Gate names must be unique"
            raise ValueError(msg)
        if {g.role for g in self.gates} != {"adequacy", "decision"}:
            msg = "A GateSet requires adequacy and decision criteria"
            raise ValueError(msg)
        return self

    def reference(self) -> ArtifactReference:
        """Return the content identity of this complete specification."""
        return ArtifactReference.from_bytes(
            self.model_dump_json().encode(), kind="op-infer-gate-set-v1"
        )

    def write(self, path: str | Path) -> ArtifactReference:
        """Atomically save criteria, refusing to replace a different contract.

        Returns:
            The saved specification's content reference.

        Raises:
            ValueError: The destination already contains different bytes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump_json().encode()
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    msg = (
                        "Saved gate specification differs; "
                        "write an amendment separately"
                    )
                    raise ValueError(msg) from None
        finally:
            temporary.unlink()
        return self.reference()

    def amend(
        self,
        *,
        gates: tuple[Gate, ...],
        key: RandomKey,
        disclosure: GateAmendment,
    ) -> GateSet:
        """Create revised criteria with an explicit prior-artifact reference.

        Returns:
            A new specification to save separately before fresh evaluation.

        Raises:
            ValueError: The disclosure does not reference this specification.
        """
        if disclosure.supersedes != self.reference():
            msg = "Amendment must reference the specification it supersedes"
            raise ValueError(msg)
        return GateSet(
            name=self.name,
            stage=self.stage,
            context=self.context,
            key=key,
            gates=gates,
            amendment=disclosure,
        )

    def evaluate(
        self,
        path: str | Path,
        evaluator: Callable[[RandomKey], Mapping[str, float | None]],
        *,
        evidence: ArtifactReference,
        context: ArtifactReference,
    ) -> GateRecord:
        """Check saved criteria/context, then evaluate using the declared key.

        Returns:
            A complete record with separate readable and passed decisions.
            Exceptions from the evaluator propagate; they cannot produce a pass.

        Raises:
            ValueError: Saved criteria or evidence context do not match.
            FileNotFoundError: No criteria have been saved at the supplied path.
        """
        saved = GateSet.model_validate_json(Path(path).read_bytes())
        if saved != self or context != self.context:
            msg = "Saved gate specification or evidence context differs"
            raise ValueError(msg)
        values = evaluator(self.key)
        outcomes = []
        for gate in self.gates:
            raw = values.get(gate.metric)
            value = None if raw is None or not math.isfinite(raw) else float(raw)
            outcomes.append(GateOutcome(name=gate.name, value=value))
        return GateRecord(
            specification=self, evidence=evidence, outcomes=tuple(outcomes)
        )


class GateOutcome(_Record):
    """The scalar read for one criterion; None means unavailable."""

    name: _Text
    value: _Finite | None


class GateRecord(_Record):
    """Immutable criteria and evidence; derive readable/passed on every load.

    ``readable`` requires all adequacy criteria to pass. ``passed`` additionally
    requires all decision criteria. An unavailable or underpowered reading
    cannot pass when its requisite evidence/floor is an adequacy criterion.
    The record retains values for failed and unreadable runs.
    """

    specification: GateSet
    evidence: ArtifactReference
    outcomes: tuple[GateOutcome, ...]

    @model_validator(mode="after")
    def _names(self) -> Self:
        if tuple(o.name for o in self.outcomes) != tuple(
            g.name for g in self.specification.gates
        ):
            msg = "Outcomes must match every criterion in specification order"
            raise ValueError(msg)
        return self

    def statuses(self) -> tuple[Literal["pass", "fail", "unavailable"], ...]:
        """Return criterion verdicts, preserving unavailable outcomes."""
        statuses: list[Literal["pass", "fail", "unavailable"]] = []
        for gate, outcome in zip(self.specification.gates, self.outcomes, strict=True):
            value = outcome.value
            if value is None:
                statuses.append("unavailable")
            else:
                passed = (gate.lower is None or value >= gate.lower) and (
                    gate.upper is None or value <= gate.upper
                )
                statuses.append("pass" if passed else "fail")
        return tuple(statuses)

    @property
    def readable(self) -> bool:
        """Whether all declared adequacy criteria pass."""
        return all(
            status == "pass"
            for gate, status in zip(
                self.specification.gates, self.statuses(), strict=True
            )
            if gate.role == "adequacy"
        )

    @property
    def passed(self) -> bool:
        """Whether a readable result passes every decision criterion."""
        return self.readable and all(status == "pass" for status in self.statuses())
