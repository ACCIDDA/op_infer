"""Design recovery, design power and value of information.

Implemented: bounded fixed-bank held-out simulate/refit comparison.
General gate orchestration, Fisher screens and EVSI remain planned.

Scope: DesignStudy on trade_study: simulate under each generating model with the
training acceptance rule, refit with design-aware estimators, score with
op_infer.compare, report recovery with Wilson intervals, three-way outcomes and counted
failures; replicate sizing from target precision; nested designs with common random
numbers; Fisher screens (explicitly screens) feeding simulate-and-refit or EVSI.

"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from op_infer.calibrate import _write_json_atomically
from op_infer.compare import UnitPredictiveScores, paired_predictive_contrast
from op_infer.posterior import ArtifactReference, Provenance
from op_infer.rng import RandomKey
from op_infer.sbc import _require_manifest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from trade_study import ResultsTable


__all__ = [
    "DesignDefinition",
    "DesignFit",
    "DesignGate",
    "DesignObservation",
    "DesignStudy",
]


@dataclass(frozen=True)
class DesignObservation:
    """Separated fitting/held-out data with one declared predictive target."""

    training: object
    held_out: object
    target: ArtifactReference


@dataclass(frozen=True)
class DesignDefinition:
    """Project a shared full cohort onto a design and record its cost/burden."""

    observe: Callable[[object], DesignObservation]
    cost: float


@dataclass(frozen=True)
class DesignGate:
    """Caller-supplied calibration evidence for one design/model estimator.

    This is an execution prerequisite for replicated design studies, not the general
    Gate/GateRecord
    API. Underpowered passes never count as calibrated estimators here.
    """

    design: str
    model: str
    estimator: ArtifactReference
    calibration: ArtifactReference
    label: Literal["pass", "fail", "underpowered pass"]


@dataclass(frozen=True)
class DesignFit:
    """Reporting fit, including joint latent draws, with explicit adequacy."""

    posterior: object
    estimator: ArtifactReference
    adequate: bool
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _Cohort:
    data: object
    error: str | None = None


def _record_path(directory: Path, design: str, generator: str, index: int) -> Path:
    """Return a stable portable artifact name for one independent cohort trial."""
    return directory / f"design_{design}_{generator}_{index:020d}.json"


@dataclass
class _DesignWorld:
    study: DesignStudy
    banks: Mapping[tuple[str, int], _Cohort]
    directory: Path
    identity: str
    seed: int

    def generate(self, config: dict[str, Any]) -> tuple[float, None]:
        """Execute or resume one design/generator/cohort with counted failures.

        Returns:
            A failure indicator for trade-study's task scheduler.

        Raises:
            ValueError: An existing artifact has mismatched provenance.
            RuntimeError: Another worker owns this cohort/design record.
        """
        design, generator, index = (
            str(config["design"]),
            str(config["generator"]),
            int(config["cohort"]),
        )
        path = _record_path(self.directory, design, generator, index)
        meta = {
            "contract": self.identity,
            "design": design,
            "generator": generator,
            "cohort": index,
        }
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("_meta") != meta or record.get("schema_version") != 1:
                msg = "Design artifact provenance/schema mismatch"
                raise ValueError(msg)
            return float(record["outcome"] == "failed"), None
        lock = path.with_suffix(".lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            msg = "Design cohort already has an active/stale worker lock"
            raise RuntimeError(msg) from None
        os.close(descriptor)
        try:
            try:
                record = self._evaluate(design, generator, index)
                json.dumps(record, allow_nan=False)
            except Exception as error:  # noqa: BLE001 - retain failures in denominators
                record = {
                    "outcome": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            record.update(schema_version=1, _meta=meta)
            _write_json_atomically(path, record)
            return float(record["outcome"] == "failed"), None
        finally:
            lock.unlink()

    def _evaluate(self, design: str, generator: str, index: int) -> dict[str, Any]:
        """Refit every candidate on fitting data and score only held-out data."""
        cohort = self.banks[(generator, index)]
        if cohort.error is not None:
            return {"outcome": "failed", "reason": cohort.error}
        observed = self.study.designs[design].observe(cohort.data)
        scores, diagnostics = {}, {}
        for model in self.study.models:
            gate = self.study.gates[(design, model)]
            if gate.label != "pass":
                return {
                    "outcome": "failed",
                    "reason": f"calibration {gate.label}: {design}/{model}",
                }
            fit_key = RandomKey(
                seed=self.seed, stream=f"design.fit.{design}.{model}", index=index
            )
            fitted = self.study.fit(model, design, observed.training, fit_key)
            if fitted.estimator != gate.estimator or fitted.adequate is not True:
                return {
                    "outcome": "failed",
                    "reason": f"estimator mismatch or inadequate fit: {design}/{model}",
                }
            score_key = RandomKey(
                seed=self.seed, stream=f"design.predict.{design}.{model}", index=index
            )
            scored = self.study.score(model, fitted, observed.held_out, score_key)
            if scored.target != observed.target:
                msg = "Held-out predictive target mismatch"
                raise ValueError(msg)
            scores[model] = scored
            diagnostics[model] = dict(fitted.diagnostics)
        contrasts = {
            model: asdict(
                paired_predictive_contrast(
                    scores[generator], scored, threshold_se=self.study.threshold_se
                )
            )
            for model, scored in scores.items()
            if model != generator
        }
        outcomes = [entry["outcome"] for entry in contrasts.values()]
        outcome = (
            "failed"
            if "failed" in outcomes
            else "generator"
            if all(o == "generator" for o in outcomes)
            else "alternative"
            if "alternative" in outcomes
            else "indistinguishable"
        )
        return {
            "outcome": outcome,
            "contrasts": contrasts,
            "fit_diagnostics": diagnostics,
            "target": observed.target.model_dump(mode="json"),
            "scores": {
                model: {
                    "unit_ids": score.unit_ids,
                    "unit_kind": score.unit_kind,
                    "values": [
                        float(v) if np.isfinite(v) else None
                        for v in np.asarray(score.values)
                    ],
                }
                for model, score in scores.items()
            },
        }


class _FailureScorer:
    @staticmethod
    def score(truth: float, observed: None, config: dict[str, Any]) -> dict[str, float]:
        """Return a stored failure flag; recovery summaries use complete records."""
        del observed, config
        return {"failed": truth}


@dataclass(frozen=True)
class DesignStudy:
    """Bounded prespecified simulate/refit/predict workflow on trade-study.

    Generators receive separate parameter/simulation keys. Each generated
    cohort is shared across designs, preserving common random numbers. Callbacks
    own model-specific inference and proper predictive scoring; every alternative
    is fitted on training data before scoring the same held-out target.
    """

    models: tuple[str, ...]
    designs: Mapping[str, DesignDefinition]
    generate: Callable[[str, RandomKey, RandomKey], object]
    fit: Callable[[str, str, object, RandomKey], DesignFit]
    score: Callable[[str, DesignFit, object, RandomKey], UnitPredictiveScores]
    gates: Mapping[tuple[str, str], DesignGate]
    provenance: Provenance
    threshold_se: float = 2.0

    def __post_init__(self) -> None:
        """Validate the complete model/design/gate bank before simulation.

        Raises:
            ValueError: The prespecified design contract is incomplete or invalid.
        """
        names = (*self.models, *self.designs)
        if (
            len(self.models) < 2
            or len(set(self.models)) != len(self.models)
            or not self.designs
        ):
            msg = (
                "A DesignStudy requires distinct candidate models and nonempty designs"
            )
            raise ValueError(msg)
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names):
            msg = "Model/design names must be portable identifiers"
            raise ValueError(msg)
        if (
            any(not np.isfinite(d.cost) or d.cost < 0 for d in self.designs.values())
            or not np.isfinite(self.threshold_se)
            or self.threshold_se <= 0
        ):
            msg = "Costs must be finite/non-negative and threshold_se positive"
            raise ValueError(msg)
        for design in self.designs:
            for model in self.models:
                gate = self.gates.get((design, model))
                if (
                    gate is None
                    or (gate.design, gate.model) != (design, model)
                    or gate.label not in {"pass", "fail", "underpowered pass"}
                ):
                    msg = f"Invalid calibration evidence for {design}/{model}"
                    raise ValueError(msg)
        Provenance.model_validate(self.provenance)

    def _identity(self, directory: Path, seed: int) -> str:
        """Return an immutable artifact contract independent of shard layout."""
        contract = {
            "schema_version": 1,
            "provenance": self.provenance.model_dump(mode="json"),
            "models": self.models,
            "designs": {name: d.cost for name, d in self.designs.items()},
            "seed": seed,
            "threshold_se": self.threshold_se,
            "gates": {
                f"{d}/{m}": {
                    "estimator": g.estimator.model_dump(mode="json"),
                    "calibration": g.calibration.model_dump(mode="json"),
                    "label": g.label,
                }
                for (d, m), g in self.gates.items()
            },
        }
        return _require_manifest(directory, contract, filename="design-manifest.meta")

    def run(
        self,
        directory: str | Path,
        *,
        replicate_ids: Sequence[int],
        seed: int,
        n_jobs: int = 1,
    ) -> None:
        """Run a prespecified bank, resuming complete successes and failures.

        Disjoint ID shards share a contract. No retry or result-dependent stop
        changes the population. Generators implement the training acceptance rule
        and likelihood support; this workflow applies no extra filter.

        Raises:
            ImportError: The optional study extra is absent.
            ValueError: Task addresses or existing provenance are invalid.
        """
        try:
            from trade_study import Direction, Observable, run_grid  # noqa: PLC0415 - optional dependency
        except ImportError as error:
            msg = "DesignStudy requires pip install 'op_infer[study]'"
            raise ImportError(msg) from error
        ids = tuple(replicate_ids)
        if not ids or len(set(ids)) != len(ids):
            msg = "Provide distinct nonempty independent cohort IDs"
            raise ValueError(msg)
        for index in ids:
            RandomKey(seed=seed, stream="design.validation", index=index)
        directory = Path(directory)
        identity = self._identity(directory, seed)
        banks = {}
        for model in self.models:
            for index in ids:
                if all(
                    _record_path(directory, d, model, index).exists()
                    for d in self.designs
                ):
                    continue
                parameter_key = RandomKey(
                    seed=seed, stream=f"design.parameters.{model}", index=index
                )
                simulation_key = RandomKey(
                    seed=seed, stream=f"design.simulation.{model}", index=index
                )
                try:
                    banks[(model, index)] = _Cohort(
                        self.generate(model, parameter_key, simulation_key)
                    )
                except Exception as error:  # noqa: BLE001 - retain shared generator failures
                    banks[(model, index)] = _Cohort(
                        None, f"{type(error).__name__}: {error}"
                    )
        configs = [
            {"design": design, "generator": model, "cohort": i}
            for design in self.designs
            for model in self.models
            for i in ids
        ]
        run_grid(
            _DesignWorld(self, banks, directory, identity, seed),
            _FailureScorer(),
            configs,
            [Observable("failed", Direction.MINIMIZE)],
            n_reps=1,
            n_jobs=n_jobs,
        )

    def summarize(
        self,
        directory: str | Path,
        *,
        replicate_ids: Sequence[int],
        seed: int,
        confidence: float = 0.95,
    ) -> ResultsTable:
        """Report three-way pipeline outcomes plus failures over all planned IDs.

        Returns:
            A trade-study table with cost and marginal Wilson intervals. Each
            independent generated cohort is one replicate; bootstrap resamples
            are not new replicates. Failed/missing trials stay in the denominator.

        Raises:
            ValueError: No planned IDs exist or artifact provenance differs.
        """
        from trade_study import ResultsTable, proportion_interval  # noqa: PLC0415 - optional dependency

        ids = tuple(replicate_ids)
        if not ids or len(set(ids)) != len(ids):
            msg = "Summaries require distinct nonempty planned cohort IDs"
            raise ValueError(msg)
        for index in ids:
            RandomKey(seed=seed, stream="design.validation", index=index)
        directory = Path(directory)
        identity = self._identity(directory, seed)
        configs, values, metadata = [], [], []
        for design, definition in self.designs.items():
            for model in self.models:
                counts = dict.fromkeys(
                    ("generator", "alternative", "indistinguishable", "failed"), 0
                )
                missing = 0
                for index in ids:
                    path = _record_path(directory, design, model, index)
                    if not path.exists():
                        counts["failed"] += 1
                        missing += 1
                        continue
                    record = json.loads(path.read_text())
                    if (
                        record.get("schema_version") != 1
                        or record.get("_meta")
                        != {
                            "contract": identity,
                            "design": design,
                            "generator": model,
                            "cohort": index,
                        }
                        or record.get("outcome") not in counts
                    ):
                        msg = "Design artifact provenance/outcome mismatch"
                        raise ValueError(msg)
                    counts[record["outcome"]] += 1
                intervals = {
                    name: asdict(
                        proportion_interval(k, len(ids), confidence=confidence)
                    )
                    for name, k in counts.items()
                }
                configs.append({"design": design, "generator": model})
                values.append(
                    [*(counts[name] / len(ids) for name in counts), definition.cost]
                )
                metadata.append(
                    {
                        "planned": len(ids),
                        "failed": counts["failed"],
                        "missing": missing,
                        "replicate_unit": "independent_generated_cohort",
                        "wilson": intervals,
                        "probability_scope": "all_prespecified_pipeline_trials",
                        "multiplicity": "marginal Wilson; prespecified joint decision",
                        "gate_power_assessed": False,
                    }
                )
        return ResultsTable(
            configs,
            np.asarray(values, dtype=float),
            [
                "favor_generator",
                "favor_alternative",
                "indistinguishable",
                "failed",
                "cost",
            ],
            metadata=metadata,
        )
