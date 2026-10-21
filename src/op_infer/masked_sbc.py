"""Masked-unit calibration on the existing fixed-bank SBC runner.

Full cohorts are simulated before masking. Projects supply the reporting and
optional comparison estimators; scheduling and fitting are not reimplemented.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from op_infer.calibrate import POWER_FLOOR, _rank_settings, aggregate
from op_infer.posterior import ArtifactReference, Provenance
from op_infer.rng import RandomKey
from op_infer.sbc import (
    SBCEstimand,
    SBCFit,
    _digest_array,
    _fit_draws,
    _require_manifest,
    run_sbc,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MaskedSBCExecution",
    "MaskedSBCPlan",
    "MaskedUnit",
    "run_masked_sbc",
    "summarize_masked_sbc",
]

_Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
_Width = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_ID = Annotated[int, Field(ge=0, lt=2**64)]
_RESERVED = "masked_sbc"


class _Record(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )


class MaskedUnit(_Record):
    """One unit on the observation array's first axis, with a declared stratum."""

    name: _Text
    stratum: _Text


class MaskedSBCPlan(_Record):
    """Prespecify the complete bank and ordered units before running shards.

    Only units with observed cells are eligible. An optional stratum restricts
    the pool. Replicate ID modulo pool length selects the removed unit, so
    selection is independent of shard/worker order and outcomes. Arbitrary ID
    subsets need not balance the units; the summary retains actual allocations.

    Examples:
        >>> plan = MaskedSBCPlan(
        ...     units=(
        ...         MaskedUnit(name="a", stratum="A"),
        ...         MaskedUnit(name="b", stratum="B"),
        ...     ),
        ...     replicate_ids=(0, 1, 2, 3),
        ...     seed=7,
        ... )
        >>> plan.replicate_ids
        (0, 1, 2, 3)
    """

    units: tuple[MaskedUnit, ...]
    replicate_ids: tuple[_ID, ...]
    seed: Annotated[int, Field(ge=0, lt=2**128)]
    stratum: _Text | None = None

    @model_validator(mode="after")
    def _layout(self) -> Self:
        if (
            len(self.units) < 2
            or len({unit.name for unit in self.units}) != len(self.units)
            or not self.replicate_ids
            or len(set(self.replicate_ids)) != len(self.replicate_ids)
            or (
                self.stratum is not None
                and self.stratum not in {u.stratum for u in self.units}
            )
        ):
            msg = (
                "Masked SBC requires distinct units/IDs and a declared eligible stratum"
            )
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class MaskedSBCExecution:
    """Shard completion counts and the independently retainable full-bank contract."""

    contract: ArtifactReference
    requested: int
    written: int
    resumed: int
    failed: int


class _Comparison(_Record):
    name: _Text
    posterior_sd: _Width
    prior_sd: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    baseline_sd: _Width | None


class _Rank(_Record):
    pit: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    contraction: _Width


class _Evidence(_Record):
    unit_index: Annotated[int, Field(ge=0)]
    reporting_draws_used: Annotated[int, Field(ge=2)]
    comparisons: tuple[_Comparison, ...]
    baseline_status: Literal["available", "unavailable", "not_requested"]
    baseline_reason: _Text | None
    baseline_diagnostics: dict[str, Any] | None

    @model_validator(mode="after")
    def _baseline(self) -> Self:
        available = self.baseline_status == "available"
        if (
            (self.baseline_reason is None) != available
            or (self.baseline_diagnostics is not None) != available
        ) or any(
            (row.baseline_sd is not None) != available for row in self.comparisons
        ):
            msg = "Baseline availability, widths and reasons disagree"
            raise ValueError(msg)
        return self


def _widths(
    fit: SBCFit, estimands: Mapping[str, SBCEstimand]
) -> tuple[dict[str, float], int]:
    draws = _fit_draws(fit)
    count = len(next(iter(draws.values())))
    widths = {}
    for name, estimand in sorted(estimands.items()):
        values = np.asarray(estimand.value(draws), dtype=np.float64)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            msg = "Masked estimands need one finite value per retained draw"
            raise ValueError(msg)
        width = float(np.std(values, ddof=1))
        if not math.isfinite(width) or not math.isfinite(width / estimand.prior_sd):
            msg = "Posterior widths and prior ratios must be finite"
            raise ValueError(msg)
        widths[name] = width
    return widths, count


@dataclass
class _MaskedEstimator:
    estimator: Callable[[ArrayLike, NDArray[np.bool_], RandomKey], SBCFit]
    baseline: Callable[[ArrayLike, NDArray[np.bool_], RandomKey], SBCFit] | None
    estimands: Mapping[str, SBCEstimand]
    mask: NDArray[np.bool_]
    unit_index: int

    def __call__(self, data: ArrayLike, key: RandomKey) -> SBCFit:
        values = np.asarray(data)
        if values.shape != self.mask.shape:
            msg = "Masked simulation must preserve the declared cohort shape"
            raise ValueError(msg)
        fit = self.estimator(values.copy(), self.mask.copy(), key)
        if _RESERVED in fit.diagnostics:
            msg = "masked_sbc is reserved for the runner's diagnostic metadata"
            raise ValueError(msg)
        widths, count = _widths(fit, self.estimands)
        baseline_widths: dict[str, float] = {}
        baseline_diagnostics = None
        status: Literal["available", "unavailable", "not_requested"] = "not_requested"
        reason: str | None = "baseline_not_requested"
        if self.baseline is not None:
            try:
                baseline_fit = self.baseline(
                    values.copy(),
                    self.mask.copy(),
                    RandomKey(
                        seed=key.seed, stream="estimator.baseline", index=key.index
                    ),
                )
                baseline_widths, baseline_count = _widths(baseline_fit, self.estimands)
                baseline_diagnostics = json.loads(
                    json.dumps(
                        {
                            "diagnostics": baseline_fit.diagnostics,
                            "sample_kind": baseline_fit.kind,
                            "adequate": baseline_fit.adequate,
                            "effective_sample_size": baseline_fit.effective_sample_size,
                            "draws_used": baseline_count,
                        },
                        allow_nan=False,
                    )
                )
                status, reason = "available", None
            except Exception as error:  # noqa: BLE001 - retain comparison failure, preserve reporting fit
                baseline_widths = {}
                baseline_diagnostics = None
                status = "unavailable"
                reason = f"{type(error).__name__}: {error}"
        evidence = _Evidence(
            unit_index=self.unit_index,
            reporting_draws_used=count,
            comparisons=tuple(
                _Comparison(
                    name=name,
                    posterior_sd=width,
                    prior_sd=self.estimands[name].prior_sd,
                    baseline_sd=baseline_widths.get(name),
                )
                for name, width in widths.items()
            ),
            baseline_status=status,
            baseline_reason=reason,
            baseline_diagnostics=baseline_diagnostics,
        )
        return replace(
            fit,
            diagnostics={
                **fit.diagnostics,
                _RESERVED: evidence.model_dump(mode="json"),
            },
        )


def _unit_provenance(
    parent: Provenance, contract: ArtifactReference, index: int
) -> Provenance:
    design = ArtifactReference.from_bytes(
        f"{contract.digest};unit_index={index}".encode(),
        kind="masked-sbc-unit-design-v1",
    )
    return Provenance.model_validate(parent.model_copy(update={"design": design}))


def _unit_mask(mask: NDArray[np.bool_], index: int) -> NDArray[np.bool_]:
    result = mask.copy()
    result[index] = False
    return result


def _compatible_baseline(reporting: Provenance, baseline: Provenance) -> None:
    if (
        reporting.prior != baseline.prior
        or reporting.design != baseline.design
        or reporting.simulator != baseline.simulator
        or reporting.target.intended != baseline.target.intended
    ):
        msg = "Baseline must share prior, design, simulator and intended target"
        raise ValueError(msg)


def run_masked_sbc(  # noqa: PLR0913 - explicit estimators, fixed bank and provenance
    prior_sampler: Callable[[RandomKey], Mapping[str, ArrayLike]],
    simulator: Callable[[Mapping[str, ArrayLike], RandomKey], ArrayLike],
    estimator: Callable[[ArrayLike, NDArray[np.bool_], RandomKey], SBCFit],
    *,
    plan: MaskedSBCPlan,
    estimands: Mapping[str, SBCEstimand],
    observed: ArrayLike,
    directory: str | Path,
    provenance: Provenance,
    replicate_ids: Sequence[int] | None = None,
    observed_mask: ArrayLike | None = None,
    baseline: Callable[[ArrayLike, NDArray[np.bool_], RandomKey], SBCFit] | None = None,
    baseline_provenance: Provenance | None = None,
    n_jobs: int = 1,
) -> MaskedSBCExecution:
    """Run a fixed masked-unit bank or disjoint shard through ``run_sbc``.

    Simulate the full cohort
    using the reporting prior/acceptance rule. The estimator receives the full
    original-shaped data, an explicit observed-cell mask and its own RandomKey.
    It must honor that mask; no missing-value encoding or row deletion is inferred.

    The baseline has a separate provenance and independent estimator stream.
    Baseline failures remain unavailable comparisons without deleting the primary
    fit's calibration result. MCMC validation/thinning follows the existing runner.

    Returns:
        Requested shard counts plus the complete prespecified bank's contract.
        Retain that reference independently for :func:`summarize_masked_sbc`.

    Raises:
        ValueError: Plan, masks, baseline context or existing artifacts disagree.
        ImportError: The optional study extra is unavailable.
    """
    plan = MaskedSBCPlan.model_validate(plan)
    provenance = Provenance.model_validate(provenance)
    if (baseline is None) != (baseline_provenance is None):
        msg = "A baseline requires both its callback and complete provenance"
        raise ValueError(msg)
    if baseline_provenance is not None:
        baseline_provenance = Provenance.model_validate(baseline_provenance)
        _compatible_baseline(provenance, baseline_provenance)
    values = np.asarray(observed)
    mask = (
        np.isfinite(values)
        if observed_mask is None
        else np.asarray(observed_mask, dtype=bool)
    )
    if (
        values.ndim < 1
        or values.shape[0] != len(plan.units)
        or mask.shape != values.shape
        or not np.all(np.isfinite(values[mask]))
        or np.count_nonzero(mask.reshape(len(plan.units), -1).any(axis=1)) < 2
    ):
        msg = "Masked SBC requires aligned first-axis units and two observed units"
        raise ValueError(msg)
    eligible = tuple(
        i
        for i, unit in enumerate(plan.units)
        if np.any(mask[i]) and (plan.stratum is None or unit.stratum == plan.stratum)
    )
    if not eligible:
        msg = "The declared stratum has no observed eligible units"
        raise ValueError(msg)
    requested = plan.replicate_ids if replicate_ids is None else tuple(replicate_ids)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(type(i) is not int or i not in plan.replicate_ids for i in requested)
    ):
        msg = "A shard must contain distinct IDs from the prespecified bank"
        raise ValueError(msg)
    if not estimands or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        or isinstance(e.prior_sd, (bool, np.bool_))
        or not np.isfinite(e.prior_sd)
        or e.prior_sd <= 0
        for name, e in estimands.items()
    ):
        msg = "Masked SBC requires estimands with finite positive prior SDs"
        raise ValueError(msg)
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in ("trade-study", "numpy", "scipy", "op_infer")
        }
    except importlib.metadata.PackageNotFoundError as error:
        msg = "Masked SBC requires op_infer[study] in an installed environment"
        raise ImportError(msg) from error
    versions["python"] = platform.python_version()
    payload = {
        "schema_version": 1,
        "plan": plan.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
        "baseline_provenance": None
        if baseline_provenance is None
        else baseline_provenance.model_dump(mode="json"),
        "estimands": {
            name: {"atoms": e.atoms, "prior_sd": e.prior_sd}
            for name, e in sorted(estimands.items())
        },
        "observed": _digest_array(values),
        "mask": _digest_array(mask),
        "eligible": eligible,
        "unit_masks": {str(i): _digest_array(_unit_mask(mask, i)) for i in eligible},
        "implementation": {
            name: ArtifactReference.from_bytes(
                Path(__file__).with_name(name).read_bytes(), kind="python-source-v1"
            ).model_dump(mode="json")
            for name in ("masked_sbc.py", "sbc.py", "calibrate.py")
        },
        "backend_versions": versions,
    }
    root = Path(directory)
    contract = ArtifactReference(
        kind="masked-sbc-contract-v1",
        digest=_require_manifest(root, payload, filename="masked-sbc-manifest.meta"),
    )
    written = resumed = failed = 0
    for index in eligible:
        ids = tuple(i for i in requested if eligible[i % len(eligible)] == index)
        if not ids:
            continue
        unit_mask = _unit_mask(mask, index)
        result = run_sbc(
            prior_sampler,
            simulator,
            _MaskedEstimator(estimator, baseline, estimands, unit_mask, index),
            estimands=estimands,
            observed=values,
            mask=unit_mask,
            directory=root / f"unit_{index:06d}",
            replicate_ids=ids,
            seed=plan.seed,
            provenance=_unit_provenance(provenance, contract, index),
            n_jobs=n_jobs,
        )
        written += result.written
        resumed += result.resumed
        failed += result.failed
    return MaskedSBCExecution(contract, len(requested), written, resumed, failed)


def _group(
    rows: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    bins: int,
    power_floor: int,
    alpha: float,
) -> dict[str, Any]:
    missing = sum(row["status"] == "missing" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    good = [row for row in rows if row["status"] == "successful"]
    table = {}
    for name in names:
        values = [
            next(c for c in row["informativeness"]["comparisons"] if c["name"] == name)
            for row in good
        ]
        prior = [v["posterior_sd"] / v["prior_sd"] for v in values]
        pairs = [v for v in values if v["baseline_sd"] is not None]
        raw_ratios = [
            v["posterior_sd"] / v["baseline_sd"] for v in pairs if v["baseline_sd"] > 0
        ]
        ratios = [value for value in raw_ratios if math.isfinite(value)]
        table[name] = {
            "n_planned": len(rows),
            "n_available": len(values),
            "n_unavailable": len(rows) - len(values),
            "prior_sd_ratio": _summary(prior),
            "n_baseline_available": len(pairs),
            "n_baseline_unavailable": len(rows) - len(pairs),
            "baseline_sd_ratio": _summary(ratios),
            "n_baseline_ratio_available": len(ratios),
            "n_baseline_zero_sd": sum(v["baseline_sd"] == 0 for v in pairs),
            "n_baseline_ratio_nonfinite": len(raw_ratios) - len(ratios),
            "sd_difference": _summary(
                [v["posterior_sd"] - v["baseline_sd"] for v in pairs]
            ),
        }
    ranks = None
    if rows and not missing:
        pits = (
            {name: [row["record"][name]["pit"] for row in good] for name in names}
            if good
            else {}
        )
        contraction = {
            name: [row["record"][name]["contraction"] for row in good] for name in names
        }
        ranks = aggregate(
            pits,
            contraction,
            n_failed=failed,
            bins=bins,
            power_floor=power_floor,
            alpha=alpha,
        )
    return {
        "n_planned": len(rows),
        "n_successful": len(good),
        "n_failed": failed,
        "n_missing": missing,
        "status": "no_planned_replicates"
        if not rows
        else ("incomplete" if missing else "complete"),
        "calibration_available": bool(good) and not missing,
        "rank_summary": ranks,
        "informativeness": table,
    }


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    if any(not math.isfinite(value) for value in values):
        msg = "Informativeness ratios/differences must remain finite"
        raise ValueError(msg)
    return {
        "mean": math.fsum(value / len(values) for value in values),
        "min": min(values),
        "max": max(values),
    }


def _read_evidence(
    record: Mapping[str, Any], index: int, payload: Mapping[str, Any]
) -> _Evidence:
    names = tuple(payload["estimands"])
    evidence = _Evidence.model_validate_json(json.dumps(record["_fit_diag"][_RESERVED]))
    if (
        evidence.unit_index != index
        or tuple(c.name for c in evidence.comparisons) != names
        or any(
            c.prior_sd != payload["estimands"][c.name]["prior_sd"]
            for c in evidence.comparisons
        )
    ):
        msg = "Masked width metadata differs from the declared estimands"
        raise ValueError(msg)
    if (payload["baseline_provenance"] is None) != (
        evidence.baseline_status == "not_requested"
    ):
        msg = "Masked baseline evidence differs from the contract"
        raise ValueError(msg)
    if evidence.reporting_draws_used != record["_fit_diag"]["draws_used"]:
        msg = "Masked retained draw counts disagree"
        raise ValueError(msg)
    if {name for name in record if not name.startswith("_")} != set(names):
        msg = "Masked replicate estimands differ from the contract"
        raise ValueError(msg)
    for comparison in evidence.comparisons:
        rank = _Rank.model_validate_json(json.dumps(record[comparison.name]))
        if not math.isclose(
            rank.contraction,
            comparison.posterior_sd / comparison.prior_sd,
            rel_tol=1e-12,
        ):
            msg = "Masked contraction disagrees with retained posterior width"
            raise ValueError(msg)
    return evidence


def summarize_masked_sbc(
    directory: str | Path,
    *,
    expected: ArtifactReference,
    bins: int = 20,
    power_floor: int = POWER_FLOOR,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Read the complete planned bank, retaining missing and failed comparisons.

    Each complete group reuses existing rank aggregation. BH adjustment is within each
    group; overlapping overall/stratum summaries are not a qualified joint gate.
    Raw width ratios/differences have no implicit informativeness threshold.

    Returns:
        Overall/stratum planned denominators, guarded records and width summaries.
        Incomplete groups have no rank reading; zero-plan strata remain visible.
        Baseline-zero SDs retain differences and unavailable ratio denominators.

    Raises:
        ValueError: Expected contract, group manifest, record or metadata differs.
        OSError: Required contract metadata cannot be read.
    """
    bins, power_floor, alpha = _rank_settings(bins, power_floor, alpha)
    root = Path(directory)
    encoded = (root / "masked-sbc-manifest.meta").read_bytes()
    actual = ArtifactReference.from_bytes(encoded, kind="masked-sbc-contract-v1")
    if actual != expected:
        msg = "Masked SBC contract differs from the independent expected reference"
        raise ValueError(msg)
    payload = json.loads(encoded)
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        msg = "Unsupported masked SBC schema"
        raise ValueError(msg)
    plan = MaskedSBCPlan.model_validate_json(json.dumps(payload["plan"]))
    provenance = Provenance.model_validate_json(json.dumps(payload["provenance"]))
    eligible = tuple(payload["eligible"])
    names = tuple(payload["estimands"])
    rows = []
    checked = {}
    for identity in plan.replicate_ids:
        index = eligible[identity % len(eligible)]
        unit = plan.units[index]
        folder = root / f"unit_{index:06d}"
        path = folder / f"rep_{identity:020d}.json"
        record = None
        evidence = None
        if not path.exists():
            status, reason = "missing", "replicate_not_written"
        else:
            if index not in checked:
                manifest = (folder / "sbc-manifest.meta").read_bytes()
                required = {
                    "schema_version": 1,
                    "provenance": _unit_provenance(
                        provenance, expected, index
                    ).model_dump(mode="json"),
                    "seed": plan.seed,
                    "estimands": payload["estimands"],
                    "observed": payload["observed"],
                    "mask": payload["unit_masks"][str(index)],
                }
                if json.loads(manifest) != required:
                    msg = "Masked unit manifest differs from the planned contract"
                    raise ValueError(msg)
                checked[index] = "sha256:" + hashlib.sha256(manifest).hexdigest()
            record = json.loads(path.read_text())
            if record.get("_meta") != {
                "runner_schema": 1,
                "contract": checked[index],
                "replicate": identity,
            }:
                msg = "Masked replicate identity or contract differs"
                raise ValueError(msg)
            if record.get("_failed") is True:
                status, reason = "failed", record["_reason"]
            else:
                evidence = _read_evidence(record, index, payload)
                status, reason = "successful", None
        rows.append(
            {
                "replicate_id": identity,
                "unit": unit.name,
                "stratum": unit.stratum,
                "status": status,
                "reason": reason,
                "informativeness": None
                if evidence is None
                else evidence.model_dump(mode="json"),
                "record": record,
            }
        )

    def summarize(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _group(selected, names, bins=bins, power_floor=power_floor, alpha=alpha)

    return {
        "schema_version": 1,
        "contract": expected.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "overall": summarize(rows),
        "baseline_provenance": payload["baseline_provenance"],
        "implementation": payload["implementation"],
        "backend_versions": payload["backend_versions"],
        "rank_settings": {"bins": bins, "power_floor": power_floor, "alpha": alpha},
        "report_implementation": {
            name: ArtifactReference.from_bytes(
                Path(__file__).with_name(name).read_bytes(), kind="python-source-v1"
            ).model_dump(mode="json")
            for name in ("masked_sbc.py", "calibrate.py")
        },
        "report_backend_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "op_infer")
        }
        | {"python": platform.python_version()},
        "by_stratum": {
            stratum: summarize([r for r in rows if r["stratum"] == stratum])
            for stratum in dict.fromkeys(u.stratum for u in plan.units)
        },
        "replicates": [{k: v for k, v in row.items() if k != "record"} for row in rows],
        "multiplicity": "within-group BH; overlapping groups need joint qualification",
        "adequacy_status": "not_assessed",
    }
