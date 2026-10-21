"""Replayable SBC execution on trade-study, writing the existing rank schema.

Scheduling belongs to trade-study; fitting
belongs to the caller's reporting estimator. This host artifact boundary accepts
NumPy-compatible observations and scalar estimand draws, not traced model arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from op_infer.calibrate import (
    assert_simulated_differs,
    pit,
    posterior_contraction,
    write_failed_replicate,
    write_replicate,
)
from op_infer.posterior import Provenance
from op_infer.rng import RandomKey, SimulationStreams

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SBCEstimand:
    """A scalar truth/draw transformation with an explicitly declared atom rule.

    ``value`` accepts either parameter scalars or posterior arrays. The caller
    supplies a prior standard deviation for contraction on the same scale.

    Examples:
        >>> estimand = SBCEstimand(lambda values: values["theta"], prior_sd=1.0)
        >>> estimand.value({"theta": np.array([1.0, 2.0])}).tolist()
        [1.0, 2.0]
        >>> estimand.atoms
        False
    """

    value: Callable[[Mapping[str, ArrayLike]], ArrayLike]
    prior_sd: float
    atoms: bool = False


@dataclass(frozen=True)
class SBCFit:
    """Reporting-estimator draws and retained per-fit adequacy evidence.

    Scalar variable arrays share one draw dimension. MCMC fits require an explicit
    adequate verdict and finite ESS; the runner selects evenly spaced draws up to
    floor(ESS). This thinning is bookkeeping, not an adequacy diagnostic or proof
    of independent MCMC draws. Adapters own diagnostics and their qualification.
    """

    draws: Mapping[str, ArrayLike]
    diagnostics: Mapping[str, Any]
    kind: Literal["independent", "mcmc"] = "independent"
    effective_sample_size: float | None = None
    adequate: bool | None = None


@dataclass(frozen=True)
class SBCExecution:
    """Requested, written, resumed and failed replicate counts; no gate verdict."""

    requested: int
    written: int
    resumed: int
    failed: int


def _digest_array(value: ArrayLike) -> str:
    """Return a host-array content identity, including shape and dtype."""
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        msg = "SBC observed data and masks must be numeric arrays"
        raise ValueError(msg)
    header = str((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def _require_manifest(
    directory: Path,
    payload: Mapping[str, Any],
    *,
    filename: str = "sbc-manifest.meta",
) -> str:
    """Atomically establish one immutable run contract shared by disjoint shards.

    Returns:
        Complete run contract content identity.

    Raises:
        ValueError: An existing run has different provenance or settings.
    """
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    path = directory / filename
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as scratch:
        scratch.write(encoded)
        temporary = Path(scratch.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                msg = "Execution provenance/settings mismatch; use a new directory"
                raise ValueError(msg) from None
    finally:
        temporary.unlink()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _fit_draws(fit: SBCFit) -> dict[str, NDArray[np.generic]]:
    """Validate draws and apply the declared MCMC ESS thinning policy.

    Returns:
        Scalar-variable draws with common sample positions.

    Raises:
        ValueError: Layout or MCMC adequacy information is invalid.
    """
    draws = {name: np.asarray(value) for name, value in fit.draws.items()}
    sizes = {value.shape for value in draws.values()}
    if len(sizes) != 1 or not draws:
        msg = "SBCFit requires common scalar-variable draw shapes"
        raise ValueError(msg)
    shape = next(iter(sizes))
    if any(not np.all(np.isfinite(value)) for value in draws.values()):
        msg = "SBCFit draws must be finite; failed fits cannot be filtered"
        raise ValueError(msg)
    if len(shape) != 1 or shape[0] < 2:
        msg = "SBCFit requires at least two draws on one sample dimension"
        raise ValueError(msg)
    if fit.kind == "mcmc":
        ess = fit.effective_sample_size
        if fit.adequate is not True or ess is None or not np.isfinite(ess) or ess < 2:
            msg = "MCMC SBC requires explicit adequacy and finite ESS >= 2"
            raise ValueError(msg)
        indices = np.linspace(0, shape[0] - 1, min(shape[0], int(ess)), dtype=int)
        draws = {name: value[indices] for name, value in draws.items()}
    elif fit.kind != "independent":
        msg = "Unknown SBCFit sample kind"
        raise ValueError(msg)
    return draws


@dataclass
class _SBCWorld:
    prior_sampler: Callable[[RandomKey], Mapping[str, ArrayLike]]
    simulator: Callable[[Mapping[str, ArrayLike], RandomKey], ArrayLike]
    estimator: Callable[[ArrayLike, RandomKey], SBCFit]
    estimands: Mapping[str, SBCEstimand]
    observed: ArrayLike
    mask: ArrayLike | None
    directory: Path
    seed: int
    identity: str

    def generate(self, config: dict[str, Any]) -> tuple[dict[str, float], None]:
        """Run one independently addressed replicate, recording all failures.

        Returns:
            Failure/resume flags for trade-study's scheduler.

        Raises:
            ValueError: A resumed record does not match this run contract.
            RuntimeError: Another worker owns the same replicate ID.
        """
        identity = int(config["replicate"])
        path = self.directory / f"rep_{identity:020d}.json"
        meta = {"runner_schema": 1, "contract": self.identity, "replicate": identity}
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("_meta") != meta:
                msg = f"SBC replicate {identity} has mismatched provenance"
                raise ValueError(msg)
            return {
                "failed": float(record.get("_failed") is True),
                "resumed": 1.0,
            }, None
        lock = path.with_suffix(".lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            msg = f"SBC replicate {identity} already has an active/stale worker lock"
            raise RuntimeError(msg) from None
        os.close(descriptor)
        try:
            return self._execute(identity, path, meta), None
        finally:
            lock.unlink()

    def _execute(
        self, identity: int, path: Path, meta: Mapping[str, Any]
    ) -> dict[str, float]:
        """Keep simulation/fit failures as records rather than aborting a bank."""
        try:
            streams = SimulationStreams(seed=self.seed, index=identity)
            truth = self.prior_sampler(streams.parameters)
            simulated = self.simulator(truth, streams.simulation)
            equal = assert_simulated_differs(
                simulated,
                self.observed,
                np.ones_like(np.asarray(self.observed), dtype=bool)
                if self.mask is None
                else self.mask,
            )
            estimator_key = RandomKey(
                seed=self.seed, stream="estimator.reporting", index=identity
            )
            fit = self.estimator(simulated, estimator_key)
            draws = _fit_draws(fit)
            pits, contraction = {}, {}
            for name, estimand in self.estimands.items():
                true_value = float(np.asarray(estimand.value(truth)).item())
                samples = np.asarray(estimand.value(draws))
                if samples.ndim != 1:
                    msg = f"Estimand {name!r} must return one value per posterior draw"
                    raise ValueError(msg)  # noqa: TRY301 - preserve this replicate failure
                key = RandomKey(
                    seed=self.seed, stream=f"calibration.rank.{name}", index=identity
                )
                pits[name] = pit(
                    samples, true_value, atoms=estimand.atoms, rng=key.numpy_generator()
                )
                contraction[name] = posterior_contraction(samples, estimand.prior_sd)
            diagnostics = {
                **fit.diagnostics,
                "equal_observed_cells": equal,
                "sample_kind": fit.kind,
                "effective_sample_size": fit.effective_sample_size,
                "adequate": fit.adequate,
                "draws_used": len(next(iter(draws.values()))),
                "streams": streams.model_dump(mode="json"),
            }
            write_replicate(
                path, pits, contraction=contraction, fit_diag=diagnostics, metadata=meta
            )
        except Exception as error:  # noqa: BLE001 - each failed replicate remains in the denominator
            write_failed_replicate(
                path, f"{type(error).__name__}: {error}", metadata=meta
            )
            return {"failed": 1.0, "resumed": 0.0}
        return {"failed": 0.0, "resumed": 0.0}


class _StatusScorer:
    @staticmethod
    def score(
        truth: dict[str, float], observed: None, config: dict[str, Any]
    ) -> dict[str, float]:
        """Return stored completion flags; scientific calibration uses the files."""
        del observed, config
        return truth


def run_sbc(  # noqa: PLR0913 - explicit scientific callbacks and artifact contract
    prior_sampler: Callable[[RandomKey], Mapping[str, ArrayLike]],
    simulator: Callable[[Mapping[str, ArrayLike], RandomKey], ArrayLike],
    estimator: Callable[[ArrayLike, RandomKey], SBCFit],
    *,
    estimands: Mapping[str, SBCEstimand],
    observed: ArrayLike,
    directory: str | Path,
    replicate_ids: Sequence[int],
    seed: int,
    provenance: Provenance,
    mask: ArrayLike | None = None,
    n_jobs: int = 1,
) -> SBCExecution:
    """Schedule a prespecified SBC bank with guarded resume and counted failures.

    Use the reporting estimator and the same prior/acceptance rule as training.
    Disjoint ID shards share one directory; failures are complete artifacts too.
    Existing failure records are retained on resume. No retry, plausibility filter
    or result-dependent stopping changes the planned denominator. Call aggregate
    and gate after all shards complete; replicate sizing uses trade-study helpers.

    Returns:
        Completion accounting for exactly the requested replicate IDs.

    Raises:
        ImportError: The optional study extra is not installed.
        ValueError: The contract is invalid or existing artifacts disagree.
    """
    try:
        from trade_study import Direction, Observable, run_grid  # noqa: PLC0415 - optional extra
    except ImportError as error:
        msg = "SBC execution requires pip install 'op_infer[study]'"
        raise ImportError(msg) from error
    ids = tuple(replicate_ids)
    if not ids or len(set(ids)) != len(ids):
        msg = "Provide nonempty, distinct stable replicate IDs"
        raise ValueError(msg)
    for identity in ids:
        SimulationStreams(seed=seed, index=identity)
    if not estimands or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in estimands
    ):
        msg = "Estimands require nonempty portable names"
        raise ValueError(msg)
    if any(not np.isfinite(e.prior_sd) or e.prior_sd <= 0 for e in estimands.values()):
        msg = "Each estimand requires a finite positive prior_sd"
        raise ValueError(msg)
    observed_array = np.asarray(observed)
    mask = np.isfinite(observed_array) if mask is None else np.asarray(mask, dtype=bool)
    if (
        mask.shape != observed_array.shape
        or not np.any(mask)
        or not np.all(np.isfinite(observed_array[mask]))
    ):
        msg = "SBC observed data require an aligned nonempty finite-cell mask"
        raise ValueError(msg)
    provenance = Provenance.model_validate(provenance)
    contract = {
        "schema_version": 1,
        "provenance": provenance.model_dump(mode="json"),
        "seed": seed,
        "estimands": {
            name: {"atoms": e.atoms, "prior_sd": e.prior_sd}
            for name, e in sorted(estimands.items())
        },
        "observed": _digest_array(observed),
        "mask": None if mask is None else _digest_array(mask),
    }
    directory = Path(directory)
    contract_identity = _require_manifest(directory, contract)
    world = _SBCWorld(
        prior_sampler,
        simulator,
        estimator,
        dict(estimands),
        observed,
        mask,
        directory,
        seed,
        contract_identity,
    )
    result = run_grid(
        world,
        _StatusScorer(),
        [{"replicate": i} for i in ids],
        [
            Observable("failed", Direction.MINIMIZE),
            Observable("resumed", Direction.MINIMIZE),
        ],
        n_reps=1,
        n_jobs=n_jobs,
    )
    failures, resumed = np.sum(result.scores, axis=0).astype(int)
    return SBCExecution(len(ids), len(ids) - int(resumed), int(resumed), int(failures))
