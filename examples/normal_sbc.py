"""Run and resume exact normal-normal SBC through the native study runner.

From a checkout: ``uv run --extra study python examples/normal_sbc.py OUTPUT``.
The default 64-replicate demonstration is underpowered for calibration claims.
The closed-form normal posterior is an analytic fixture, not a new sampler.
"""

from __future__ import annotations

import argparse
import importlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from op_infer.calibrate import aggregate, gate, load_replicates
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.sbc import SBCEstimand, SBCExecution, SBCFit, run_sbc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike

    from op_infer.rng import RandomKey


def prior(key: RandomKey) -> Mapping[str, ArrayLike]:
    """Draw one prior parameter using the parameter stream."""
    return {"theta": key.numpy_generator().normal()}


def simulate(parameters: Mapping[str, ArrayLike], key: RandomKey) -> ArrayLike:
    """Generate four independent unit-variance observations on a separate stream."""
    return float(np.asarray(parameters["theta"])) + key.numpy_generator().normal(size=4)


def fit(data: ArrayLike, key: RandomKey) -> SBCFit:
    """Draw from the exact posterior N(sum(data)/(n+1), 1/(n+1))."""
    values = np.asarray(data)
    variance = 1.0 / (values.size + 1)
    draws = key.numpy_generator().normal(
        values.sum() * variance, np.sqrt(variance), 1024
    )
    return SBCFit(
        {"theta": draws}, {"method": "analytic-normal-normal", "variance": variance}
    )


def provenance() -> Provenance:
    """Identify the analytic fixture, callbacks, dependencies and estimator policy.

    Returns:
        Complete metadata without claiming general statistical qualification.

    Raises:
        RuntimeError: A source module cannot be identified.
    """
    source = ArtifactReference.from_bytes(
        Path(__file__).read_bytes(), kind="normal-sbc-source-v1"
    )
    sources = {"example": source.digest}
    for name in ("op_infer.sbc", "op_infer.calibrate", "op_infer.rng"):
        location = importlib.import_module(name).__file__
        if location is None:
            msg = f"Cannot identify source for {name}"
            raise RuntimeError(msg)
        sources[name] = ArtifactReference.from_bytes(
            Path(location).read_bytes(), kind="python-source-v1"
        ).digest
    target = ArtifactReference.from_bytes(
        b"theta~N(0,1); y_i|theta~N(theta,1); n=4", kind="analytic-target-v1"
    )
    return Provenance(
        schema_version=1,
        prior=ArtifactReference.from_bytes(b"theta~N(0,1)", kind="prior-v1"),
        design=ArtifactReference.from_bytes(
            b"4 independent observations; real data=[0.2,-0.1,0.3,0.4]",
            kind="design-v1",
        ),
        simulator=source,
        target=LikelihoodTargetReference(
            kind="analytic",
            intended=target,
            evaluated=target,
            qualification=Qualification(
                status="unqualified",
                scope=target,
                reason="Analytic fixture; general calibration is unqualified",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="reporting_sbc",
        estimator_settings=ArtifactReference.from_bytes(
            b"exact normal posterior; 1024 independent draws; RandomKey v1",
            kind="estimator-v1",
        ),
        backend_versions=tuple(
            BackendVersion(name=name, version=version(name))
            for name in ("numpy", "op_infer", "trade-study")
        ),
        commit=ArtifactReference.from_bytes(
            json.dumps(sources, sort_keys=True).encode(), kind="source-manifest-v1"
        ),
    )


def run_example(
    directory: Path, *, replicate_ids: Sequence[int] = tuple(range(64))
) -> dict[str, Any]:
    """Execute the fixed bank, resume it and return its calibration ledger."""
    identity = provenance()

    def run() -> SBCExecution:
        """Use the same immutable contract for execution and resume."""
        return run_sbc(
            prior,
            simulate,
            fit,
            estimands={"theta": SBCEstimand(lambda p: p["theta"], prior_sd=1.0)},
            observed=np.array([0.2, -0.1, 0.3, 0.4]),
            directory=directory,
            replicate_ids=replicate_ids,
            seed=2026,
            provenance=identity,
        )

    first = run()
    resumed = run()
    pits, contraction, diagnostics = load_replicates(directory)
    report = aggregate(pits, contraction, n_failed=diagnostics["n_failed"])
    verdict = gate(report, max_flagged=0)
    return {
        "requested": first.requested,
        "failed": first.failed,
        "resumed": resumed.resumed,
        "underpowered": report["underpowered"],
        "label": verdict["label"],
        "n_total": report["n_total"],
    }


def main() -> None:
    """Write guarded replicate artifacts and print a bounded demonstration ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_example(args.directory), indent=2))  # noqa: T201 -- executable example output


if __name__ == "__main__":
    main()
