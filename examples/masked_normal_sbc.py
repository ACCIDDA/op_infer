"""Run a small exact-normal masked-cohort bank and print stratum/width counts.

Requires the study extra. The 64-replicate example is deliberately underpowered;
it demonstrates masking, baseline comparison, guarded resume and denominators.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from op_infer.masked_sbc import (
    MaskedSBCPlan,
    MaskedUnit,
    run_masked_sbc,
    summarize_masked_sbc,
)
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.sbc import SBCEstimand, SBCFit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from op_infer.masked_sbc import MaskedSBCExecution
    from op_infer.rng import RandomKey

_SIGMA = np.array([[1.0], [1.0], [2.0], [2.0]])


def prior(key: RandomKey) -> Mapping[str, ArrayLike]:
    """Draw one scalar from the reporting prior."""
    return {"theta": key.numpy_generator().normal()}


def simulate(truth: Mapping[str, ArrayLike], key: RandomKey) -> ArrayLike:
    """Simulate all four units before any unit is removed."""
    return np.asarray(
        float(np.asarray(truth["theta"]))
        + key.numpy_generator().normal(size=(4, 1)) * _SIGMA,
        dtype=np.float64,
    )


def fit(data: ArrayLike, mask: NDArray[np.bool_], key: RandomKey) -> SBCFit:
    """Draw the exact conditional normal posterior using only unmasked cells."""
    variance = 1 / (1 + np.sum(mask / _SIGMA**2))
    mean = variance * np.sum(np.where(mask, np.asarray(data) / _SIGMA**2, 0))
    return SBCFit(
        {"theta": key.numpy_generator().normal(mean, np.sqrt(variance), 512)},
        {"method": "analytic_normal", "key": key.model_dump(mode="json")},
    )


def _context() -> Provenance:
    source = ArtifactReference.from_bytes(
        Path(__file__).read_bytes(), kind="masked-normal-example-source-v1"
    )
    target = ArtifactReference.from_bytes(
        b"theta~N(0,1); y_i|theta~N(theta,sigma_i^2); "
        b"sigma=(1,1,2,2); apply declared cell mask",
        kind="masked-normal-target-v1",
    )
    return Provenance(
        schema_version=1,
        prior=ArtifactReference.from_bytes(b"theta~N(0,1)", kind="prior-v1"),
        design=ArtifactReference.from_bytes(
            b"four ordered units, one observation each; sigma=(1,1,2,2)",
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
                reason="Analytic illustration; no consumer gate qualification",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="masked_example",
        estimator_settings=ArtifactReference.from_bytes(
            b"exact normal; 512 draws; explicit unit mask; no filtering",
            kind="analytic-estimator-v1",
        ),
        backend_versions=tuple(
            BackendVersion(name=name, version=importlib.metadata.version(name))
            for name in ("numpy", "op_infer", "scipy", "trade-study")
        ),
        commit=source,
    )


def run_example(directory: Path) -> dict[str, Any]:
    """Run/resume the complete small bank; retain full per-replicate artifacts.

    Returns:
        A guarded summary with separate stratum and width denominators. It is
        underpowered and has no combined calibration or informativeness verdict.
    """
    plan = MaskedSBCPlan(
        units=tuple(
            MaskedUnit(name=f"u{i}", stratum="precise" if i < 2 else "noisy")
            for i in range(4)
        ),
        replicate_ids=tuple(range(64)),
        seed=2026,
    )
    context = _context()
    baseline_context = context.model_copy(
        update={"estimator_stage": "independent_analytic_baseline"}
    )
    estimands = {
        "theta": SBCEstimand(lambda p: p["theta"], prior_sd=1.0),
        "positive": SBCEstimand(
            lambda p: np.maximum(p["theta"], 0),
            prior_sd=float(np.sqrt(0.5 - 1 / (2 * np.pi))),
            atoms=True,
        ),
    }

    def execute() -> MaskedSBCExecution:
        return run_masked_sbc(
            prior,
            simulate,
            fit,
            plan=plan,
            estimands=estimands,
            observed=np.array([[0.2], [-0.1], [0.3], [0.4]]),
            directory=directory,
            provenance=context,
            baseline=fit,
            baseline_provenance=baseline_context,
        )

    first = execute()
    resumed = execute()
    if resumed.written or resumed.resumed != 64 or first.contract != resumed.contract:
        msg = "Masked bank did not resume the complete declared plan"
        raise RuntimeError(msg)
    result = summarize_masked_sbc(directory, expected=first.contract)
    result["resumed"] = resumed.resumed
    return result


def render_table(summary: dict[str, Any]) -> str:
    """Print all tested groups and unavailable comparison counts.

    Returns:
        Markdown table of raw widths and planned denominators, without pass labels.
    """
    lines = [
        (
            "| Removed stratum | Quantity | Successful / planned | Failed / missing | "
            "Baseline ratios | "
            "Mean SD / prior SD | Mean SD / baseline SD |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, group in {
        "overall": summary["overall"],
        **summary["by_stratum"],
    }.items():
        for name, width in group["informativeness"].items():
            prior_ratio = width["prior_sd_ratio"]
            baseline_ratio = width["baseline_sd_ratio"]
            a = "unavailable" if prior_ratio is None else f"{prior_ratio['mean']:.4f}"
            b = (
                "unavailable"
                if baseline_ratio is None
                else f"{baseline_ratio['mean']:.4f}"
            )
            lines.append(
                f"| {group_name} | {name} | {group['n_successful']} / "
                f"{group['n_planned']} | {group['n_failed']} / {group['n_missing']} | "
                f"{width['n_baseline_ratio_available']} | "
                f"{a} | {b} |"
            )
    return "\n".join(lines)


def main() -> None:
    """Write/resume the example in the supplied directory and print its table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = run_example(args.directory)
    print(render_table(summary))  # noqa: T201 - executable example output
    print(  # noqa: T201 - make its underpowered status visible
        f"\nResumed {summary['resumed']} records; 64 replicates, underpowered. "
        "Adequacy not assessed."
    )


if __name__ == "__main__":
    main()
