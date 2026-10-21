"""Tabulate named diagnostics on analytic draws and deliberate failure fixtures.

This example does not fit a sampler or qualify an adequacy gate. Its shifted,
constant and nonfinite quantities demonstrate how every selected row is retained.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from op_infer.adapters.arviz import DiagnosticValue, MCMCDiagnostics, diagnose_mcmc
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    Coordinate,
    LikelihoodTargetReference,
    Posterior,
    Provenance,
    Qualification,
    SampledVariable,
)
from op_infer.rng import RandomKey


def run_example() -> MCMCDiagnostics:
    """Draw the analytic Normal posterior and report all five named elements.

    Returns:
        Raw diagnostics with two deliberately incomplete elements, not a gate pass.
    """
    key = RandomKey(seed=2026, stream="diagnostic_example")
    theta = key.numpy_generator().normal(1.0, np.sqrt(0.2), size=(4, 256))
    shifted = theta.copy()
    shifted[0] += 2
    invalid = theta.copy()
    invalid[0, 0] = np.nan
    source = ArtifactReference.from_bytes(
        Path(__file__).read_bytes(), kind="diagnostics-example-source-v1"
    )
    target = ArtifactReference.from_bytes(
        b"theta~Normal(0,1); y=(0.5,1.5,1,2); y_i|theta~Normal(theta,1)",
        kind="analytic-normal-target-v1",
    )
    provenance = Provenance(
        schema_version=1,
        prior=ArtifactReference.from_bytes(b"theta~Normal(0,1)", kind="prior-v1"),
        design=ArtifactReference.from_bytes(b"y=(0.5,1.5,1,2)", kind="data-v1"),
        simulator=source,
        target=LikelihoodTargetReference(
            kind="analytic",
            intended=target,
            evaluated=target,
            qualification=Qualification(
                status="unqualified",
                scope=target,
                reason="Analytic demonstration with deliberately invalid quantities",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="analytic_diagnostic_example",
        estimator_settings=ArtifactReference.from_bytes(
            key.model_dump_json().encode() + b"; shape=(4,256); Normal(1,sqrt(0.2))",
            kind="analytic-draw-settings-v1",
        ),
        backend_versions=tuple(
            BackendVersion(name=name, version=importlib.metadata.version(name))
            for name in ("numpy", "op_infer")
        ),
        commit=source,
    )
    posterior = Posterior(
        draws=(
            SampledVariable("theta", theta),
            SampledVariable(
                "reported", np.stack((shifted, 2 * theta), axis=-1), ("component",)
            ),
            SampledVariable("constant", np.zeros_like(theta)),
            SampledVariable("invalid", invalid),
        ),
        coordinates=(
            Coordinate(name="chain", labels=(4, 1, 9, 2)),
            Coordinate(name="draw", labels=tuple(range(256))),
            Coordinate(name="component", labels=("shifted", "scaled")),
        ),
        provenance=provenance,
    )
    return diagnose_mcmc(
        posterior,
        quantities=("theta", "reported", "constant", "invalid"),
        definition=source,
    )


def _cell(value: DiagnosticValue) -> str:
    if value.value is None:
        return f"{value.status} ({value.reason})"
    return f"{value.value:.3f}"


def render_table(report: MCMCDiagnostics) -> str:
    """Render original named event labels and every result as a Markdown table.

    Returns:
        A table of raw values/reasons without thresholds or adequacy decisions.
    """
    coords = {coordinate.name: coordinate.labels for coordinate in report.coordinates}
    rows = [
        "| Quantity | Rank R-hat | Bulk ESS | Tail ESS |",
        "| --- | --- | --- | --- |",
    ]
    for quantity in report.quantities:
        for component in quantity.components:
            labels = ", ".join(
                f"{dim}={coords[dim][index]}"
                for dim, index in zip(quantity.dims, component.index, strict=True)
            )
            name = quantity.name + (f"[{labels}]" if labels else "")
            values = (
                component.rank_rhat,
                component.bulk_ess,
                component.tail_ess,
            )
            rows.append(f"| {name} | " + " | ".join(_cell(v) for v in values) + " |")
    return "\n".join(rows)


def main() -> None:
    """Print the table and its complete denominator without creating artifacts."""
    report = run_example()
    print(render_table(report))  # noqa: T201 -- documented example output
    print(  # noqa: T201 -- include failed rows in the displayed denominator
        f"\nSelected elements: {report.n_components}; "
        f"incomplete: {report.n_incomplete}; adequacy: {report.adequacy_status}."
    )


if __name__ == "__main__":
    main()
