"""Generate fixed-coverage planning tables and reference TARP curves.

Run with the tarp extra. Rendering additionally needs Matplotlib; --check verifies
committed rounded plotted inputs, source identity and exports without Matplotlib.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from illustration_artifacts import IllustrationData, build_artifacts

from op_infer.adapters.tarp import tarp_coverage
from op_infer.coverage import (
    CoverageRule,
    coverage_operating_characteristics,
    plan_coverage_replicates,
)
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.rng import RandomKey


def _provenance(key: RandomKey, source: ArtifactReference, offset: float) -> Provenance:
    prior = ArtifactReference.from_bytes(b"theta~Normal(0,I_2)", kind="prior-v1")
    design = ArtifactReference.from_bytes(
        b"one independent unit-variance observation per coordinate", kind="design-v1"
    )
    target = ArtifactReference.from_bytes(
        b"theta|y~Normal(y/2,I_2/2)", kind="analytic-normal-target-v1"
    )
    return Provenance(
        schema_version=1,
        prior=prior,
        design=design,
        simulator=source,
        target=LikelihoodTargetReference(
            kind="analytic",
            intended=target,
            evaluated=target,
            qualification=Qualification(
                status="unqualified",
                scope=target,
                reason="Analytic fixture; no gate qualification",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="analytic_coverage_example",
        estimator_settings=ArtifactReference.from_bytes(
            json.dumps({"key": key.model_dump(), "mean_offset": offset}).encode(),
            kind="analytic-draw-settings-v1",
        ),
        backend_versions=tuple(
            BackendVersion(name=name, version=importlib.metadata.version(name))
            for name in ("numpy", "op_infer", "scipy", "tarp")
        ),
        commit=source,
    )


def run_example() -> dict[str, Any]:
    """Compute a hypothetical coverage family and analytic posterior TARP curves.

    Returns:
        Complete plotted inputs rounded to 12 decimal places for portable figure
        checks. Full-precision diagnostic APIs are unchanged. The numeric criteria
        and budgets here are illustration settings, not package defaults.
    """
    rules = tuple(
        CoverageRule(name=f"theta_{i}_50", nominal=0.5, lower=0.45, upper=0.55)
        for i in range(20)
    )
    plan = plan_coverage_replicates(rules, max_false_fail=0.01, max_replicates=4000)
    chosen = plan["n_replicates"]
    counts = list(range(100, 1601, 10))
    reports = [
        coverage_operating_characteristics(rules, n_replicates=n) for n in counts
    ]
    table = []
    for count in (300, 1000, chosen):
        record = coverage_operating_characteristics(rules, n_replicates=count)
        bounds = record["family_rejection_probability"]
        table.append(
            {
                "n": count,
                "lower": round(bounds["lower"], 12),
                "upper": round(bounds["upper"], 12),
            }
        )
    key = RandomKey(seed=2026, stream="coverage_example")
    rng = key.numpy_generator()
    truth = rng.normal(size=(600, 2))
    observed = truth + rng.normal(size=truth.shape)
    samples = rng.normal(observed / 2, np.sqrt(0.5), size=(256, *truth.shape))
    references = observed + rng.normal(scale=2, size=truth.shape)
    source = ArtifactReference.from_bytes(
        Path(__file__).read_bytes(), kind="coverage-example-source-v1"
    )
    curves = {}
    for name, offset in (("Analytic posterior", 0.0), ("Mean shifted by 1.5", 1.5)):
        report = tarp_coverage(
            samples + offset,
            truth,
            references=references,
            replicate_ids=tuple(range(600)),
            quantities=("theta_1", "theta_2"),
            definition=source,
            provenance=_provenance(key, source, offset),
            num_bins=20,
        )
        if report["curve"] is None:
            msg = f"Illustration reference calculation unavailable: {report['reason']}"
            raise RuntimeError(msg)
        curve = report["curve"]
        curves[name] = {
            "expected_coverage": np.round(curve["expected_coverage"], 12).tolist(),
            "credibility": np.round(curve["credibility"], 12).tolist(),
            "max_abs_deviation": round(curve["max_abs_deviation"], 12),
            "n_total": report["n_total"],
            "n_usable": report["n_usable"],
            "n_failed": report["n_failed"],
            "inputs": report["inputs"],
        }
    return {
        "display_rounding_decimals": 12,
        "selected_replicates": chosen,
        "coverage_table": table,
        "planning": {
            "n": counts,
            "lower": [
                round(r["family_rejection_probability"]["lower"], 12) for r in reports
            ],
            "upper": [
                round(r["family_rejection_probability"]["upper"], 12) for r in reports
            ],
            "max_false_fail": 0.01,
            "rules": [rule.model_dump(mode="json") for rule in rules],
        },
        "tarp": curves,
        "adequacy_status": "not_assessed",
    }


def render_table(result: dict[str, Any]) -> str:
    """Format the computed bounds and curve summaries without pass labels.

    Returns:
        Markdown tables of the hypothetical family and the two reference curves.
    """
    rows = [
        "| Usable replicates | Family rejection lower bound | Upper bound |",
        "| ---: | ---: | ---: |",
    ]
    rows.extend(
        f"| {r['n']} | {r['lower']:.6f} | {r['upper']:.6f} |"
        for r in result["coverage_table"]
    )
    rows += [
        "",
        "| TARP fixture | Usable / planned rows | Failed rows | Max deviation |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows.extend(
        f"| {name} | {r['n_usable']} / {r['n_total']} | {r['n_failed']} | "
        f"{r['max_abs_deviation']:.4f} |"
        for name, r in result["tarp"].items()
    )
    return "\n".join(rows) + "\n"


def _render(output: Path, data: dict[str, Any]) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update(
        {"svg.hashsalt": "op-infer-coverage", "svg.fonttype": "none", "font.size": 10}
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    plan = data["planning"]
    axes[0].fill_between(
        plan["n"],
        plan["lower"],
        plan["upper"],
        color="#d4e2e6",
        label="Bounds with unrestricted dependence",
    )
    axes[0].plot(plan["n"], plan["lower"], color="#0d6b6e")
    axes[0].plot(plan["n"], plan["upper"], color="#24678e")
    axes[0].axhline(
        0.01, color="#b15b26", linestyle="--", label="Illustration target: 1%"
    )
    axes[0].axvline(
        data["selected_replicates"],
        color="#666666",
        linestyle=":",
        label=f"First qualifying size: {data['selected_replicates']}",
    )
    axes[0].set(
        xlabel="Usable replicates in a fixed bank",
        ylabel="False-failure probability",
        yscale="log",
        title="20 criteria: inclusive 45% to 55% coverage",
    )
    axes[1].plot(
        [0, 1], [0, 1], color="#777777", linestyle="--", label="Uniform distance ranks"
    )
    for (name, curve), color in zip(
        data["tarp"].items(), ("#0d6b6e", "#b15b26"), strict=True
    ):
        axes[1].plot(
            curve["credibility"],
            curve["expected_coverage"],
            marker=".",
            color=color,
            label=name,
        )
    axes[1].set(
        xlabel="Credibility level",
        ylabel="Expected coverage",
        title="TARP: two analytic normal examples",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="best")
    figure.savefig(output / "figure.svg", metadata={"Date": None})
    figure.savefig(
        output / "figure.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)


def main() -> None:
    """Generate or check retained sources, rounded data, tables and vector exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/coverage",
    )
    args = parser.parse_args()
    data = run_example()
    root = Path(__file__).resolve().parents[1]
    build_artifacts(
        args.output,
        IllustrationData(render_table(data), data),
        [
            Path(__file__),
            *(
                root / "src/op_infer" / name
                for name in (
                    "coverage.py",
                    "adapters/tarp.py",
                    "rng.py",
                    "posterior.py",
                )
            ),
        ],
        lambda output: _render(output, data),
        check=args.check,
    )
    if not args.check:
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["reference_versions"] = {
            name: importlib.metadata.version(name)
            for name in ("scipy", "tarp", "op_infer")
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(render_table(data), end="")  # noqa: T201 - documented example output


if __name__ == "__main__":
    main()
