"""Illustrate the exact-normal fixture and its deliberately underpowered SBC bank.

Regenerate with the study extra, Python 3.12 and Matplotlib:
uv run --extra study --with matplotlib==3.11.2 python examples/illustrate_normal_sbc.py.
Use --check with the study extra and no plotting dependency.
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from illustration_artifacts import IllustrationData, build_artifacts
from normal_sbc import fit, run_example

from op_infer.calibrate import load_replicates
from op_infer.rng import RandomKey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _render(
    output: Path, pits: NDArray[np.float64], draws: NDArray[np.float64]
) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"svg.hashsalt": "op-infer-normal-sbc", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    positions = np.linspace(-1.5, 1.8, 300)
    variance = 1.0 / 5.0
    mean = 0.8 / 5.0
    density = np.exp(-0.5 * (positions - mean) ** 2 / variance) / np.sqrt(
        2 * np.pi * variance
    )
    axes[0].hist(
        draws,
        bins=25,
        density=True,
        color="#0d6b6e",
        alpha=0.35,
        label="1,024 draws from analytic fixture",
    )
    axes[0].plot(positions, density, color="#24678e", label="Exact posterior density")
    axes[0].set(
        xlabel="Theta",
        ylabel="Density",
        title="One observed dataset: analytic reference",
    )
    axes[1].hist(pits, bins=20, range=(0.0, 1.0), color="#24678e", edgecolor="white")
    axes[1].axhline(
        pits.size / 20,
        color="#87641a",
        linestyle="--",
        label="Uniform expected bin count",
    )
    axes[1].set(
        xlabel="Randomized-rank PIT",
        ylabel="Replicate count",
        title=f"SBC bank: {pits.size} replicates; underpowered",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output / "figure.svg", metadata={"Date": None})
    figure.savefig(
        output / "figure.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/normal-sbc",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="normal-sbc-illustration-") as directory:
        ledger = run_example(Path(directory))
        values, _, _ = load_replicates(directory)
        pits = np.asarray(values["theta"], dtype=np.float64)
    observed = np.array([0.2, -0.1, 0.3, 0.4])
    draws = np.asarray(
        fit(observed, RandomKey(seed=2026, stream="illustration")).draws["theta"],
        dtype=np.float64,
    )
    lines = [
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Requested replicates | {ledger['requested']} |",
        f"| Failed replicates | {ledger['failed']} |",
        f"| Resumed replicates | {ledger['resumed']} |",
        f"| Total in ledger | {ledger['n_total']} |",
        f"| Underpowered | {ledger['underpowered']} |",
        f"| Gate label | {ledger['label']} |",
        f"| Exact posterior mean (observed data) | {observed.sum() / 5:.4f} |",
        f"| Exact posterior SD | {np.sqrt(1 / 5):.4f} |",
    ]
    build_artifacts(
        args.output,
        IllustrationData(
            "\n".join(lines) + "\n",
            {"pits": pits.tolist(), "posterior_draws": draws.tolist()},
        ),
        [Path(__file__), Path(__file__).with_name("normal_sbc.py")],
        lambda output: _render(output, pits, draws),
        check=args.check,
    )


if __name__ == "__main__":
    _main()
