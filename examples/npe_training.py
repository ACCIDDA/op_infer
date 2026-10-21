"""Measure native CPU NPE storage and resume a Gaussian demonstration bank.

Run each storage mode in a separate process for meaningful peak RSS readings.
This example tests execution; short fits do not establish posterior calibration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from op_infer.adapters.sbi import (
    NpeSettings,
    fit_npe,
    npe_settings_reference,
    save_npe,
    sbi_backend_versions,
)
from op_infer.posterior import (
    ArtifactReference,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.rng import RandomKey


def _reference(payload: bytes, kind: str) -> ArtifactReference:
    return ArtifactReference.from_bytes(payload, kind=kind)


def _provenance(settings: NpeSettings, rows: int) -> Provenance:
    source = {}
    for name in ("op_infer.adapters.sbi", "op_infer.adapters._sbi_training"):
        location = importlib.import_module(name).__file__
        if location is None:
            msg = f"Missing source for {name}"
            raise RuntimeError(msg)
        source[name] = _reference(
            Path(location).read_bytes(), "python-source-v1"
        ).digest
    source["example"] = _reference(
        Path(__file__).read_bytes(), "python-source-v1"
    ).digest
    code = _reference(json.dumps(source, sort_keys=True).encode(), "source-manifest-v1")
    prior = _reference(b"theta[2]~iid Normal(0,1)", "prior-v1")
    design = _reference(
        json.dumps([rows, settings.n_max, settings.feature_dim, "seed=51"]).encode(),
        "gaussian-cohort-design-v1",
    )
    target = _reference(b"x[i,j,k]~Normal(theta[i,0]+0.3*theta[i,1],1)", "target-v1")
    return Provenance(
        schema_version=1,
        prior=prior,
        design=design,
        simulator=code,
        target=LikelihoodTargetReference(
            kind="analytic",
            intended=target,
            evaluated=target,
            qualification=Qualification(
                status="unqualified",
                scope=target,
                reason="Synthetic execution and memory demonstration, not calibration",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="training_demonstration",
        estimator_settings=npe_settings_reference(settings),
        backend_versions=sbi_backend_versions(),
        commit=code,
    )


def _prepare(directory: Path, rows: int, units: int, features: int) -> None:
    paths = (directory / "theta.npy", directory / "observations.npy")
    if all(path.exists() for path in paths):
        return
    if any(path.exists() for path in paths):
        msg = "Incomplete bank; prepare in a new directory"
        raise ValueError(msg)
    generator = np.random.default_rng(51)
    theta = generator.standard_normal((rows, 2), dtype=np.float32)
    np.save(paths[0], theta)
    # The supported NumPy 2.0 boundary has no typing for this native writer.
    format_module = importlib.import_module("numpy.lib.format")
    mapped = format_module.open_memmap(
        paths[1], mode="w+", dtype=np.float32, shape=(rows, units, features)
    )
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        mapped[start:stop] = (
            theta[start:stop, 0, None, None]
            + 0.3 * theta[start:stop, 1, None, None]
            + generator.standard_normal(
                (stop - start, units, features), dtype=np.float32
            )
        )
    mapped[::3, -1] = np.nan
    mapped.flush()


def main() -> None:
    """Print a native training record; peak RSS includes imports and mapped pages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=("native", "borrowed"), default="borrowed")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--until-epoch", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    _prepare(args.directory, args.rows, args.units, args.features)
    if args.prepare:
        return
    parameters = np.load(args.directory / "theta.npy", mmap_mode="c")
    observations = np.load(args.directory / "observations.npy", mmap_mode="c")
    if parameters.shape != (args.rows, 2) or observations.shape != (
        args.rows,
        args.units,
        args.features,
    ):
        msg = "Existing bank shape differs; use a new directory"
        raise ValueError(msg)
    settings = NpeSettings(
        key=RandomKey(seed=2026, stream="npe_training_demo"),
        parameter_names=("theta", "latent"),
        n_max=args.units,
        feature_dim=args.features,
        hidden_features=8,
        latent_dim=4,
        embedding_dim=4,
        num_transforms=1,
        max_epochs=3,
        batch_size=128,
        mask_probability=0.5,
    )
    provenance = _provenance(settings, args.rows)
    checkpoint = args.directory / f"checkpoint-{args.mode}.pt"
    # Torch 2.6 distribution constructors are also untyped native boundaries.
    distributions = importlib.import_module("torch.distributions")
    prior = distributions.Independent(
        distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    started = time.perf_counter()
    fit = fit_npe(
        torch.from_numpy(parameters),
        torch.from_numpy(observations),
        prior=prior,
        settings=settings,
        provenance=provenance,
        borrow_bank=args.mode == "borrowed",
        checkpoint_path=checkpoint,
        until_epoch=args.until_epoch,
        resume_from=checkpoint if args.resume else None,
    )
    elapsed = time.perf_counter() - started
    fit_path = args.directory / f"fit-{args.mode}.pt"
    save_npe(fit, fit_path)
    record = {
        "mode": args.mode,
        "bank_mib": (parameters.nbytes + observations.nbytes) / 1024**2,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if platform.system() == "Darwin" else 1024),
        "elapsed_seconds": elapsed,
        "completed_epochs": fit.epochs,
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "provenance": provenance.model_dump(mode="json"),
        "fit_state_digest": torch.load(fit_path, weights_only=True)["state_digest"],
    }
    (args.directory / f"result-{args.mode}.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(record, sort_keys=True))  # noqa: T201 - executable report


if __name__ == "__main__":
    main()
