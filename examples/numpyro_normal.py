"""Run a native Gaussian fit; print raw diagnostics without an adequacy claim."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from op_infer.adapters.numpyro import (
    NutsSettings,
    SiteLayout,
    fit_nuts,
    numpyro_backend_versions,
)
from op_infer.posterior import (
    ArtifactReference,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.rng import RandomKey

if TYPE_CHECKING:
    from jax import Array


def model(data: Array) -> None:
    """Define the native prior and observation model for the Gaussian example."""
    values = jnp.asarray(data)
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))
    with numpyro.plate("observation", values.shape[0]):
        numpyro.sample("measurement", dist.Normal(theta, 1.0), obs=values)


def main() -> None:
    """Fit the Gaussian model and compare with its analytic reference."""
    observed = np.asarray([0.5, 1.5, 1.0, 2.0])
    config = NutsSettings(
        key=RandomKey(seed=2026, stream="reference_fit"),
        sites=(SiteLayout(name="theta"),),
        num_warmup=100,
        num_samples=512,
    )
    source = ArtifactReference.from_bytes(
        Path(__file__).read_bytes(), kind="normal-example-source-v1"
    )
    source_manifest = {"example": source.digest}
    for name in (
        "op_infer",
        "op_infer.adapters",
        "op_infer.adapters.numpyro",
        "op_infer.adapters.rng",
        "op_infer.posterior",
        "op_infer.rng",
    ):
        location = importlib.import_module(name).__file__
        if location is None:
            msg = f"Cannot record example source identity for {name}"
            raise RuntimeError(msg)
        source_manifest[name] = ArtifactReference.from_bytes(
            Path(location).read_bytes(), kind="python-source-v1"
        ).digest
    code = ArtifactReference.from_bytes(
        json.dumps(source_manifest, sort_keys=True).encode(),
        kind="normal-example-source-manifest-v1",
    )
    prior = ArtifactReference.from_bytes(b"theta~Normal(0,1)", kind="prior-v1")
    design = ArtifactReference.from_bytes(
        json.dumps({"independent_observations": observed.tolist()}).encode(),
        kind="normal-data-v1",
    )
    target = ArtifactReference.from_bytes(
        json.dumps(
            {"prior": prior.digest, "data": design.digest, "model": source.digest}
        ).encode(),
        kind="analytic-normal-target-v1",
    )
    provenance = Provenance(
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
                reason="Analytic smoke fixture; no general target certification",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="reference_fit",
        estimator_settings=config.reference,
        backend_versions=numpyro_backend_versions(),
        commit=code,
    )
    result = fit_nuts(
        model,
        settings=config,
        provenance=provenance,
        model_args=(jnp.asarray(observed),),
    )
    diagnostic = result.diagnostics[0]
    print(  # noqa: T201 -- this executable example emits its result as JSON
        json.dumps(
            {
                "analytic_mean": 1.0,
                "analytic_variance": 0.2,
                "sample_mean": float(np.mean(result.posterior.draws[0].values)),
                "sample_variance": float(np.var(result.posterior.draws[0].values)),
                "numpyro_ess": float(diagnostic.ess),
                "classic_split_r_hat": (
                    None
                    if diagnostic.split_r_hat is None
                    else float(diagnostic.split_r_hat)
                ),
                "divergences_per_chain": result.divergences_per_chain,
                "nonfinite_values": result.nonfinite_values,
                "adequacy": result.adequacy_status,
                "estimator_settings": json.loads(config.manifest_json()),
                "source_manifest": source_manifest,
                "provenance": provenance.identity,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
