"""Modern diagnostics on the actual native NumPyro posterior boundary."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from op_infer.adapters.arviz import diagnose_mcmc
from op_infer.adapters.numpyro import SiteLayout, fit_nuts, numpyro_backend_versions
from op_infer.posterior import Coordinate
from tests.test_numpyro_nuts import normal_model
from tests.test_numpyro_settings import fit_provenance, settings
from tests.test_provenance import reference

pytest.importorskip("arviz_stats")
pytest.importorskip("numpyro")
jax = pytest.importorskip("jax")
xp = importlib.import_module("jax.numpy")
array_stats = importlib.import_module("arviz_stats.base").array_stats


def test_real_native_posterior_preserves_named_quantities_and_classic_diagnostics() -> (
    None
):
    """An explicit transfer reports modern metrics without changing native results."""
    config = settings(
        num_samples=128,
        sites=(
            SiteLayout(name="pair", dims=("side",)),
            SiteLayout(name="theta"),
            SiteLayout(name="constant"),
        ),
        coordinates=(Coordinate(name="side", labels=("right", "left")),),
    )
    identity = fit_provenance(config, versions=numpyro_backend_versions()).model_copy(
        update={
            "simulator": reference("native Gaussian fixture plus ordered pair/constant")
        }
    )
    fitted = fit_nuts(
        normal_model,
        settings=config,
        provenance=identity,
        model_args=(xp.asarray([0.5, 1.5, 1.0, 2.0]),),
        model_kwargs={"derived": True},
    )
    definition = reference(
        "report pair=(2*theta,theta), theta and constant; retained transitions"
    )
    names = ("pair", "constant", "theta")
    missing = diagnose_mcmc(fitted.posterior, quantities=names, definition=definition)
    assert missing.n_components == missing.n_incomplete == 4
    assert "Explicitly transfer" in (
        missing.quantities[0].components[0].rank_rhat.reason or ""
    )
    classic = fitted.diagnostics
    result = diagnose_mcmc(
        fitted.posterior, quantities=names, definition=definition, to_numpy=np.asarray
    )
    assert result.n_components == 4
    assert result.n_incomplete == 1
    assert tuple(q.name for q in result.quantities) == names
    assert result.coordinates == fitted.posterior.coordinates
    assert result.provenance == fitted.posterior.provenance
    assert result.quantities[0].dims == ("side",)
    assert result.quantities[1].components[0].rank_rhat.reason == "constant_quantity"
    theta = next(v for v in fitted.posterior.draws if v.name == "theta")
    values = np.asarray(theta.values, dtype=np.float64)
    assert abs(values.mean() - 1.0) < 0.15
    assert abs(values.var() - 0.2) < 0.08
    assert result.quantities[2].components[0].rank_rhat.value == pytest.approx(
        float(array_stats.rhat(values, chain_axis=0, draw_axis=1, method="rank"))
    )
    assert result.quantities[2].components[0].tail_ess.value == pytest.approx(
        float(
            array_stats.ess(
                values,
                chain_axis=0,
                draw_axis=1,
                method="tail",
                relative=False,
                prob=(0.05, 0.95),
            )
        )
    )
    assert fitted.diagnostics is classic
    assert fitted.adequacy_status == "not_assessed"
    assert result.adequacy_status == "not_assessed"
    assert isinstance(theta.values, jax.Array)
