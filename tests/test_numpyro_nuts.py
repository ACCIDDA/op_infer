"""Real native NUTS qualification against an independent analytic posterior."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_infer.adapters.numpyro import (
    NutsResult,
    SiteLayout,
    fit_nuts,
    numpyro_backend_versions,
)
from op_infer.posterior import Coordinate
from tests.test_numpyro_settings import fit_provenance, settings
from tests.test_provenance import reference

if TYPE_CHECKING:
    from op_infer.adapters.numpyro import NutsSettings
    from op_infer.posterior import Provenance

numpyro = pytest.importorskip("numpyro")
jax = pytest.importorskip("jax")
xp = importlib.import_module("jax.numpy")
dist = importlib.import_module("numpyro.distributions")


def normal_model(data: object, *, derived: bool = False) -> None:
    """One scalar Gaussian prior and four independent Gaussian observations."""
    values = xp.asarray(data)
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))
    with numpyro.plate("observation", values.shape[0]):
        numpyro.sample("measurement", dist.Normal(theta, 1.0), obs=values)
    if derived:
        numpyro.deterministic("pair", xp.stack((2 * theta, theta)))
        numpyro.deterministic("constant", xp.asarray(3.0))


def run(config: NutsSettings, provenance: Provenance | None = None) -> NutsResult:
    """Run the actual native model using dynamic data inputs and explicit identity."""
    identity = provenance or fit_provenance(config, versions=numpyro_backend_versions())
    return fit_nuts(
        normal_model,
        settings=config,
        provenance=identity,
        model_args=(xp.asarray([0.5, 1.5, 1.0, 2.0]),),
    )


@pytest.fixture(scope="module")
def gaussian_fit() -> NutsResult:
    """Retain one actual two-chain fit for the independent boundary assertions."""
    return run(settings())


def test_analytic_gaussian_posterior_and_native_boundary(
    gaussian_fit: NutsResult,
) -> None:
    """Known posterior mean=1, variance=1/5 qualify binding without sampler code."""
    result = gaussian_fit
    variable = result.posterior.draws[0]
    draws = np.asarray(variable.values)
    assert draws.shape == (2, 512)
    assert abs(float(np.mean(draws)) - 1.0) < 0.10
    assert abs(float(np.var(draws)) - 0.2) < 0.06
    native = result.native.get_samples(group_by_chain=True)
    assert variable.values is native["theta"]
    assert isinstance(result.native, importlib.import_module("numpyro.infer").MCMC)
    assert result.posterior.estimator_stage == "reference_fit"
    assert result.posterior.provenance.estimator_settings == settings().reference
    assert result.adequacy_status == "not_assessed"
    assert result.nonfinite_values == 0
    assert sum(result.divergences_per_chain) == 0
    assert result.posterior.coordinates[0].labels == (0, 1)
    assert result.posterior.coordinates[1].labels == tuple(range(512))


def test_native_diagnostics_are_retained_with_their_scope(
    gaussian_fit: NutsResult,
) -> None:
    """Retain sampled fields and the native classic diagnostics."""
    result = gaussian_fit
    native_stats = result.native.get_extra_fields(group_by_chain=True)
    assert {variable.name for variable in result.posterior.sample_stats} == {
        "accept_prob",
        "diverging",
        "num_steps",
        "potential_energy",
    }
    for variable in result.posterior.sample_stats:
        assert variable.values is native_stats[variable.name]
        assert variable.values.shape == (2, 512)
    raw = importlib.import_module("numpyro.diagnostics")
    values = np.asarray(result.posterior.draws[0].values)
    diagnostic = result.diagnostics[0]
    assert diagnostic.name == "theta"
    assert diagnostic.dims == ()
    assert diagnostic.ess.shape == ()
    assert diagnostic.split_r_hat is not None
    np.testing.assert_array_equal(diagnostic.ess, raw.effective_sample_size(values))
    np.testing.assert_array_equal(
        diagnostic.split_r_hat, raw.split_gelman_rubin(values)
    )
    assert np.isfinite(diagnostic.ess).all()
    assert np.isfinite(diagnostic.split_r_hat).all()


def test_exact_native_replay(gaussian_fit: NutsResult) -> None:
    """The same full settings/key/data reproduce native CPU samples and statistics."""
    replay = run(settings())
    np.testing.assert_array_equal(
        gaussian_fit.posterior.draws[0].values, replay.posterior.draws[0].values
    )
    for original, restored in zip(
        gaussian_fit.posterior.sample_stats, replay.posterior.sample_stats, strict=True
    ):
        assert original.name == restored.name
        np.testing.assert_array_equal(original.values, restored.values)


def test_vector_coordinates_constant_sites_and_single_chain() -> None:
    """Retain ordering and undefined constant diagnostics without an adequacy pass."""
    config = settings(
        num_warmup=20,
        num_samples=16,
        num_chains=1,
        sites=(
            SiteLayout(name="pair", dims=("side",)),
            SiteLayout(name="theta"),
            SiteLayout(name="constant"),
        ),
        coordinates=(Coordinate(name="side", labels=("right", "left")),),
    )
    identity = fit_provenance(config, versions=numpyro_backend_versions()).model_copy(
        update={
            "simulator": reference("normal fixture with pair/constant derived outputs")
        }
    )
    result = fit_nuts(
        normal_model,
        settings=config,
        provenance=identity,
        model_args=(xp.asarray([0.5, 1.5, 1.0, 2.0]),),
        model_kwargs={"derived": True},
    )
    assert [variable.name for variable in result.posterior.draws] == [
        "pair",
        "theta",
        "constant",
    ]
    assert result.posterior.draws[0].values.shape == (1, 16, 2)
    assert result.posterior.coordinates[-1].labels == ("right", "left")
    pair, theta, _ = result.posterior.draws
    np.testing.assert_array_equal(pair.values[..., 0], 2 * theta.values)
    np.testing.assert_array_equal(pair.values[..., 1], theta.values)
    assert all(item.split_r_hat is None for item in result.diagnostics)
    constant = next(item for item in result.diagnostics if item.name == "constant")
    assert not np.isfinite(constant.ess).all()
    assert result.adequacy_status == "not_assessed"
    assert len(result.posterior.draws) == len(result.diagnostics) == 3


def test_jax_precision_mismatch_prevents_model_access() -> None:
    """The adapter checks recorded precision without silently changing JAX globals."""
    config = settings(enable_x64=not bool(jax.config.jax_enable_x64))
    with pytest.raises(ValueError, match="precision differs"):
        run(config)


def invalid_model(data: object) -> None:
    """An impossible initialization must surface as an attempted-fit failure."""
    normal_model(data)
    numpyro.factor("impossible", xp.asarray(float("nan")))


def test_native_initialization_failure_is_not_a_posterior() -> None:
    """A failed native fit is propagated; no samples or adequacy pass are fabricated."""
    config = settings(num_warmup=1, num_samples=4, num_chains=1)
    # 0.15 rejects the factor during validation; current NumPyro rejects initialization.
    with pytest.raises(
        (RuntimeError, ValueError), match=r"valid initial|invalid log_factor"
    ):
        fit_nuts(
            invalid_model,
            settings=config,
            provenance=fit_provenance(config, versions=numpyro_backend_versions()),
            model_args=(xp.asarray([0.5, 1.5, 1.0, 2.0]),),
        )


def test_undeclared_native_site_is_not_silently_dropped() -> None:
    """Unexpected native outputs cannot disappear at the wrapper boundary."""
    config = settings(num_warmup=1, num_samples=4, num_chains=1)
    with pytest.raises(ValueError, match="every returned native site"):
        fit_nuts(
            normal_model,
            settings=config,
            provenance=fit_provenance(config, versions=numpyro_backend_versions()),
            model_args=(xp.asarray([0.5, 1.5, 1.0, 2.0]),),
            model_kwargs={"derived": True},
        )
