"""Required optional CPU distribution conformance, JIT, gradients and RNG replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from op_infer.adapters.rng import jax_key
from op_infer.observe import IIDObservation
from op_infer.priors import (
    Beta,
    Gamma,
    JointPrior,
    LogNormal,
    Normal,
    PriorParameter,
    Uniform,
)
from op_infer.rng import RandomKey


@pytest.fixture(params=["jax", "torch"])
def native(request: pytest.FixtureRequest) -> Any:  # noqa: ANN401
    """Use actual dependencies; the required CI recipe imports these before pytest."""
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        pytest.importorskip("numpyro")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        with enable():
            yield pytest.importorskip("jax.numpy")
    else:
        yield pytest.importorskip("torch")


def host(value: Any) -> np.ndarray[Any, Any]:  # noqa: ANN401
    """Convert only at the independent test/receipt boundary."""
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "family",
    ["normal", "lognormal", "poisson", "nb2", "binomial", "beta_binomial", "bernoulli"],
)
def test_native_density_support_moments_and_replay(
    native: Any,  # noqa: ANN401
    family: str,
    dtype: str,
) -> None:
    """Both real backends agree with independently parameterized SciPy densities."""
    model = IIDObservation(family=family, scale=0.4, trials=8, concentration=3.0)  # type: ignore[arg-type]
    location = 0.3
    latent = native.asarray(location, dtype=getattr(native, dtype))
    points = native.asarray([0.0, 1.0, 2.0, -1.0, 0.5], dtype=getattr(native, dtype))
    laws = {
        "normal": stats.norm(location, 0.4),
        "lognormal": stats.lognorm(0.4, scale=np.exp(location)),
        "poisson": stats.poisson(location),
        "nb2": stats.nbinom(3, 3 / (3 + location)),
        "binomial": stats.binom(8, location),
        "beta_binomial": stats.betabinom(8, 3 * location, 3 * (1 - location)),
        "bernoulli": stats.bernoulli(location),
    }
    law = laws[family]
    expected = (
        law.logpdf(host(points))
        if family in {"normal", "lognormal"}
        else law.logpmf(host(points))
    )
    np.testing.assert_allclose(
        host(model.log_prob(latent, points)),
        expected,
        rtol=3e-5 if dtype == "float32" else 1e-12,
        atol=3e-5 if dtype == "float32" else 1e-12,
    )
    key = RandomKey(seed=73, stream="observation")
    state_before = (
        None
        if native.__name__.startswith("jax")
        else native.random.get_rng_state().clone()
    )
    draws = model.sample(latent, key, sample_shape=(12000,))
    np.testing.assert_array_equal(
        host(draws), host(model.sample(latent, key, sample_shape=(12000,)))
    )
    mean, variance = law.stats(moments="mv")
    assert abs(host(draws).mean() - mean) < 7 * np.sqrt(variance / 12000)
    assert abs(host(draws).var() - variance) < 0.12 * variance
    if state_before is not None:
        assert native.equal(state_before, native.random.get_rng_state())


@pytest.mark.parametrize(
    "law",
    [
        Normal(loc=0.3, scale=0.4),
        LogNormal(loc=0.3, scale=0.4),
        Uniform(low=-1.0, high=2.0),
        Gamma(concentration=2.0, rate=3.0),
        Beta(concentration1=2.0, concentration0=3.0),
    ],
)
def test_prior_native_density_shapes_and_gradients(native: Any, law: Any) -> None:  # noqa: ANN401
    """Native site event shapes and physical-coordinate gradients match references."""
    prior = JointPrior(
        parameters=(PriorParameter(name="x", distribution=law, shape=(2,)),)
    )
    points = np.array([[0.2, 0.4], [0.5, 0.7]])
    values = native.asarray(points, dtype=native.float64)
    expected = law.scipy().logpdf(points).sum(-1)
    np.testing.assert_allclose(
        host(prior.log_prob({"x": values})), expected, rtol=1e-12, atol=1e-12
    )
    if native.__name__.startswith("jax"):
        import jax  # noqa: PLC0415

        sites = prior.numpyro_sites()
        actual = jax.jit(jax.grad(lambda x: prior.log_prob({"x": x}).sum()))(values)
        assert sites["x"].event_shape == (2,)
    else:
        values.requires_grad_()
        sites = prior.torch_sites(dtype=native.float64)
        prior.log_prob({"x": values}).sum().backward()
        actual = values.grad
        assert sites["x"].event_shape == (2,)
    epsilon = 1e-5
    reference = (
        law.scipy().logpdf(points + epsilon) - law.scipy().logpdf(points - epsilon)
    ) / (2 * epsilon)
    np.testing.assert_allclose(host(actual), reference, rtol=1e-8, atol=1e-8)


def test_jax_observation_jit_vmap_and_gradient_mask() -> None:
    """Native keys permit changing noise inside a reused jitted execution graph."""
    jax = pytest.importorskip("jax")
    xp = pytest.importorskip("jax.numpy")
    pytest.importorskip("numpyro")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        model = IIDObservation(
            family="normal", scale=0.5, lower_limit=-1.0, upper_limit=2.0
        )
        data = xp.array([0.0, xp.nan])
        mask = xp.array([True, False])
        value, gradient = jax.jit(
            jax.value_and_grad(lambda m: model.log_prob(m, data, mask=mask).sum())
        )(xp.array([0.3, 0.7]))
        np.testing.assert_allclose(value, stats.norm(0.3, 0.5).logpdf(0.0))
        np.testing.assert_allclose(gradient, [-1.2, 0.0])
        keys = jax.random.split(jax_key(RandomKey(seed=73, stream="s")), 4)
        draws = jax.jit(jax.vmap(lambda key: model.sample(xp.array(0.3), key)))(keys)
        assert len(np.unique(draws)) > 1


def test_torch_masked_gradient_and_sbi_vector_layout() -> None:
    """Missing cells have zero gradient and sbi receives a real-vector native prior."""
    torch = pytest.importorskip("torch")
    model = IIDObservation(family="normal", scale=0.5)
    latent = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    density = model.log_prob(
        latent, torch.tensor([0.0, float("nan")]), mask=torch.tensor([True, False])
    ).sum()
    density.backward()
    np.testing.assert_allclose(latent.grad, [-1.2, 0.0])
    prior = JointPrior(
        parameters=(
            PriorParameter(name="a", distribution=Normal(loc=2.0, scale=3.0)),
            PriorParameter(name="b", distribution=Normal(loc=-1.0, scale=0.5)),
        )
    )
    vector = prior.torch_normal_vector(dtype=torch.float64)
    assert vector.event_shape == (2,)
    values = torch.tensor([[2.0, -1.0], [3.0, 0.0]], dtype=torch.float64)
    np.testing.assert_allclose(
        vector.log_prob(values), prior.log_prob({"a": values[:, 0], "b": values[:, 1]})
    )
    with pytest.raises(ValueError, match="scalar Normal"):
        JointPrior(
            parameters=(PriorParameter(name="a", distribution=Uniform()),)
        ).torch_normal_vector()


@pytest.mark.parametrize("family", ["poisson", "nb2", "binomial", "bernoulli"])
def test_native_count_boundary_mass_is_exact(native: Any, family: str) -> None:  # noqa: ANN401
    """Impossible boundary events remain impossible despite native logit clamps."""
    model = IIDObservation(family=family, trials=4)  # type: ignore[arg-type]
    latent = native.asarray(0.0, dtype=native.float64)
    points = native.asarray([0.0, 1.0], dtype=native.float64)
    np.testing.assert_array_equal(host(model.log_prob(latent, points)), [0.0, -np.inf])


@pytest.mark.parametrize(
    "family",
    ["normal", "lognormal", "poisson", "nb2", "binomial", "beta_binomial", "bernoulli"],
)
def test_native_observation_gradient_matches_independent_density(
    native: Any,  # noqa: ANN401
    family: str,
) -> None:
    """Physical latent gradients agree with centered SciPy reference differences."""
    model = IIDObservation(family=family, scale=0.4, concentration=3.0, trials=8)  # type: ignore[arg-type]
    location = native.asarray(0.3, dtype=native.float64)
    observed = native.asarray(1.0, dtype=native.float64)
    if native.__name__.startswith("jax"):
        import jax  # noqa: PLC0415

        gradient = jax.jit(jax.grad(lambda x: model.log_prob(x, observed)))(location)
    else:
        location.requires_grad_()
        model.log_prob(location, observed).backward()
        gradient = location.grad
    epsilon = 1e-5
    reference = (
        model.log_prob(np.array(0.3 + epsilon), np.array(1.0))
        - model.log_prob(np.array(0.3 - epsilon), np.array(1.0))
    ) / (2 * epsilon)
    np.testing.assert_allclose(host(gradient), reference, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "law",
    [
        Normal(),
        LogNormal(loc=0.3, scale=0.4),
        Uniform(),
        Gamma(concentration=2.0, rate=3.0),
        Beta(concentration1=2.0, concentration0=3.0),
    ],
)
def test_native_prior_draws_match_independent_moments(native: Any, law: Any) -> None:  # noqa: ANN401
    """Actual native sampling checks lowerings beyond their interior density values."""
    if native.__name__.startswith("jax"):
        values = law.numpyro().sample(
            jax_key(RandomKey(seed=73, stream="parameters")), sample_shape=(12000,)
        )
    else:
        with native.random.fork_rng():
            native.manual_seed(73)
            values = law.torch(dtype=native.float64).sample((12000,))
    mean, variance = law.scipy().stats(moments="mv")
    assert abs(host(values).mean() - mean) < 7 * np.sqrt(variance / 12000)
    assert abs(host(values).var() - variance) < 0.12 * variance
