"""Independent support, shape, moments and addressed prior contracts."""

import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

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


def test_named_streams_replay_and_identity_tracks_structure() -> None:
    """Reordering names preserves their draws but changes the ordered identity."""
    a = PriorParameter(name="a", distribution=Normal(loc=2.0, scale=3.0), shape=(2,))
    b = PriorParameter(name="b", distribution=Gamma(concentration=2.0, rate=3.0))
    prior = JointPrior(parameters=(a, b))
    reordered = JointPrior(parameters=(b, a))
    key = RandomKey(seed=73, stream="parameters", index=4)
    samples = prior.sample(key, sample_shape=(3, 4))
    assert samples["a"].shape == (3, 4, 2)
    assert samples["b"].shape == (3, 4)
    for name, value in samples.items():
        np.testing.assert_array_equal(
            value, reordered.sample(key, sample_shape=(3, 4))[name]
        )
    assert prior.reference != reordered.reference
    assert (
        JointPrior.model_validate_json(prior.model_dump_json()).reference
        == prior.reference
    )
    expected = stats.norm(2, 3).logpdf(samples["a"]).sum(-1) + stats.gamma(
        2, scale=1 / 3
    ).logpdf(samples["b"])
    np.testing.assert_allclose(prior.log_prob(samples), expected)


@pytest.mark.parametrize(
    ("law", "point", "mean", "variance"),
    [
        (Normal(loc=2.0, scale=3.0), 1.0, 2.0, 9.0),
        (
            LogNormal(loc=0.3, scale=0.4),
            1.0,
            math.exp(0.38),
            math.expm1(0.16) * math.exp(0.76),
        ),
        (Uniform(low=-1.0, high=3.0), 1.0, 1.0, 16 / 12),
        (Gamma(concentration=2.0, rate=3.0), 1.0, 2 / 3, 2 / 9),
        (Beta(concentration1=2.0, concentration0=3.0), 0.4, 0.4, 0.04),
    ],
)
def test_scalar_laws_match_independent_moments(
    law: Normal | LogNormal | Uniform | Gamma | Beta,
    point: float,
    mean: float,
    variance: float,
) -> None:
    """Fresh draws and independent moments check shape/rate and log-scale choices."""
    prior = JointPrior(parameters=(PriorParameter(name="x", distribution=law),))
    values = prior.sample(
        RandomKey(seed=73, stream="parameters"), sample_shape=(30000,)
    )["x"]
    assert abs(values.mean() - mean) < 6 * math.sqrt(variance / len(values))
    assert abs(values.var() - variance) < 0.07 * variance
    np.testing.assert_allclose(
        prior.log_prob({"x": np.asarray(point)}), law.scipy().logpdf(point)
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Normal(scale=0.0),
        lambda: Normal(loc=float("nan")),
        lambda: Uniform(low=2.0, high=1.0),
        lambda: Gamma(concentration=-1.0, rate=2.0),
        lambda: Beta(concentration1=0.0, concentration0=1.0),
        lambda: PriorParameter(name="x", distribution=Normal(), shape=(True,)),
        lambda: JointPrior(parameters=()),
        lambda: JointPrior(
            parameters=(PriorParameter(name="x", distribution=Normal()),) * 2
        ),
    ],
)
def test_invalid_declarations_fail_before_sampling(factory: object) -> None:
    """Improper, ambiguous or nonfinite declarations cannot acquire an identity."""
    with pytest.raises((ValidationError, ValueError)):
        factory()  # type: ignore[operator]


def test_support_and_layout_failures_are_explicit() -> None:
    """Uniform endpoint conventions, empty batches and shape/name errors agree."""
    prior = JointPrior(parameters=(PriorParameter(name="x", distribution=Uniform()),))
    assert np.isneginf(prior.log_prob({"x": np.array([1.0, -1.0])})).all()
    assert prior.sample(RandomKey(seed=1, stream="p"), sample_shape=(0,))[
        "x"
    ].shape == (0,)
    with pytest.raises(ValueError, match="names"):
        prior.log_prob({"y": np.array(0.0)})
    with pytest.raises(ValueError, match="sample_shape"):
        prior.sample(RandomKey(seed=1, stream="p"), sample_shape=(True,))
