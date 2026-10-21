"""Independent observation equations, censoring, masks and sampling moments."""

import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from op_infer.observe import IIDObservation
from op_infer.rng import RandomKey


@pytest.mark.parametrize(
    ("model", "latent", "mean", "variance"),
    [
        (IIDObservation(family="normal", scale=0.4), 1.0, 1.0, 0.16),
        (
            IIDObservation(family="lognormal", scale=0.4),
            0.3,
            math.exp(0.38),
            math.expm1(0.16) * math.exp(0.76),
        ),
        (IIDObservation(family="poisson"), 2.0, 2.0, 2.0),
        (IIDObservation(family="nb2", concentration=3.0), 2.0, 2.0, 2 + 4 / 3),
        (IIDObservation(family="binomial", trials=10), 0.3, 3.0, 2.1),
        (
            IIDObservation(family="beta_binomial", trials=10, concentration=4.0),
            0.3,
            3.0,
            2.1 * 14 / 5,
        ),
        (
            IIDObservation(family="bernoulli", sensitivity=0.8, specificity=0.9),
            0.3,
            0.31,
            0.31 * 0.69,
        ),
    ],
)
def test_sampling_moments_and_simulate_then_score(
    model: IIDObservation, latent: float, mean: float, variance: float
) -> None:
    """Independent moments expose reversed NB probabilities and beta concentration."""
    location = np.asarray(latent)
    key = RandomKey(seed=73, stream="measurements")
    draws = model.sample(location, key, sample_shape=(30000,))
    np.testing.assert_array_equal(
        draws, model.sample(location, key, sample_shape=(30000,))
    )
    assert abs(draws.mean() - mean) < 6 * math.sqrt(variance / len(draws))
    assert abs(draws.var() - variance) < 0.08 * variance
    assert np.isfinite(model.log_prob(location, draws)).all()


def test_mask_affine_map_and_exact_normal_equation() -> None:
    """Missing NaNs contribute zero and observed zeros remain observations."""
    model = IIDObservation(family="normal", scale=0.5, offset=1.0, slope=2.0)
    latent = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, float("nan"), 3.0])
    mask = np.array([True, False, True])
    result = model.log_prob(latent, y, mask=mask)
    expected = stats.norm.logpdf(y[mask], loc=(1 + 2 * latent)[mask], scale=0.5)
    np.testing.assert_allclose(result[mask], expected)
    assert result[1] == 0
    with pytest.raises(ValueError, match="mask"):
        model.log_prob(latent, y, mask=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="mask"):
        model.log_prob(latent, y, mask=np.array([True]))


def test_censoring_scores_probability_mass_and_replays_clipping() -> None:
    """Two tails and the interior integrate to one on an independent Normal law."""
    model = IIDObservation(
        family="normal", scale=2.0, lower_limit=-1.0, upper_limit=3.0
    )
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([-1.0, 0.0, 3.0])
    expected = np.array(
        [stats.norm.logcdf(-0.5), stats.norm.logpdf(0, scale=2), stats.norm.logsf(1.5)]
    )
    np.testing.assert_allclose(model.log_prob(x, y), expected)
    assert np.isneginf(model.log_prob(np.array(0.0), np.array(-2.0)))
    draws = model.sample(
        np.array(0.0), RandomKey(seed=1, stream="s"), sample_shape=(30000,)
    )
    assert draws.min() == -1
    assert draws.max() == 3
    assert abs(np.mean(draws == -1) - stats.norm.cdf(-0.5)) < 0.015
    assert abs(np.mean(draws == 3) - stats.norm.sf(1.5)) < 0.01
    with pytest.raises(ValidationError):
        IIDObservation(family="poisson", lower_limit=0.0)
    with pytest.raises(ValidationError):
        IIDObservation(family="normal", lower_limit=1.0, upper_limit=0.0)


def test_counts_reject_fractional_and_out_of_support_measurements() -> None:
    """Invalid observed values are impossible rather than generalized densities."""
    for family in ["poisson", "nb2", "binomial", "beta_binomial", "bernoulli"]:
        model = IIDObservation(family=family, trials=4)  # type: ignore[arg-type]
        assert np.isneginf(
            model.log_prob(np.array(0.5), np.array([-0.5, 0.5, -1.0]))
        ).all()
    model = IIDObservation(family="nb2")
    np.testing.assert_array_equal(
        model.log_prob(np.array(0.0), np.array([0.0, 1.0])), [0.0, -np.inf]
    )
    assert (
        IIDObservation(family="normal").reference
        != IIDObservation(family="normal", scale=2.0).reference
    )
