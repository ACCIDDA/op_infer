"""Independent mathematical qualification for the #41 design prerequisite.

These synthetic reference integrals do not implement a distribution family, inference
adapter or reporting policy. Active consumers #23/#25/#26 must qualify their own code.
"""

import math
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Quadrature = Callable[[int], tuple[FloatArray, FloatArray]]

# NumPy 2.0 omits annotations; both algorithms return float64 nodes/weights.
_hermgauss = cast("Quadrature", np.polynomial.hermite.hermgauss)
_leggauss = cast("Quadrature", np.polynomial.legendre.leggauss)


def _normal_density(value: FloatArray, mean: float, variance: float) -> FloatArray:
    """Evaluate a normal density for the latent-integral reference only."""
    return np.asarray(
        np.exp(-0.5 * (value - mean) ** 2 / variance)
        / math.sqrt(2 * math.pi * variance),
        dtype=np.float64,
    )


def _normal_expectation(
    integrand: Callable[[FloatArray], FloatArray], order: int = 64
) -> float:
    """Integrate against a standard-normal law by Gauss-Hermite quadrature."""
    nodes, weights = _hermgauss(order)
    return float(weights @ integrand(math.sqrt(2) * nodes) / math.sqrt(math.pi))


def _analytic_marginal(y: float, theta: float, variance: float) -> float:
    """Use the independent closed-form Gaussian convolution."""
    return math.exp(-((y - theta) ** 2) / (2 * variance)) / math.sqrt(
        2 * math.pi * variance
    )


@pytest.mark.parametrize("theta", [-1.3, 0.0, 0.8])
@pytest.mark.parametrize(("tau", "sigma"), [(0.7, 0.9), (0.4, 1.2)])
def test_hierarchical_integral_matches_analytic_reference(
    theta: float, tau: float, sigma: float
) -> None:
    """Converged latent integration recovers per-unit and joint exact marginals."""
    exact_units = []
    integrated_units = []
    for observation in [-1.1, 0.6, 1.5]:
        exact = _analytic_marginal(observation, theta, tau**2 + sigma**2)

        def integrand(z: FloatArray, y: float = observation) -> FloatArray:
            return _normal_density(theta + tau * z, y, sigma**2)

        low_order = _normal_expectation(integrand, order=32)
        high_order = _normal_expectation(integrand, order=64)
        assert high_order == pytest.approx(low_order, rel=1e-11)
        assert high_order == pytest.approx(exact, rel=1e-12)
        exact_units.append(exact)
        integrated_units.append(high_order)
    assert math.prod(integrated_units) == pytest.approx(
        math.prod(exact_units), rel=1e-12
    )


@pytest.mark.parametrize("theta", [-1.1, 0.0, 0.8])
def test_bounded_latent_integral_changes_hierarchical_likelihood(theta: float) -> None:
    """Prior renormalization cannot restore omitted conditional latent mass."""
    observation, tau, sigma = 0.6, 0.7, 0.9
    lower, upper = -1.0, 1.4
    nodes, weights = _leggauss(128)
    half_width = (upper - lower) / 2
    z = half_width * nodes + (upper + lower) / 2
    box_integral = float(
        half_width
        * weights
        @ (
            _normal_density(z, theta, tau**2)
            * _normal_density(z, observation, sigma**2)
        )
    )
    marginal = _analytic_marginal(observation, theta, tau**2 + sigma**2)
    conditional_mean = (sigma**2 * theta + tau**2 * observation) / (tau**2 + sigma**2)
    conditional_sd = math.sqrt(tau**2 * sigma**2 / (tau**2 + sigma**2))
    conditional_mass = 0.5 * (
        math.erf((upper - conditional_mean) / (math.sqrt(2) * conditional_sd))
        - math.erf((lower - conditional_mean) / (math.sqrt(2) * conditional_sd))
    )
    prior_mass = 0.5 * (
        math.erf((upper - theta) / (math.sqrt(2) * tau))
        - math.erf((lower - theta) / (math.sqrt(2) * tau))
    )
    assert box_integral == pytest.approx(marginal * conditional_mass, rel=1e-12)
    assert box_integral < marginal
    assert box_integral / prior_mass != pytest.approx(marginal, rel=1e-4)


@pytest.mark.parametrize("theta", [-1.1, 0.0, 0.8])
def test_unbiased_likelihood_has_biased_log(theta: float) -> None:
    """An augmented marginal is correct although its log estimate is biased."""
    likelihood = _analytic_marginal(0.6, theta, 0.7**2 + 0.9**2)
    scale = 0.25 + theta**2

    def log_estimate(u: FloatArray) -> FloatArray:
        return math.log(likelihood) + scale * u - scale**2 / 2

    integrated = _normal_expectation(lambda u: np.exp(log_estimate(u)))
    assert integrated == pytest.approx(likelihood, rel=1e-12)
    assert _normal_expectation(log_estimate) == pytest.approx(
        math.log(likelihood) - scale**2 / 2
    )
    assert _normal_expectation(log_estimate) != pytest.approx(math.log(likelihood))


@pytest.mark.parametrize("scale", [0.4, 1.2, 1.8])
def test_unbiased_log_is_not_an_unbiased_likelihood(scale: float) -> None:
    """Exponentiating an unbiased log estimate changes the augmented marginal."""
    likelihood = _analytic_marginal(0.6, 0.2, 1.3)

    def log_estimate(u: FloatArray) -> FloatArray:
        return math.log(likelihood) + scale * u

    assert _normal_expectation(log_estimate) == pytest.approx(math.log(likelihood))
    integrated = _normal_expectation(lambda u: np.exp(log_estimate(u)))
    assert integrated == pytest.approx(likelihood * math.exp(scale**2 / 2))
    assert integrated != pytest.approx(likelihood)


@pytest.mark.parametrize("frozen_draw", [-1.0, 0.0, 0.7])
def test_frozen_randomness_changes_target_ratios(frozen_draw: float) -> None:
    """A shared fixed draw is reproducible but fails to integrate auxiliary state."""
    first_theta, second_theta = 0.1, 0.9
    first = _analytic_marginal(0.6, first_theta, 1.3)
    second = _analytic_marginal(0.6, second_theta, 1.3)
    first_scale = 0.25 + first_theta**2
    second_scale = 0.25 + second_theta**2
    first_frozen = first * math.exp(first_scale * frozen_draw - first_scale**2 / 2)
    second_frozen = second * math.exp(second_scale * frozen_draw - second_scale**2 / 2)
    assert first_frozen / second_frozen != pytest.approx(first / second, rel=1e-4)


@pytest.mark.parametrize("scale", [0.25, 1.0, 2.0])
def test_auxiliary_box_omits_parameter_dependent_mass(scale: float) -> None:
    """Bounded integration and conditional renormalization both change likelihood."""
    bound = 1.5
    nodes, weights = _leggauss(128)
    u = bound * nodes
    likelihood = _analytic_marginal(0.6, math.sqrt(scale - 0.25), 1.3)
    box_integral = float(
        bound
        * weights
        @ (likelihood * np.exp(scale * u - scale**2 / 2) * _normal_density(u, 0.0, 1.0))
    )
    shifted_mass = 0.5 * (
        math.erf((bound - scale) / math.sqrt(2))
        - math.erf((-bound - scale) / math.sqrt(2))
    )
    auxiliary_mass = math.erf(bound / math.sqrt(2))
    assert box_integral == pytest.approx(likelihood * shifted_mass, rel=1e-12)
    assert box_integral < likelihood
    assert box_integral / auxiliary_mass != pytest.approx(likelihood, rel=1e-4)


def test_shared_auxiliary_draw_biases_product_of_unbiased_unit_estimates() -> None:
    """Correct individual expectations do not guarantee a correct joint product."""
    first, second = 0.2, 0.3
    first_scale, second_scale = 0.6, 1.1

    def first_estimate(u: FloatArray) -> FloatArray:
        return np.asarray(
            first * np.exp(first_scale * u - first_scale**2 / 2), dtype=np.float64
        )

    def second_estimate(u: FloatArray) -> FloatArray:
        return np.asarray(
            second * np.exp(second_scale * u - second_scale**2 / 2), dtype=np.float64
        )

    independent_product = _normal_expectation(first_estimate) * _normal_expectation(
        second_estimate
    )
    shared_product = _normal_expectation(
        lambda u: first_estimate(u) * second_estimate(u)
    )
    assert independent_product == pytest.approx(first * second, rel=1e-12)
    assert shared_product == pytest.approx(
        first * second * math.exp(first_scale * second_scale), rel=1e-12
    )
    assert shared_product > independent_product


def test_finite_raw_is_number_can_be_inadequate() -> None:
    """A finite raw-IS fallback and many draws do not guarantee useful ESS."""
    log_weights = np.full(1000, -100.0)
    log_weights[0] = 0.0
    log_weights[1] = -math.inf
    weights = np.exp(log_weights)
    estimate = float(np.mean(weights))
    effective_size = float(np.sum(weights) ** 2 / np.sum(weights**2))
    assert math.isfinite(estimate)
    assert np.count_nonzero(np.isneginf(log_weights)) == 1
    assert effective_size == pytest.approx(1.0)
    # The contract requires a declared policy; no runtime diagnostic gate exists yet.
    assert effective_size < 100


def test_documented_unit_counts_and_conditional_denominators() -> None:
    """Excluded, non-finite and finite-but-inadequate units remain distinguishable."""
    first = np.array([1, -math.inf, 3, math.nan, 5, math.inf])
    second = np.array([0, 2, -math.inf, math.nan, 4, 0])
    included = np.array([True, True, True, True, True, False])
    adequate = np.array([True, True, True, True, False, True])
    first_bad = ~np.isfinite(first)
    second_bad = ~np.isfinite(second)
    eligible_count = int(np.count_nonzero(included))
    paired_finite = included & ~first_bad & ~second_bad
    paired_usable = paired_finite & adequate
    assert first.size == 6
    assert eligible_count == 5
    assert np.count_nonzero(first_bad) == 3  # Includes the excluded unit's infinity.
    assert np.count_nonzero(first_bad & included) == 2
    assert np.count_nonzero(second_bad & included) == 2
    assert np.count_nonzero(first_bad & second_bad & included) == 1
    assert np.count_nonzero(paired_finite) == 5 - 2 - 2 + 1 == 2
    assert np.flatnonzero(paired_finite).tolist() == [0, 4]
    assert np.count_nonzero(paired_usable) == 1
    assert np.mean(first[paired_usable] - second[paired_usable]) == 1
    assert np.count_nonzero(included & ~adequate) / eligible_count == 1 / 5
