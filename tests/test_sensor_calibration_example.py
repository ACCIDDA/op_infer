"""Focused checks for the public synthetic sensor-calibration example."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from op_infer.observe import IIDObservation
from op_infer.priors import JointPrior, Normal, PriorParameter

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "sensor_calibration_study.py"
)


def test_sensor_calibration_check_is_replayable() -> None:
    """The bounded workflow is finite, explicit and exactly replayable."""
    namespace = runpy.run_path(str(EXAMPLE))
    first = namespace["run_study"](check=True)
    second = namespace["run_study"](check=True)

    assert first == second
    assert first["status"] == "illustrative_success"
    assert first["mode"] == "check"
    assert first["n_calibration_readings"] == 6
    assert first["held_out_check"]["n_readings"] == 4
    assert first["diagnostics"]["design_rank"] == 2
    assert np.isfinite(first["diagnostics"]["log_likelihood_at_posterior_mean"])
    assert first["randomness"]["seed"] == 20261023
    assert first["provenance"]["prior"]["digest"].startswith("sha256:")
    assert len(first["limitations"]) >= 4


def test_sensor_calibration_cli_check_emits_json() -> None:
    """The documented smoke command exits successfully with a JSON record."""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and local script
        [sys.executable, str(EXAMPLE), "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "illustrative_success"
    assert result["diagnostics"]["posterior_draw_count"] == 200


def test_sensor_calibration_refuses_an_unidentified_gain() -> None:
    """A constant reference level fails visibly instead of producing a fit."""
    namespace = runpy.run_path(str(EXAMPLE))
    data = namespace["ReadingSet"](
        reference=np.zeros(3, dtype=np.float64),
        reading=np.ones(3, dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="do not identify both offset and gain"):
        namespace["_fit_exact"](
            data,
            prior_mean=np.array([0.0, 1.0]),
            prior_sd=np.array([0.25, 0.20]),
            noise_sd=0.08,
        )


def test_analytic_inputs_come_from_recorded_declarations() -> None:
    """A full run fits with the recorded prior and observation scale."""
    namespace = runpy.run_path(str(EXAMPLE))
    prior = JointPrior(
        parameters=(
            PriorParameter(name="offset", distribution=Normal(loc=-0.1, scale=0.4)),
            PriorParameter(name="gain", distribution=Normal(loc=0.9, scale=0.3)),
        )
    )
    observation = IIDObservation(family="normal", scale=0.13)

    result = namespace["run_study"](
        check=True,
        prior=prior,
        observation=observation,
    )
    calibration_levels, _ = namespace["_levels"](check=True)
    calibration = namespace["_generate"](
        calibration_levels,
        observation=observation,
        stream="sensor_calibration/training",
    )
    x = calibration.reference
    y = calibration.reading
    noise_variance = 0.13**2
    p00 = 1 / 0.4**2 + x.size / noise_variance
    p01 = float(np.sum(x) / noise_variance)
    p11 = 1 / 0.3**2 + float(np.sum(x**2) / noise_variance)
    determinant = p00 * p11 - p01**2
    expected_covariance = np.array(
        [
            [p11 / determinant, -p01 / determinant],
            [-p01 / determinant, p00 / determinant],
        ]
    )
    right_hand_side = np.array(
        [
            -0.1 / 0.4**2 + float(np.sum(y) / noise_variance),
            0.9 / 0.3**2 + float(np.sum(x * y) / noise_variance),
        ]
    )
    expected_mean = expected_covariance @ right_hand_side

    assert result["assumptions"]["prior_mean_offset_gain"] == [-0.1, 0.9]
    assert result["assumptions"]["prior_sd_offset_gain"] == [0.4, 0.3]
    assert result["assumptions"]["known_noise_sd_sensor_units"] == 0.13
    assert result["provenance"]["prior"] == prior.reference.model_dump(mode="json")
    assert result["provenance"]["observation_model"] == (
        observation.reference.model_dump(mode="json")
    )
    np.testing.assert_allclose(
        [
            result["parameters"]["offset"]["posterior_mean"],
            result["parameters"]["gain"]["posterior_mean"],
        ],
        expected_mean,
    )


@pytest.mark.parametrize(
    "observation",
    [
        IIDObservation(family="normal", offset=0.5),
        IIDObservation(family="normal", slope=2.0),
        IIDObservation(family="normal", lower_limit=-1.0),
    ],
)
def test_analytic_study_refuses_transformed_or_censored_observations(
    observation: IIDObservation,
) -> None:
    """The closed-form fit refuses observation declarations it does not model."""
    namespace = runpy.run_path(str(EXAMPLE))
    with pytest.raises(
        RuntimeError,
        match="uncensored Normal observation with offset=0 and slope=1",
    ):
        namespace["run_study"](check=True, observation=observation)


def test_analytic_study_refuses_shaped_prior_coordinates() -> None:
    """The scalar affine fit rejects array-valued prior coordinates early."""
    namespace = runpy.run_path(str(EXAMPLE))
    prior = JointPrior(
        parameters=(
            PriorParameter(
                name="offset",
                distribution=Normal(loc=0.0, scale=0.25),
                shape=(2,),
            ),
            PriorParameter(name="gain", distribution=Normal(loc=1.0, scale=0.20)),
        )
    )
    with pytest.raises(RuntimeError, match="scalar offset and gain"):
        namespace["run_study"](check=True, prior=prior)


def test_holdout_summary_matches_independent_predictive_formula() -> None:
    """Held-out residuals and density include posterior and observation variance."""
    namespace = runpy.run_path(str(EXAMPLE))
    mean = np.array([0.1, 1.2])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    reference = np.array([-0.5, 0.25])
    reading = np.array([-0.35, 0.55])
    noise_sd = 0.2
    fit = namespace["GaussianFit"](
        mean=mean,
        covariance=covariance,
        precision_condition_number=1.0,
    )
    holdout = namespace["ReadingSet"](reference=reference, reading=reading)

    result = namespace["_holdout_summary"](holdout, fit, noise_sd=noise_sd)
    predictive_mean = mean[0] + mean[1] * reference
    predictive_variance = (
        covariance[0, 0]
        + 2 * reference * covariance[0, 1]
        + reference**2 * covariance[1, 1]
        + noise_sd**2
    )
    residuals = (reading - predictive_mean) / np.sqrt(predictive_variance)
    log_density = -0.5 * (np.log(2 * np.pi * predictive_variance) + residuals**2)

    np.testing.assert_allclose(result["standardized_residuals"], residuals)
    assert result["mean_log_predictive_density"] == pytest.approx(
        float(np.mean(log_density))
    )
    assert result["within_nominal_95_interval"] == int(
        np.count_nonzero(np.abs(residuals) <= 1.959963984540054)
    )
