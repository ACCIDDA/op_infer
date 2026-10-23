"""Fit an invented scalar sensor calibration with an analytic Gaussian model.

This example uses independent reference readings to estimate a sensor offset and
gain.  It is a software demonstration, not a calibration certificate or a
recommended laboratory protocol.  Run a quick check with
``python examples/sensor_calibration_study.py --check``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from op_infer.observe import IIDObservation
from op_infer.posterior import ArtifactReference
from op_infer.priors import JointPrior, Normal, PriorParameter
from op_infer.rng import RandomKey

if TYPE_CHECKING:
    from numpy.typing import NDArray


SEED = 20261023
NOISE_SD = 0.08
TRUE_OFFSET = 0.12
TRUE_GAIN = 1.08


class CalibrationExampleError(RuntimeError):
    """Report an anticipated data or numerical failure without hiding its cause."""


@dataclass(frozen=True)
class ReadingSet:
    """Keep reference values and corresponding scalar readings together."""

    reference: NDArray[np.float64]
    reading: NDArray[np.float64]


@dataclass(frozen=True)
class GaussianFit:
    """Store exact moments and basic numerical diagnostics for the affine fit."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    precision_condition_number: float


def _reference(payload: object, *, kind: str) -> ArtifactReference:
    """Hash a JSON-serializable object using stable, explicit JSON bytes."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return ArtifactReference.from_bytes(encoded, kind=kind)


def _levels(*, check: bool) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return centered calibration levels and separate intermediate holdouts."""
    if check:
        calibration = np.repeat(np.array([-1.0, 0.0, 1.0]), 2)
        holdout = np.array([-0.75, -0.25, 0.25, 0.75])
    else:
        calibration = np.repeat(np.linspace(-1.0, 1.0, 5), 3)
        holdout = np.repeat(np.array([-0.75, -0.25, 0.25, 0.75]), 2)
    return calibration, holdout


def _normal_prior_arrays(
    prior: JointPrior,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read analytic-fit inputs from the recorded ordered prior declaration."""
    if prior.parameter_names != ("offset", "gain"):
        msg = "The analytic fit requires ordered offset and gain prior coordinates"
        raise CalibrationExampleError(msg)
    mean_values: list[float] = []
    scale_values: list[float] = []
    for parameter in prior.parameters:
        if parameter.shape:
            msg = "The analytic fit requires scalar offset and gain prior coordinates"
            raise CalibrationExampleError(msg)
        distribution = parameter.distribution
        if not isinstance(distribution, Normal):
            msg = "The analytic fit requires independent Normal prior coordinates"
            raise CalibrationExampleError(msg)
        mean_values.append(distribution.loc)
        scale_values.append(distribution.scale)
    means = np.asarray(mean_values, dtype=np.float64)
    scales = np.asarray(scale_values, dtype=np.float64)
    return means, scales


def _generate(
    reference: NDArray[np.float64],
    *,
    observation: IIDObservation,
    stream: str,
) -> ReadingSet:
    """Generate one fixed synthetic reading set from an addressed random stream."""
    latent = TRUE_OFFSET + TRUE_GAIN * reference
    reading = np.asarray(
        observation.sample(latent, RandomKey(seed=SEED, stream=stream)),
        dtype=np.float64,
    )
    return ReadingSet(reference=reference, reading=reading)


def _fit_exact(
    data: ReadingSet,
    *,
    prior_mean: NDArray[np.float64],
    prior_sd: NDArray[np.float64],
    noise_sd: float,
) -> GaussianFit:
    """Return the exact posterior for a Normal affine model with known noise.

    The two columns of the design matrix are the offset and gain effects.  This
    is conjugate linear-Gaussian algebra, not a newly implemented sampler.
    """
    if data.reference.shape != data.reading.shape or data.reference.ndim != 1:
        msg = "Reference values and readings must be matching one-dimensional arrays"
        raise CalibrationExampleError(msg)
    if data.reference.size < 2 or not np.all(
        np.isfinite(np.concatenate((data.reference, data.reading)))
    ):
        msg = "At least two finite reference-reading pairs are required"
        raise CalibrationExampleError(msg)
    design = np.column_stack((np.ones(data.reference.size), data.reference))
    if np.linalg.matrix_rank(design) < 2:
        msg = "The calibration levels do not identify both offset and gain"
        raise CalibrationExampleError(msg)

    prior_precision = np.diag(1.0 / np.square(prior_sd))
    posterior_precision = prior_precision + design.T @ design / noise_sd**2
    condition = float(np.linalg.cond(posterior_precision))
    if not np.isfinite(condition) or condition > 1e12:
        msg = "The posterior precision matrix is too ill-conditioned"
        raise CalibrationExampleError(msg)
    try:
        covariance = np.linalg.inv(posterior_precision)
        np.linalg.cholesky(covariance)
        mean = covariance @ (
            prior_precision @ prior_mean + design.T @ data.reading / noise_sd**2
        )
    except np.linalg.LinAlgError as error:
        msg = "The posterior covariance is not positive definite"
        raise CalibrationExampleError(msg) from error
    return GaussianFit(
        mean=np.asarray(mean),
        covariance=np.asarray(covariance),
        precision_condition_number=condition,
    )


def _parameter_summary(
    fit: GaussianFit, prior_sd: NDArray[np.float64]
) -> dict[str, dict[str, float]]:
    """Summarize exact marginal posteriors and uncertainty contraction."""
    labels = ("offset", "gain")
    truth = (TRUE_OFFSET, TRUE_GAIN)
    posterior_sd = np.sqrt(np.diag(fit.covariance))
    critical = float(norm.ppf(0.975))
    return {
        name: {
            "true_value": true,
            "posterior_mean": float(fit.mean[index]),
            "posterior_sd": float(posterior_sd[index]),
            "credible_interval_95_lower": float(
                fit.mean[index] - critical * posterior_sd[index]
            ),
            "credible_interval_95_upper": float(
                fit.mean[index] + critical * posterior_sd[index]
            ),
            "posterior_to_prior_sd_ratio": float(posterior_sd[index] / prior_sd[index]),
        }
        for index, (name, true) in enumerate(zip(labels, truth, strict=True))
    }


def _holdout_summary(
    holdout: ReadingSet, fit: GaussianFit, *, noise_sd: float
) -> dict[str, object]:
    """Evaluate descriptive posterior predictions on separately generated readings."""
    design = np.column_stack((np.ones(holdout.reference.size), holdout.reference))
    predictive_mean = design @ fit.mean
    predictive_variance = (
        np.einsum("ij,jk,ik->i", design, fit.covariance, design) + noise_sd**2
    )
    predictive_sd = np.sqrt(predictive_variance)
    standardized_residual = (holdout.reading - predictive_mean) / predictive_sd
    within_95 = np.abs(standardized_residual) <= norm.ppf(0.975)
    return {
        "n_readings": int(holdout.reference.size),
        "mean_log_predictive_density": float(
            np.mean(norm.logpdf(holdout.reading, predictive_mean, predictive_sd))
        ),
        "standardized_residuals": standardized_residual.tolist(),
        "within_nominal_95_interval": int(np.count_nonzero(within_95)),
        "interpretation": (
            "Descriptive check only; this small synthetic holdout does not establish "
            "coverage or calibration."
        ),
    }


def run_study(
    *,
    check: bool = False,
    prior: JointPrior | None = None,
    observation: IIDObservation | None = None,
) -> dict[str, object]:
    """Generate, fit and check one replayable synthetic calibration study."""
    if prior is None:
        prior = JointPrior(
            parameters=(
                PriorParameter(name="offset", distribution=Normal(loc=0.0, scale=0.25)),
                PriorParameter(name="gain", distribution=Normal(loc=1.0, scale=0.20)),
            )
        )
    if observation is None:
        observation = IIDObservation(family="normal", scale=NOISE_SD)
    if observation.family != "normal":
        msg = "The analytic sensor example requires a Normal observation declaration"
        raise CalibrationExampleError(msg)
    if (
        observation.offset != 0.0
        or observation.slope != 1.0
        or observation.lower_limit is not None
        or observation.upper_limit is not None
    ):
        msg = (
            "The analytic sensor example requires an uncensored Normal observation "
            "with offset=0 and slope=1"
        )
        raise CalibrationExampleError(msg)
    prior_mean, prior_sd = _normal_prior_arrays(prior)
    noise_sd = float(observation.scale)
    calibration_levels, holdout_levels = _levels(check=check)
    calibration = _generate(
        calibration_levels,
        observation=observation,
        stream="sensor_calibration/training",
    )
    holdout = _generate(
        holdout_levels,
        observation=observation,
        stream="sensor_calibration/holdout",
    )
    fit = _fit_exact(
        calibration,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        noise_sd=noise_sd,
    )

    posterior_draws = (
        RandomKey(seed=SEED, stream="sensor_calibration/posterior_check")
        .numpy_generator()
        .multivariate_normal(
            fit.mean,
            fit.covariance,
            size=200 if check else 4_000,
        )
    )
    posterior_mean_error = np.max(np.abs(posterior_draws.mean(axis=0) - fit.mean))
    # Score the fitted latent mean through the same declared observation law.
    fitted_latent = fit.mean[0] + fit.mean[1] * calibration.reference
    log_likelihood = observation.log_prob(fitted_latent, calibration.reading)
    prior_log_density = prior.log_prob(
        {
            "offset": np.asarray(fit.mean[0]),
            "gain": np.asarray(fit.mean[1]),
        }
    )
    data_manifest = {
        "reference": calibration.reference.tolist(),
        "reading": calibration.reading.tolist(),
        "held_out_reference": holdout.reference.tolist(),
        "held_out_reading": holdout.reading.tolist(),
    }
    source = ArtifactReference.from_bytes(
        Path(__file__).read_bytes(), kind="sensor-calibration-example-source-v1"
    )
    return {
        "status": "illustrative_success",
        "mode": "check" if check else "full",
        "question": "What offset and gain map reference values to sensor readings?",
        "assumptions": {
            "model": "reading = offset + gain * reference + independent Normal noise",
            "known_noise_sd_sensor_units": noise_sd,
            "prior_mean_offset_gain": prior_mean.tolist(),
            "prior_sd_offset_gain": prior_sd.tolist(),
            "prior": "independent Normal coordinates",
            "measurement_dependence": "independent by construction",
        },
        "synthetic_truth": {"offset": TRUE_OFFSET, "gain": TRUE_GAIN},
        "n_calibration_readings": int(calibration.reference.size),
        "parameters": _parameter_summary(fit, prior_sd),
        "held_out_check": _holdout_summary(holdout, fit, noise_sd=noise_sd),
        "diagnostics": {
            "design_rank": 2,
            "posterior_precision_condition_number": (fit.precision_condition_number),
            "posterior_draw_count": int(posterior_draws.shape[0]),
            "posterior_draw_mean_max_absolute_error": float(posterior_mean_error),
            "log_likelihood_at_posterior_mean": float(np.sum(log_likelihood)),
            "prior_log_density_at_posterior_mean": float(prior_log_density),
        },
        "randomness": {
            "seed": SEED,
            "streams": [
                "sensor_calibration/training",
                "sensor_calibration/holdout",
                "sensor_calibration/posterior_check",
            ],
        },
        "provenance": {
            "source": source.model_dump(mode="json"),
            "prior": prior.reference.model_dump(mode="json"),
            "observation_model": observation.reference.model_dump(mode="json"),
            "synthetic_data": _reference(
                data_manifest, kind="sensor-calibration-synthetic-data-v1"
            ).model_dump(mode="json"),
        },
        "limitations": [
            "All values and units are invented and provide no application evidence.",
            "The residual standard deviation is fixed rather than inferred.",
            "The model assumes independent Gaussian readings with a linear response.",
            "The example has no correlated errors, hierarchical prior, or censoring.",
            "The held-out summary is descriptive, not a calibration certificate.",
        ],
    }


def main() -> int:
    """Run the example and emit a complete JSON result or anticipated failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Use a smaller design and posterior-draw diagnostic for a quick check.",
    )
    args = parser.parse_args()
    try:
        result = run_study(check=args.check)
    except CalibrationExampleError as error:
        print(  # noqa: T201 - command-line failure record
            json.dumps({"status": "failed", "reason": str(error)}, indent=2)
        )
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))  # noqa: T201 - CLI result
    return 0


if __name__ == "__main__":
    sys.exit(main())
