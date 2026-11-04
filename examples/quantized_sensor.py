"""Compare quantized-sensor draws with independent Normal interval probabilities.

Run from a checkout with ``uv run python examples/quantized_sensor.py``.
All inputs are synthetic instrument coordinates; bin codes are labels rather
than replacement measurements at interval midpoints.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from op_infer.observe import BinnedNormalObservation
from op_infer.rng import RandomKey


def run_example() -> dict[str, object]:
    """Retain the declared law and compare all four bin frequencies."""
    sensor = BinnedNormalObservation(cutpoints=(-0.5, 0.25, 1.0), scale=0.7)
    latent = np.array(0.3)
    key = RandomKey(seed=2026, stream="quantized-sensor")
    codes = sensor.sample(latent, key, sample_shape=(30000,))
    expected = np.diff(
        stats.norm.cdf([-np.inf, *sensor.cutpoints, np.inf], loc=0.3, scale=0.7)
    )
    predicted = np.exp(sensor.log_prob(latent, np.arange(4)))
    empirical = np.array([(codes == code).mean() for code in range(4)])
    np.testing.assert_allclose(predicted, expected, rtol=1e-12)
    np.testing.assert_allclose(empirical, expected, atol=0.01)
    np.testing.assert_array_equal(
        codes, sensor.sample(latent, key, sample_shape=(30000,))
    )
    return {
        "law": sensor.model_dump(mode="json"),
        "reference": sensor.reference.model_dump(mode="json"),
        "key": key.model_dump(mode="json"),
        "draws": len(codes),
        "expected_probability": expected.tolist(),
        "empirical_frequency": empirical.tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))  # noqa: T201
