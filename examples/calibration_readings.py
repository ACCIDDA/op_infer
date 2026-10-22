"""Enumerate public toy reading operating characteristics, with seeded checks.

Run with an empty output directory. The policy is saved before enumerating any
outcomes. These toy choices are not recommended application thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom

from op_infer.posterior import ArtifactReference
from op_infer.readings import (
    CalibrationEvidence,
    CalibrationPolicy,
    CalibrationRule,
    wilson_coverage,
    wilson_method,
)


def run(directory: Path) -> dict[str, object]:
    """Save fixed criteria, enumerate count outcomes and independently simulate.

    Returns:
        Exact finite-binomial and seeded Monte Carlo failure probabilities.
    """
    context = ArtifactReference.from_bytes(
        b"iid-Bernoulli-public-toy-v1", kind="toy-design-v1"
    )
    trials = 80
    policy = CalibrationPolicy(
        name="public-binomial-toy",
        context=context,
        governing="interval",
        rationale="Illustrate compatibility and finite-sample resolution",
        replicate_ids=tuple(str(i) for i in range(trials)),
        minimum_usable=trials,
        max_failure_fraction=0.0,
        max_failed_quantities=0,
        must_pass=("quantity",),
        rules=(
            CalibrationRule(
                name="coverage",
                quantity="quantity",
                metric="coverage",
                method=wilson_method(confidence=0.95),
                lower=0.4,
                upper=0.6,
            ),
        ),
    )
    path = directory / "policy.json"
    policy.write(path)
    failed = np.zeros((trials + 1, 2), dtype=bool)
    for count in range(trials + 1):
        evidence = CalibrationEvidence(
            context=context,
            source=ArtifactReference.from_bytes(
                f"toy-success-count:{count}".encode(), kind="toy-count-v1"
            ),
            usable=policy.replicate_ids,
            failures=(),
            metrics=(
                wilson_coverage(
                    quantity="quantity",
                    metric="coverage",
                    successes=count,
                    trials=trials,
                    confidence=0.95,
                ),
            ),
        )
        record = policy.evaluate(path, evidence)
        failed[count] = [r.state == "fail" for r in record.readings()]
        if count == 51:
            (directory / "example-record.json").write_text(
                record.model_dump_json(indent=2) + "\n"
            )
    rng = np.random.default_rng(82397)
    scenarios = []
    for name, probability in (("calibrated-toy", 0.5), ("misspecified-toy", 0.8)):
        exact = binom.pmf(np.arange(trials + 1), trials, probability) @ failed
        monte_carlo = failed[rng.binomial(trials, probability, size=20_000)].mean(
            axis=0
        )
        np.testing.assert_allclose(monte_carlo, exact, atol=0.004)
        scenarios.append(
            {
                "name": name,
                "success_probability": probability,
                "exact_failure_probability": exact.tolist(),
                "seeded_failure_frequency": monte_carlo.tolist(),
            }
        )
    result = {
        "schema_version": 1,
        "scope": "independent finite-binomial public toy; no application defaults",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "policy": policy.model_dump(mode="json"),
        "panel_order": ["point", "interval"],
        "scenarios": scenarios,
        "monte_carlo": {"seed": 82397, "cohorts_per_scenario": 20_000},
    }
    (directory / "operating-characteristics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory), indent=2))  # noqa: T201 - CLI result
