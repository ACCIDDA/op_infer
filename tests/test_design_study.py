"""Fixed-bank held-out recovery with counted failures and contract replay."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import norm

from op_infer.compare import UnitPredictiveScores, paired_predictive_contrast
from op_infer.design import (
    DesignDefinition,
    DesignFit,
    DesignGate,
    DesignObservation,
    DesignStudy,
)
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from pathlib import Path

    from op_infer.rng import RandomKey

pytest.importorskip("trade_study")


def _study() -> DesignStudy:
    target = reference("held-out normal observations")
    estimator = reference("normal-normal conjugate mean")
    models = ("negative", "positive")

    def generate(model: str, parameters: RandomKey, simulation: RandomKey) -> object:
        assert parameters != simulation
        location = -3.0 if model == "negative" else 3.0
        return location + simulation.numpy_generator().normal(size=22)

    def observe(data: object) -> DesignObservation:
        values = np.asarray(data)
        return DesignObservation(values[:2].copy(), values[2:].copy(), target)

    def fit(model: str, design: str, data: object, key: RandomKey) -> DesignFit:
        del key
        assert design in {"small", "large"}
        values = np.asarray(data)
        assert values.shape == (2,)  # Held-out values never reach fitting.
        mean = -3.0 if model == "negative" else 3.0
        posterior_mean = (100.0 * mean + values.sum()) / 102.0
        return DesignFit(
            posterior=posterior_mean,
            estimator=estimator,
            adequate=True,
            diagnostics={"training_units": len(values)},
        )

    def score(
        model: str, fitted: DesignFit, data: object, key: RandomKey
    ) -> UnitPredictiveScores:
        del model, key
        values = np.asarray(data)
        assert values.shape == (20,)
        return UnitPredictiveScores(
            norm.logpdf(values, loc=fitted.posterior, scale=np.sqrt(1.0 + 1.0 / 102.0)),
            tuple(str(i) for i in range(20)),
            target,
            "independent_unit",
        )

    return DesignStudy(
        models,
        {
            "small": DesignDefinition(observe, 2.0),
            "large": DesignDefinition(observe, 4.0),
        },
        generate,
        fit,
        score,
        {
            (d, m): DesignGate(d, m, estimator, reference(f"gate/{d}/{m}"), "pass")
            for d in ("small", "large")
            for m in models
        },
        provenance(),
    )


def test_paired_contrasts_align_units_and_retain_nonfinite_failures() -> None:
    """The fixed reading rule is symmetric and never drops a failed unit."""
    target = reference("target")
    a = UnitPredictiveScores(
        [4.0, 5.0, 6.0], ("a", "b", "c"), target, "independent_unit"
    )
    b = UnitPredictiveScores(
        [1.0, 1.0, 1.0], ("c", "a", "b"), target, "independent_unit"
    )
    assert paired_predictive_contrast(a, b).outcome == "generator"
    assert paired_predictive_contrast(b, a).outcome == "alternative"
    assert paired_predictive_contrast(a, a).outcome == "indistinguishable"
    failed = replace(b, values=[1.0, np.nan, 1.0])
    result = paired_predictive_contrast(a, failed)
    assert (result.outcome, result.n_units, result.nonfinite_units) == ("failed", 3, 1)
    with pytest.raises(ValueError, match="identical targets"):
        paired_predictive_contrast(a, replace(b, target=reference("other")))
    huge = replace(a, values=[1e308, -1e308, 1e308])
    assert paired_predictive_contrast(huge, b).outcome == "failed"


def test_real_conjugate_refits_and_proper_heldout_scores(tmp_path: Path) -> None:
    """Well-separated normal models recover with one trial per generated cohort."""
    study = _study()
    study.run(tmp_path, replicate_ids=list(range(40)), seed=8)
    table = study.summarize(tmp_path, replicate_ids=list(range(40)), seed=8)
    np.testing.assert_array_equal(table.scores[:, :4], [[1.0, 0.0, 0.0, 0.0]] * 4)
    assert all(m["planned"] == 40 for m in table.metadata)
    interval = table.metadata[0]["wilson"]["generator"]
    assert interval["low"] < 1.0
    assert interval["high"] == 1.0
    assert table.metadata[0]["replicate_unit"] == "independent_generated_cohort"
    assert table.metadata[0]["gate_power_assessed"] is False


def test_shards_order_and_completed_failures_replay(tmp_path: Path) -> None:
    """CRN, task keys and artifact names do not depend on bank/shard ordering."""
    study = _study()
    whole, shards = tmp_path / "whole", tmp_path / "shards"
    study.run(whole, replicate_ids=[7, 1, 2], seed=10, n_jobs=2)
    study.run(shards, replicate_ids=[2], seed=10)
    study.run(shards, replicate_ids=[1, 7], seed=10)
    for path in whole.glob("*.json"):
        assert path.read_bytes() == (shards / path.name).read_bytes()
    before = {p.name: p.read_bytes() for p in shards.glob("*.json")}
    study.run(shards, replicate_ids=[7, 1, 2], seed=10)
    assert before == {p.name: p.read_bytes() for p in shards.glob("*.json")}
    changed = replace(study, threshold_se=3.0)
    with pytest.raises(ValueError, match="mismatch"):
        changed.run(shards, replicate_ids=[1], seed=10)


def test_failed_gates_fits_scores_and_missing_trials_stay_in_denominator(
    tmp_path: Path,
) -> None:
    """No success-only denominator hides unavailable comparison pipelines."""
    study = _study()
    gates = dict(study.gates)
    gates[("small", "positive")] = replace(
        gates[("small", "positive")], label="underpowered pass"
    )
    gated = replace(study, gates=gates)
    gated.run(tmp_path, replicate_ids=[0, 1], seed=11)
    table = gated.summarize(tmp_path, replicate_ids=[0, 1, 2], seed=11)
    np.testing.assert_array_equal(table.scores[:2, 3], [1.0, 1.0])
    assert table.metadata[0]["missing"] == 1
    assert table.metadata[0]["failed"] == 3
    records = [json.loads(p.read_text()) for p in tmp_path.glob("design_small*.json")]
    assert all("underpowered" in r["reason"] for r in records)

    def invalid(
        model: str, fitted: DesignFit, data: object, key: RandomKey
    ) -> UnitPredictiveScores:
        scored = study.score(model, fitted, data, key)
        return replace(scored, values=[np.nan] * 20)

    other = tmp_path / "invalid"
    replace(study, score=invalid).run(other, replicate_ids=[0], seed=1)
    failed = study.summarize(other, replicate_ids=[0], seed=1)
    np.testing.assert_array_equal(failed.scores[:, 3], np.ones(4))
    records = [json.loads(p.read_text()) for p in other.glob("*.json")]
    assert all(
        next(iter(r["contrasts"].values()))["nonfinite_units"] == 20 for r in records
    )


def test_mutated_artifact_and_duplicate_active_worker_are_refused(
    tmp_path: Path,
) -> None:
    """Existing paths do not imply valid provenance or ownership."""
    study = _study()
    study.run(tmp_path, replicate_ids=[0], seed=3)
    path = next(tmp_path.glob("*.json"))
    data = json.loads(path.read_text())
    data["_meta"]["contract"] = "wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        study.run(tmp_path, replicate_ids=[0], seed=3)
    other = tmp_path / "locked"
    other.mkdir()
    (other / "design_small_negative_00000000000000000001.lock").touch()
    with pytest.raises(RuntimeError, match="worker lock"):
        study.run(other, replicate_ids=[1], seed=3)


@pytest.mark.parametrize("inadequate", [False, True])
def test_nonserializable_fit_diagnostics_are_counted_as_failed(
    tmp_path: Path,
    *,
    inadequate: bool,
) -> None:
    """An unavailable diagnostic artifact cannot abort or disappear from a bank."""
    study = _study()

    def bad(model: str, design: str, data: object, key: RandomKey) -> DesignFit:
        return replace(
            study.fit(model, design, data, key),
            diagnostics={} if inadequate else {"bad": np.nan},
            adequate=not inadequate,
        )

    changed = replace(study, fit=bad)
    changed.run(tmp_path, replicate_ids=[0], seed=9)
    table = changed.summarize(tmp_path, replicate_ids=[0], seed=9)
    np.testing.assert_array_equal(table.scores[:, 3], np.ones(4))
