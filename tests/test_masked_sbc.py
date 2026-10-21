"""Masked conjugate cohorts, strata, baseline failures and guarded fixed banks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.masked_sbc import (
    MaskedSBCPlan,
    MaskedUnit,
    run_masked_sbc,
    summarize_masked_sbc,
)
from op_infer.posterior import ArtifactReference
from op_infer.sbc import SBCEstimand, SBCFit
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

    from op_infer.rng import RandomKey

pytest.importorskip("trade_study")
SIGMA = np.array([[1.0], [1.0], [2.0], [2.0]])


def plan(n: int = 16, *, stratum: str | None = None) -> MaskedSBCPlan:
    """Declare four ordered units and a fixed complete ID bank."""
    return MaskedSBCPlan(
        units=tuple(
            MaskedUnit(name=f"u{i}", stratum="A" if i < 2 else "B") for i in range(4)
        ),
        replicate_ids=tuple(range(n)),
        seed=831,
        stratum=stratum,
    )


def prior(key: RandomKey) -> Mapping[str, ArrayLike]:
    """Use the same exact Normal prior for full and masked cohorts."""
    return {"theta": key.numpy_generator().normal()}


def simulate(truth: Mapping[str, ArrayLike], key: RandomKey) -> ArrayLike:
    """Simulate the full cohort before choosing which unit is unobserved."""
    return np.asarray(
        float(np.asarray(truth["theta"]))
        + key.numpy_generator().normal(size=(4, 1)) * SIGMA,
        dtype=np.float64,
    )


def fit(data: ArrayLike, mask: NDArray[np.bool_], key: RandomKey) -> SBCFit:
    """Use only declared observed cells in the analytic conjugate posterior."""
    values = np.asarray(data)
    precision = 1 + np.sum(mask / SIGMA**2)
    variance = 1 / precision
    mean = variance * np.sum(np.where(mask, values / SIGMA**2, 0))
    draws = key.numpy_generator().normal(mean, np.sqrt(variance), size=512)
    return SBCFit({"theta": draws}, {"key": key.model_dump(mode="json")})


def estimands() -> dict[str, SBCEstimand]:
    """Include a reported coordinate with a genuine point mass at zero."""
    return {
        "theta": SBCEstimand(lambda p: p["theta"], prior_sd=1.0),
        "positive": SBCEstimand(
            lambda p: np.maximum(p["theta"], 0),
            prior_sd=float(np.sqrt(0.5 - 1 / (2 * np.pi))),
            atoms=True,
        ),
    }


def run(directory: Path, **kwargs: Any) -> Any:  # noqa: ANN401 - invalid-argument fixtures too
    """Run the public API with a compatible independently identified baseline."""
    use_baseline = kwargs.pop("baseline", fit)
    baseline_identity = kwargs.pop(
        "baseline_provenance",
        None
        if use_baseline is None
        else provenance().model_copy(update={"estimator_stage": "comparison"}),
    )
    return run_masked_sbc(
        kwargs.pop("prior_sampler", prior),
        kwargs.pop("simulator", simulate),
        kwargs.pop("estimator", fit),
        plan=kwargs.pop("plan", plan()),
        estimands=estimands(),
        observed=kwargs.pop("observed", np.full((4, 1), 1000.0)),
        directory=directory,
        provenance=provenance(),
        baseline=use_baseline,
        baseline_provenance=baseline_identity,
        **kwargs,
    )


def test_calibrated_masked_bank_and_stratum_informativeness(tmp_path: Path) -> None:
    """Conjugate masks rank both continuous and atom-aware quantities by stratum."""
    result = run(tmp_path, plan=plan(1024))
    assert result.failed == 0
    summary = summarize_masked_sbc(tmp_path, expected=result.contract)
    assert summary["overall"]["n_planned"] == summary["overall"]["n_successful"] == 1024
    assert summary["overall"]["calibration_available"]
    assert summary["adequacy_status"] == "not_assessed"
    for name in ("A", "B"):
        group = summary["by_stratum"][name]
        assert group["n_planned"] == group["n_successful"] == 512
        assert group["n_failed"] == group["n_missing"] == 0
        for row in group["rank_summary"]["estimands"].values():
            assert abs(row["mean_pit"] - 0.5) < 0.05
        widths = group["informativeness"]["theta"]
        expected_sd = np.sqrt(0.4 if name == "A" else 1 / 3.25)
        assert widths["prior_sd_ratio"]["mean"] == pytest.approx(expected_sd, rel=0.04)
        assert 0.9 < widths["baseline_sd_ratio"]["mean"] < 1.1
        assert widths["n_baseline_available"] == 512
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_shard_order_parallel_resume_and_independent_baseline_stream(
    tmp_path: Path,
) -> None:
    """Per-ID records are byte-identical after changed ordering and worker layout."""
    whole, shards = tmp_path / "whole", tmp_path / "shards"
    complete = run(whole, n_jobs=2)
    first = run(shards, replicate_ids=(7, 3, 8, 0))
    incomplete = summarize_masked_sbc(shards, expected=first.contract)
    assert incomplete["overall"]["n_missing"] == 12
    assert incomplete["overall"]["rank_summary"] is None
    run(shards, replicate_ids=tuple(i for i in range(16) if i not in (7, 3, 8, 0)))
    for path in whole.glob("unit_*/rep_*.json"):
        assert path.read_bytes() == (shards / path.relative_to(whole)).read_bytes()
    assert complete.contract == first.contract
    resumed = run(shards, replicate_ids=(3, 0))
    assert resumed.written == 0
    assert resumed.resumed == 2
    record = json.loads(next(whole.glob("unit_*/rep_*.json")).read_text())
    assert record["_fit_diag"]["key"]["stream"] == "estimator.reporting"
    assert record["_fit_diag"]["masked_sbc"]["baseline_status"] == "available"
    baseline = record["_fit_diag"]["masked_sbc"]["baseline_diagnostics"]
    assert baseline["diagnostics"]["key"]["stream"] == "estimator.baseline"


@pytest.mark.parametrize("mode", ["exception", "inadequate", "diagnostics"])
def test_baseline_failures_do_not_delete_reporting_calibration(
    tmp_path: Path, mode: str
) -> None:
    """Unavailable comparison widths keep the full planned denominator."""

    def broken(_data: ArrayLike, _mask: NDArray[np.bool_], key: RandomKey) -> SBCFit:
        assert key.stream == "estimator.baseline"
        if mode == "inadequate":
            return SBCFit(
                {"theta": np.zeros(512)},
                {},
                kind="mcmc",
                adequate=False,
                effective_sample_size=12.0,
            )
        if mode == "diagnostics":
            return SBCFit({"theta": np.zeros(512)}, {"not_serializable": object()})
        msg = "comparison failure"
        raise RuntimeError(msg)

    result = run(tmp_path, baseline=broken)
    assert result.failed == 0
    summary = summarize_masked_sbc(tmp_path, expected=result.contract)
    widths = summary["overall"]["informativeness"]["theta"]
    assert widths["n_available"] == 16
    assert widths["n_baseline_available"] == 0
    assert widths["n_baseline_unavailable"] == 16
    assert widths["baseline_sd_ratio"] is None
    assert summary["replicates"][0]["informativeness"]["baseline_reason"]


def test_reporting_failure_inadequate_mcmc_and_observed_copy_are_counted(
    tmp_path: Path,
) -> None:
    """Masking cannot turn an observed-data copy into simulated evidence."""

    def inadequate(data: ArrayLike, mask: NDArray[np.bool_], key: RandomKey) -> SBCFit:
        result = fit(data, mask, key)
        return SBCFit(
            result.draws, {}, kind="mcmc", adequate=False, effective_sample_size=12.0
        )

    result = run(tmp_path / "inadequate", estimator=inadequate)
    assert result.failed == 16
    summary = summarize_masked_sbc(tmp_path / "inadequate", expected=result.contract)
    assert summary["overall"]["n_failed"] == 16
    assert not summary["overall"]["calibration_available"]
    assert summary["overall"]["informativeness"]["theta"]["n_unavailable"] == 16

    def copied_observations(
        _truth: Mapping[str, ArrayLike], key: RandomKey
    ) -> ArrayLike:
        values = np.full((4, 1), 1000.0)
        values[key.index % 4] += 1  # Only the removed unit differs from real data.
        return values

    copied = run(tmp_path / "copy", simulator=copied_observations)
    assert copied.failed == 16


def test_zero_baseline_width_retains_difference_and_undefined_ratio(
    tmp_path: Path,
) -> None:
    """Zero-width comparisons are visible without fabricating a finite ratio."""

    def constant(_data: ArrayLike, _mask: NDArray[np.bool_], _key: RandomKey) -> SBCFit:
        return SBCFit({"theta": np.zeros(512)}, {})

    result = run(tmp_path, baseline=constant)
    summary = summarize_masked_sbc(tmp_path, expected=result.contract)
    row = summary["overall"]["informativeness"]["theta"]
    assert row["n_baseline_available"] == row["n_baseline_zero_sd"] == 16
    assert row["n_baseline_ratio_available"] == 0
    assert row["baseline_sd_ratio"] is None
    assert row["sd_difference"]["min"] > 0


def test_stratum_restriction_and_zero_plan_strata_stay_visible(tmp_path: Path) -> None:
    """Overall summaries do not drop untested strata or fabricate their results."""
    result = run(tmp_path, plan=plan(stratum="B"), baseline=None)
    summary = summarize_masked_sbc(tmp_path, expected=result.contract)
    assert summary["by_stratum"]["A"]["n_planned"] == 0
    assert summary["by_stratum"]["A"]["status"] == "no_planned_replicates"
    assert summary["by_stratum"]["A"]["rank_summary"] is None
    assert summary["by_stratum"]["B"]["n_successful"] == 16
    assert (
        summary["replicates"][0]["informativeness"]["baseline_status"]
        == "not_requested"
    )


def test_contract_and_record_tampering_are_refused(tmp_path: Path) -> None:
    """All planned counts are guarded before rank or informativeness reuse."""
    result = run(tmp_path)
    with pytest.raises(ValueError, match="independent expected"):
        summarize_masked_sbc(tmp_path, expected=reference("wrong"))
    with pytest.raises(ValueError, match="mismatch"):
        run(tmp_path, plan=plan(17))
    path = next(tmp_path.glob("unit_*/rep_*.json"))
    original = path.read_text()
    record = json.loads(original)
    record["_fit_diag"]["masked_sbc"]["unit_index"] = 42
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="width metadata"):
        summarize_masked_sbc(tmp_path, expected=result.contract)
    record = json.loads(original)
    record["theta"]["contraction"] *= 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contraction disagrees"):
        summarize_masked_sbc(tmp_path, expected=result.contract)
    path.write_text(original)
    path.unlink()
    partial = summarize_masked_sbc(tmp_path, expected=result.contract)
    assert partial["overall"]["n_missing"] == 1
    assert partial["overall"]["rank_summary"] is None


def test_bad_masks_baselines_and_plan_inputs_fail_before_execution(
    tmp_path: Path,
) -> None:
    """Invalid scientific contracts cannot establish a plausible-looking bank."""
    with pytest.raises(ValueError, match="both its callback"):
        run(tmp_path / "no_baseline_context", baseline_provenance=None)
    with pytest.raises(ValueError, match="share prior"):
        run(
            tmp_path / "wrong_baseline",
            baseline_provenance=provenance().model_copy(
                update={"prior": reference("other")}
            ),
        )
    with pytest.raises(ValueError, match="two observed units"):
        run(tmp_path / "mask", observed_mask=np.array([[1], [0], [0], [0]], dtype=bool))
    with pytest.raises(ValueError, match="prespecified bank"):
        run(tmp_path / "IDs", replicate_ids=(99,))
    with pytest.raises(ValidationError):
        MaskedSBCPlan(units=plan().units, replicate_ids=(True,), seed=1)
    # Expected artifact references must still carry their own semantic kind.
    result = run(tmp_path / "kind")
    wrong_kind = ArtifactReference(kind="other", digest=result.contract.digest)
    with pytest.raises(ValueError, match="contract differs"):
        summarize_masked_sbc(tmp_path / "kind", expected=wrong_kind)
