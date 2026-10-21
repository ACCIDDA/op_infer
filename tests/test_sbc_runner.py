"""SBC banks keep reproducible identities, calibrated ranks and failed attempts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_infer.calibrate import aggregate, load_replicates
from op_infer.sbc import SBCEstimand, SBCFit, run_sbc
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import ArrayLike

    from op_infer.rng import RandomKey

pytest.importorskip("trade_study")


def _prior(key: RandomKey) -> Mapping[str, ArrayLike]:
    return {"theta": key.numpy_generator().normal()}


def _simulate(truth: Mapping[str, ArrayLike], key: RandomKey) -> ArrayLike:
    return np.array(
        [float(np.asarray(truth["theta"])) + key.numpy_generator().normal()]
    )


def _fit(data: ArrayLike, key: RandomKey) -> SBCFit:
    draws = key.numpy_generator().normal(
        float(np.asarray(data)[0]) / 2.0, np.sqrt(0.5), 1000
    )
    return SBCFit({"theta": draws}, {"adapter": "analytic-normal"})


def _estimands() -> dict[str, SBCEstimand]:
    return {"theta": SBCEstimand(lambda p: p["theta"], prior_sd=1.0)}


def _run(directory: Path, ids: list[int], **kwargs: Any) -> Any:  # noqa: ANN401
    return run_sbc(
        _prior,
        _simulate,
        kwargs.pop("estimator", _fit),
        estimands=_estimands(),
        observed=np.array([1000.0]),
        directory=directory,
        replicate_ids=ids,
        seed=71,
        provenance=kwargs.pop("provenance", provenance()),
        **kwargs,
    )


def test_exact_conjugate_bank_is_calibrated_and_contract_is_retained(
    tmp_path: Path,
) -> None:
    """An exact posterior passes a powered bank, with contraction beside ranks."""
    result = _run(tmp_path, list(range(1000)))
    assert (result.requested, result.written, result.resumed, result.failed) == (
        1000,
        1000,
        0,
        0,
    )
    pits, contraction, diagnostics = load_replicates(tmp_path)
    report = aggregate(pits, contraction, n_failed=diagnostics["n_failed"])
    assert report["n_flagged"] == 0
    assert not report["underpowered"]
    assert 0.68 < np.mean(contraction["theta"]) < 0.74
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["_meta"]["contract"].startswith("sha256:")
    assert record["_fit_diag"]["draws_used"] == 1000


def test_order_shards_and_resume_replay_identical_files(tmp_path: Path) -> None:
    """Stable IDs rather than grid positions identify every random stream."""
    whole, sharded = tmp_path / "whole", tmp_path / "shards"
    _run(whole, [4, 1, 9, 2], n_jobs=2)
    _run(sharded, [9, 4])
    _run(sharded, [2, 1])
    for path in whole.glob("*.json"):
        assert path.read_bytes() == (sharded / path.name).read_bytes()
    resumed = _run(sharded, [1, 9])
    assert (resumed.written, resumed.resumed) == (0, 2)
    changed = provenance().model_copy(update={"design": reference("different design")})
    with pytest.raises(ValueError, match="mismatch"):
        _run(sharded, [1], provenance=changed)


def test_fit_failures_and_mcmc_inadequacy_remain_in_denominator(tmp_path: Path) -> None:
    """Invalid fits become failed records; resume cannot silently retry them."""

    def invalid(_data: ArrayLike, _key: RandomKey) -> SBCFit:
        return SBCFit(
            {"theta": np.arange(100.0)},
            {"rhat": 1.8},
            kind="mcmc",
            adequate=False,
            effective_sample_size=10.0,
        )

    result = _run(tmp_path, [0, 1, 2], estimator=invalid)
    assert result.failed == 3
    _pits, _contraction, diag = load_replicates(tmp_path)
    assert (diag["n_failed"], diag["n_total"]) == (3, 3)
    resumed = _run(tmp_path, [0, 1, 2], estimator=invalid)
    assert (resumed.failed, resumed.resumed) == (3, 3)


def test_observed_copy_is_rejected_before_estimator_runs(tmp_path: Path) -> None:
    """A simulator returning the actual observations must never produce ranks."""

    def never(_data: ArrayLike, _key: RandomKey) -> SBCFit:
        pytest.fail("Observed copy reached fitting")

    result = run_sbc(
        _prior,
        lambda _p, _k: np.array([3.0]),
        never,
        estimands=_estimands(),
        observed=np.array([3.0]),
        directory=tmp_path,
        replicate_ids=[0],
        seed=1,
        provenance=provenance(),
    )
    assert result.failed == 1
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "equal the real data" in record["_reason"]


def test_mcmc_ess_thinning_is_recorded(tmp_path: Path) -> None:
    """The runner preserves the adapter's verdict and reports the used count."""

    def adequate(_data: ArrayLike, key: RandomKey) -> SBCFit:
        return SBCFit(
            {"theta": key.numpy_generator().normal(size=100)},
            {"rhat": 1.001},
            kind="mcmc",
            adequate=True,
            effective_sample_size=12.8,
        )

    result = _run(tmp_path, [3], estimator=adequate)
    assert result.failed == 0
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["_fit_diag"]["draws_used"] == 12
    assert record["_fit_diag"]["rhat"] == 1.001
