"""Rank-based SBC: PITs, multiplicity, power floor, failures, replicate schema."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_infer.calibrate import (
    aggregate,
    assert_simulated_differs,
    benjamini_hochberg,
    format_table,
    gate,
    load_replicates,
    main,
    pit,
    posterior_contraction,
    write_failed_replicate,
    write_ledger,
    write_replicate,
)

if TYPE_CHECKING:
    from pathlib import Path


def _toy_normal_pits(
    post_var: float, n_rep: int, seed: int, n_draws: int = 1000
) -> list[float]:
    """PITs of theta ~ N(0,1), x ~ N(theta,1), posterior N(x/2, post_var).

    The posterior is exact when post_var = 1/2.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        theta = rng.normal()
        x = rng.normal(theta, 1.0)
        draws = rng.normal(x / 2.0, np.sqrt(post_var), size=n_draws)
        out.append(pit(draws, theta))
    return out


def test_pit_basic_and_atoms() -> None:
    """Ordinary PIT is the mid-rank; the randomized form stays inside the atom."""
    draws = np.array([1.0, 2.0, 2.0, 3.0])
    assert pit(draws, 2.0) == pytest.approx((1 + 0.5 * 2) / 4)
    assert pit(draws, 0.0) == 0.0
    assert pit(draws, 9.0) == 1.0
    with pytest.raises(ValueError, match="Generator"):
        pit(draws, 2.0, atoms=True)
    rng = np.random.default_rng(0)
    values = [pit(draws, 2.0, rng=rng, atoms=True) for _ in range(20)]
    assert all(0.25 <= v <= 0.75 for v in values)
    assert len(set(values)) > 1
    with pytest.raises(ValueError, match="no posterior draws"):
        pit([], 1.0)
    with pytest.raises(ValueError, match="finite"):
        pit([1.0, np.nan], 1.0)
    with pytest.raises(ValueError, match="finite"):
        pit([1.0], np.inf)


def test_randomized_pit_is_reproducible_with_a_seeded_generator() -> None:
    """The same seed gives the same randomized PIT, so a gate is replayable."""
    draws = [0.0, 0.0, 0.0, 1.0]
    a = pit(draws, 0.0, rng=np.random.default_rng(11), atoms=True)
    b = pit(draws, 0.0, rng=np.random.default_rng(11), atoms=True)
    assert a == b


def test_uniform_pits_are_not_flagged() -> None:
    """Uniform PITs pass at the power floor with a clean label."""
    rng = np.random.default_rng(1)
    pits = {f"e{i}": rng.uniform(size=1000) for i in range(7)}
    result = aggregate(pits)
    assert result["n_flagged"] == 0
    assert result["n_replicates"] == 1000
    assert not result["underpowered"]
    assert result["underpowered_banner"] is None
    assert gate(result, max_flagged=1, must_pass=["e0"])["label"] == "pass"


def test_false_flag_rate_under_calibration_is_small() -> None:
    """Across many null experiments the flagged fraction stays near alpha."""
    rng = np.random.default_rng(12)
    flagged = sum(
        aggregate({f"e{i}": rng.uniform(size=300) for i in range(6)})["n_flagged"] > 0
        for _ in range(60)
    )
    assert flagged <= 8  # nominal FWER-like rate 0.05 -> about 3 of 60


def test_calibrated_by_construction_normal_toy() -> None:
    """The exact conjugate posterior passes; an overconfident one is flagged."""
    exact = aggregate({"theta": _toy_normal_pits(0.5, 1000, seed=2)})
    over = aggregate({"theta": _toy_normal_pits(0.25, 1000, seed=3)})
    assert exact["n_flagged"] == 0
    assert over["flagged"] == ["theta"]
    assert over["estimands"]["theta"]["coverage50"] < 0.45
    assert gate(over, max_flagged=0)["label"] == "fail"
    assert gate(exact, max_flagged=0)["label"] == "pass"


def test_biased_posterior_is_flagged_through_the_mean_pit() -> None:
    """A shifted posterior piles PITs toward one end."""
    rng = np.random.default_rng(13)
    out = []
    for _ in range(600):
        theta = rng.normal()
        x = rng.normal(theta, 1.0)
        out.append(pit(rng.normal(x / 2.0 + 0.5, np.sqrt(0.5), size=1000), theta))
    result = aggregate({"theta": out})
    assert result["flagged"] == ["theta"]
    assert result["estimands"]["theta"]["mean_pit"] < 0.45


def test_underpowered_pass_label_and_failure_at_any_n() -> None:
    """A pass at n < floor is labelled underpowered; a failure is a failure at any n."""
    rng = np.random.default_rng(4)
    result = aggregate({"a": rng.uniform(size=50), "b": rng.uniform(size=50)})
    assert result["underpowered"]
    assert "UNDERPOWERED" in result["underpowered_banner"]
    verdict = gate(result, max_flagged=1)
    assert verdict["label"] == "underpowered pass"
    assert verdict["passed"]
    assert verdict["underpowered"]
    biased = aggregate({"a": np.full(50, 0.97)})
    assert gate(biased, max_flagged=0)["label"] == "fail"


def test_power_floor_is_configurable() -> None:
    """The floor moves with ``power_floor`` and the settings record it."""
    rng = np.random.default_rng(14)
    pits = {"a": rng.uniform(size=120)}
    assert aggregate(pits, power_floor=100)["underpowered"] is False
    result = aggregate(pits, power_floor=500)
    assert result["underpowered"]
    assert result["settings"]["power_floor"] == 500


def test_discrete_quantity_needs_randomized_pit() -> None:
    """For K = floor(theta), only the randomized PIT of the exact posterior passes."""
    rng = np.random.default_rng(5)
    n_rep, n_draws = 1000, 1000
    plain, midrank, randomized = [], [], []
    for _ in range(n_rep):
        theta = rng.gamma(2.0, 1.0)
        x = rng.poisson(theta)
        draws = np.floor(rng.gamma(2.0 + x, 1.0 / 2.0, size=n_draws))
        truth = float(np.floor(theta))
        plain.append(float(np.mean(draws < truth)))
        midrank.append(pit(draws, truth))
        randomized.append(pit(draws, truth, rng=rng, atoms=True))
    assert aggregate({"k": plain})["flagged"] == ["k"]
    assert aggregate({"k": midrank})["flagged"] == ["k"]
    assert aggregate({"k": randomized})["n_flagged"] == 0


def test_point_mass_at_zero_needs_randomized_pit() -> None:
    """A zero-inflated quantity is mis-ranked by ordinary, not randomized, PIT."""
    rng = np.random.default_rng(16)
    ordinary, randomized = [], []
    for _ in range(800):
        zero_prob = rng.beta(2.0, 2.0)
        truth = 0.0 if rng.uniform() < zero_prob else float(rng.exponential())
        draws = np.where(
            rng.uniform(size=1000) < zero_prob, 0.0, rng.exponential(size=1000)
        )
        ordinary.append(pit(draws, truth))
        randomized.append(pit(draws, truth, rng=rng, atoms=True))
    assert aggregate({"q": ordinary})["flagged"] == ["q"]
    assert aggregate({"q": randomized})["n_flagged"] == 0


def test_assert_simulated_differs() -> None:
    """Equal cells are counted; a refit of the real data is refused."""
    observed = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[True, True, True], [True, False, True]])
    simulated = np.array([[1, 9, 3], [0, 5, 6]])
    assert assert_simulated_differs(simulated, observed, mask) == 3
    with pytest.raises(ValueError, match="refits the real data"):
        assert_simulated_differs(observed.copy(), observed, mask)
    changed = observed.copy()
    changed[1, 1] = 99  # only an unobserved cell differs
    with pytest.raises(ValueError, match="refits"):
        assert_simulated_differs(changed, observed, mask)
    with pytest.raises(ValueError, match="no observed"):
        assert_simulated_differs(simulated, observed, np.zeros_like(mask))
    with pytest.raises(ValueError, match="shape"):
        assert_simulated_differs(simulated[:1], observed, mask)


def test_benjamini_hochberg_hand_computed() -> None:
    """Hand-computed step-up q-values, including the monotone correction."""
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.02]
    )
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert benjamini_hochberg([0.9, 0.8]) == pytest.approx([0.9, 0.9])
    assert benjamini_hochberg([0.5]) == pytest.approx([0.5])
    assert benjamini_hochberg([0.4, 0.9, 0.0]).tolist() == pytest.approx(
        [0.6, 0.9, 0.0]
    )
    assert benjamini_hochberg([]).size == 0


def test_flag_uses_q_not_raw_p() -> None:
    """One mildly small raw p among many estimands is not flagged after multiplicity."""
    rng = np.random.default_rng(16)
    pits = {f"e{i}": rng.uniform(size=400) for i in range(30)}
    result = aggregate(pits)
    rows = result["estimands"].values()
    assert all(r["q_bh"] >= r["p_chisq"] for r in rows)
    assert all(
        r["flagged"] == (r["q_bh"] <= 0.05) for r in result["estimands"].values()
    )


def test_aggregate_summaries_and_contraction() -> None:
    """Summaries are exact on a hand example; invalid input is refused."""
    pits = {"a": [0.1, 0.3, 0.5, 0.7, 0.99], "b": [0.5, 0.5, 0.5, 0.5, 0.5]}
    result = aggregate(pits, {"a": [0.2, 0.4]})
    a = result["estimands"]["a"]
    assert a["n"] == 5
    assert a["mean_pit"] == pytest.approx(0.518)
    assert a["median_pit"] == pytest.approx(0.5)
    assert a["coverage50"] == pytest.approx(3 / 5)
    assert a["coverage90"] == pytest.approx(4 / 5)
    assert a["mean_contraction"] == pytest.approx(0.3)
    assert result["estimands"]["b"]["mean_contraction"] is None
    with pytest.raises(ValueError, match="not finite values in"):
        aggregate({"a": [0.5, 1.2]})
    with pytest.raises(ValueError, match="no replicates"):
        aggregate({})
    with pytest.raises(ValueError, match="no PITs"):
        aggregate({"a": []})


def test_posterior_contraction() -> None:
    """Contraction is the SD ratio; a posterior equal to the prior gives about one."""
    rng = np.random.default_rng(17)
    assert posterior_contraction(
        rng.normal(0.0, 1.0, size=100_000), 1.0
    ) == pytest.approx(1.0, abs=0.02)
    assert posterior_contraction(
        rng.normal(0.0, 0.25, size=100_000), 1.0
    ) == pytest.approx(0.25, abs=0.01)
    with pytest.raises(ValueError, match="prior_sd"):
        posterior_contraction([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        posterior_contraction([], 1.0)


def test_prior_equal_posterior_is_uniform_but_shows_no_contraction() -> None:
    """Ranks alone cannot see an uninformative fit; the contraction beside them does."""
    rng = np.random.default_rng(18)
    pits, contraction = [], []
    for _ in range(500):
        theta = rng.normal()
        draws = rng.normal(size=500)
        pits.append(pit(draws, theta))
        contraction.append(posterior_contraction(draws, 1.0))
    result = aggregate({"theta": pits}, {"theta": contraction})
    assert result["n_flagged"] == 0
    assert result["estimands"]["theta"]["mean_contraction"] == pytest.approx(
        1.0, abs=0.05
    )


def _write(directory: Path, name: str, payload: dict[str, object]) -> None:
    (directory / name).write_text(json.dumps(payload))


def _make_replicates(directory: Path, n_ok: int, n_failed: int, seed: int = 6) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_ok):
        _write(
            directory,
            f"rep_{i:04d}.json",
            {
                "a": {"pit": float(rng.uniform()), "contraction": 0.5},
                "b": {"pit": float(rng.uniform())},
                "_fit_diag": {"rhat_max": 1.01},
            },
        )
    for j in range(n_failed):
        _write(directory, f"rep_f{j:04d}.json", {"_failed": True})


def test_load_replicates_counts_failures_and_gate(tmp_path: Path) -> None:
    """Failures stay in the denominator; exactly 10% passes, 17% fails."""
    _make_replicates(tmp_path, n_ok=270, n_failed=30)
    pits, contraction, diag = load_replicates(tmp_path)
    assert len(pits["a"]) == 270
    assert len(pits["b"]) == 270
    assert contraction == {"a": [0.5] * 270}
    assert diag["n_failed"] == 30
    assert diag["n_total"] == 300
    assert len(diag["fit_diag"]["rhat_max"]) == 270
    result = aggregate(pits, contraction, n_failed=diag["n_failed"])
    verdict = gate(result, max_flagged=2)
    assert verdict["n_failed"] == 30
    assert verdict["passed"]
    assert verdict["label"] == "pass"

    worse = tmp_path / "worse"
    worse.mkdir()
    _make_replicates(worse, n_ok=250, n_failed=50)
    pits, contraction, diag = load_replicates(worse)
    verdict = gate(
        aggregate(pits, contraction, n_failed=diag["n_failed"]), max_flagged=2
    )
    assert not verdict["passed"]
    assert verdict["label"] == "fail"
    assert any("failed" in r for r in verdict["reasons"])


def test_all_replicates_failed_fails_the_gate(tmp_path: Path) -> None:
    """A directory of only failures aggregates to a failed gate, not an error."""
    _make_replicates(tmp_path, n_ok=0, n_failed=5)
    pits, contraction, diag = load_replicates(tmp_path)
    result = aggregate(pits, contraction, n_failed=diag["n_failed"])
    assert result["failed_fraction"] == 1.0
    assert gate(result, max_flagged=0)["label"] == "fail"


def test_load_replicates_errors(tmp_path: Path) -> None:
    """Missing files, missing estimands, non-finite PITs and non-objects are refused."""
    with pytest.raises(ValueError, match=r"no \*\.json"):
        load_replicates(tmp_path)
    _write(tmp_path, "r0.json", {"a": {"pit": 0.2}, "b": {"pit": 0.3}})
    _write(tmp_path, "r1.json", {"a": {"pit": 0.4}})
    with pytest.raises(ValueError, match="missing estimand 'b'"):
        load_replicates(tmp_path)
    _write(tmp_path, "r1.json", {"a": {"pit": float("nan")}, "b": {"pit": 0.3}})
    with pytest.raises(ValueError, match="non-finite"):
        load_replicates(tmp_path)
    (tmp_path / "r1.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_replicates(tmp_path)


def test_gate_must_pass_and_unknown_names() -> None:
    """A flagged must-pass estimand fails the gate; a misspelled name is an error."""
    result = aggregate({"a": np.full(300, 0.97), "b": np.linspace(0.0, 1.0, 300)})
    assert result["flagged"] == ["a"]
    assert gate(result, max_flagged=1)["passed"]
    verdict = gate(result, max_flagged=1, must_pass=["a"])
    assert not verdict["passed"]
    assert "must-pass" in verdict["reasons"][0]
    with pytest.raises(ValueError, match="not in the result"):
        gate(result, max_flagged=1, must_pass=["nope"])


def test_write_replicate_roundtrip_is_atomic_and_schema_compatible(
    tmp_path: Path,
) -> None:
    """Written replicates load back, leave no scratch file and record failures."""
    write_replicate(
        tmp_path / "sub" / "rep_0.json",
        {"a": 0.25, "b": 0.75},
        contraction={"a": 0.4},
        fit_diag={"rhat_max": 1.002, "ess_min": 900},
    )
    write_replicate(tmp_path / "sub" / "rep_1.json", {"a": 0.5, "b": 0.5})
    write_failed_replicate(tmp_path / "sub" / "rep_2.json", reason="timeout")
    assert sorted(p.name for p in (tmp_path / "sub").iterdir()) == [
        "rep_0.json",
        "rep_1.json",
        "rep_2.json",
    ]
    raw: dict[str, Any] = json.loads((tmp_path / "sub" / "rep_0.json").read_text())
    assert raw["a"] == {"pit": 0.25, "contraction": 0.4}
    assert raw["_fit_diag"]["ess_min"] == 900
    assert json.loads((tmp_path / "sub" / "rep_2.json").read_text()) == {
        "_failed": True,
        "_reason": "timeout",
    }
    pits, contraction, diag = load_replicates(tmp_path / "sub")
    assert pits == {"a": [0.25, 0.5], "b": [0.75, 0.5]}
    assert contraction == {"a": [0.4]}
    assert diag["n_failed"] == 1
    assert diag["fit_diag"] == {"rhat_max": [1.002], "ess_min": [900]}
    with pytest.raises(ValueError, match="not finite values"):
        write_replicate(tmp_path / "bad.json", {"a": float("nan")})
    with pytest.raises(ValueError, match="underscore"):
        write_replicate(tmp_path / "bad.json", {"_a": 0.5})
    assert not (tmp_path / "bad.json").exists()


def test_replicate_schema_preserves_optional_metadata(tmp_path: Path) -> None:
    """A replicate with private metadata keys and optional contraction loads."""
    _write(
        tmp_path,
        "r0.json",
        {
            "a": {"pit": 0.37, "contraction": 0.41},
            "b": {"pit": 0.92},
            "_fit_diag": {"x": 1},
        },
    )
    _write(
        tmp_path,
        "r1.json",
        {"a": {"pit": 0.11}, "b": {"pit": 0.55}, "_note": "kept out of the estimands"},
    )
    pits, contraction, _ = load_replicates(tmp_path)
    assert set(pits) == {"a", "b"}
    assert contraction == {"a": [0.41]}


def test_ledger_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line prints the banner and writes the ledger (exit 0)."""
    reps = tmp_path / "reps"
    reps.mkdir()
    _make_replicates(reps, n_ok=40, n_failed=0)
    out = tmp_path / "out" / "ledger.json"
    status = main(
        [
            "aggregate",
            str(reps),
            "--out",
            str(out),
            "--max-flagged",
            "1",
            "--must-pass",
            "a",
        ]
    )
    assert status == 0
    ledger = json.loads(out.read_text())
    assert ledger["meta"]["script"] == "op_infer.calibrate aggregate"
    assert ledger["meta"]["settings"]["must_pass"] == ["a"]
    assert "timestamp_utc" in ledger["meta"]
    assert "git_sha" in ledger["meta"]
    assert ledger["result"]["gate"]["label"] == "underpowered pass"
    assert ledger["result"]["n_replicates"] == 40
    text = capsys.readouterr().out
    assert "underpowered pass" in text
    assert "UNDERPOWERED" in text
    assert "estimand" in format_table(ledger["result"])

    write_ledger(tmp_path / "x.json", {"k": 1}, meta={"script": "t", "git_sha": "abc"})
    assert json.loads((tmp_path / "x.json").read_text())["meta"]["git_sha"] == "abc"


def test_cli_exit_status_one_on_failure(tmp_path: Path) -> None:
    """A failed gate gives exit status 1 through ``python -m``."""
    reps = tmp_path / "reps"
    reps.mkdir()
    for i in range(60):
        write_replicate(reps / f"r{i}.json", {"a": 0.97})
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    proc = subprocess.run(  # noqa: S603 -- fixed arguments from this test
        [
            sys.executable,
            "-m",
            "op_infer.calibrate",
            "aggregate",
            str(reps),
            "--out",
            str(tmp_path / "l.json"),
            "--max-flagged",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 1, proc.stderr
    assert "gate: fail" in proc.stdout
    assert (
        json.loads((tmp_path / "l.json").read_text())["result"]["gate"]["label"]
        == "fail"
    )


@pytest.mark.parametrize("invalid", [-1, 0.5, 2.0, True, np.nan, np.inf])
def test_impossible_failure_counts_cannot_produce_a_gate(invalid: object) -> None:
    """Negative, fractional and boolean counts cannot enter a rank ledger."""
    ranks = np.tile((np.arange(20) + 0.5) / 20, 50)
    with pytest.raises(ValueError, match="n_failed must be a nonnegative integer"):
        aggregate({"theta": ranks}, n_failed=invalid)  # type: ignore[arg-type]  # invalid count fixture


def test_failure_count_normalization_and_all_failed_bank() -> None:
    """Integer scalar counts retain consistent totals and all-failed verdicts."""
    report = aggregate({"theta": [0.5]}, n_failed=np.int64(3))  # type: ignore[arg-type]  # NumPy integer scalar
    assert report["n_failed"] == 3
    assert type(report["n_failed"]) is int
    assert report["n_total"] == 4
    assert report["failed_fraction"] == 0.75
    failed = aggregate({}, n_failed=4)
    assert failed["n_total"] == failed["n_failed"] == 4
    assert failed["failed_fraction"] == 1
    assert gate(failed, max_flagged=0)["passed"] is False
