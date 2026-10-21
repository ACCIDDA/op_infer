"""Invalid rank/gate controls cannot bypass failed calibration evidence."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_infer import calibrate
from op_infer.calibrate import aggregate, gate, main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "settings",
    [
        {"bins": 1},
        {"bins": 0},
        {"bins": -1},
        {"bins": True},
        {"bins": 2.5},
        {"power_floor": 0},
        {"power_floor": -1},
        {"power_floor": True},
        {"power_floor": 1.5},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {"alpha": -0.1},
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": True},
        {"alpha": np.bool_(0)},
    ],
)
def test_invalid_settings_cannot_suppress_a_flag(settings: dict[str, Any]) -> None:
    """The formerly false-passing one-bin/NaN cases fail before statistics run."""
    pits = {"biased": np.zeros(1000)}
    assert gate(aggregate(pits), max_flagged=0)["label"] == "fail"
    with pytest.raises(ValueError, match=next(iter(settings))):
        aggregate(pits, **settings)
    # Validate controls even when there are no successful fits.
    with pytest.raises(ValueError, match=next(iter(settings))):
        aggregate({}, n_failed=3, **settings)


@pytest.mark.parametrize(
    "controls",
    [
        {"max_flagged": -1},
        {"max_flagged": True},
        {"max_flagged": 0.5},
        {"max_flagged": float("nan")},
        {"max_flagged": np.bool_(0)},
        {"max_failed_fraction": float("nan")},
        {"max_failed_fraction": float("inf")},
        {"max_failed_fraction": -0.1},
        {"max_failed_fraction": 1.1},
        {"max_failed_fraction": True},
        {"max_failed_fraction": np.bool_(0)},
    ],
)
def test_invalid_gate_controls_cannot_bypass_evidence(controls: dict[str, Any]) -> None:
    """NaN comparisons cannot erase flagged estimands or an excessive failure rate."""
    result = aggregate({"biased": np.zeros(1000)}, n_failed=1000)
    arguments = {"max_flagged": 0, **controls}
    with pytest.raises(ValueError, match=next(iter(controls))):
        gate(result, **arguments)


def test_gate_refuses_invalid_settings_on_saved_older_reports() -> None:
    """Reading an older aggregate must not revive its unsafe configuration."""
    original = aggregate({"theta": np.tile((np.arange(20) + 0.5) / 20, 50)})
    for updates in ({"bins": 1}, {"alpha": float("nan")}, {"power_floor": -1}):
        result = copy.deepcopy(original)
        result["settings"].update(updates)
        with pytest.raises(ValueError, match=next(iter(updates))):
            gate(result, max_flagged=0)
    del original["settings"]
    with pytest.raises(ValueError, match="requires the settings"):
        gate(original, max_flagged=0)


def test_valid_numpy_settings_normalize_without_changing_results() -> None:
    """Accepted scalar counts produce portable Python settings and the same gate."""
    pits = {"theta": np.tile((np.arange(20) + 0.5) / 20, 50)}
    settings: dict[str, Any] = {
        "bins": np.int64(20),
        "power_floor": np.int64(200),
        "alpha": np.float64(0.05),
    }
    result = aggregate(pits, **settings)
    assert result == aggregate(pits)
    assert all(type(result["settings"][key]) is int for key in ("bins", "power_floor"))
    assert type(result["settings"]["alpha"]) is float
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    controls: dict[str, Any] = {
        "max_flagged": np.int64(0),
        "max_failed_fraction": np.float64(0.1),
    }
    assert gate(result, **controls) == gate(result, max_flagged=0)


def test_failure_fraction_endpoints_and_inclusive_limit_are_preserved() -> None:
    """Zero and one remain valid failure policies; exact bounds stay inclusive."""
    result = aggregate({"theta": np.tile((np.arange(20) + 0.5) / 20, 50)}, n_failed=1)
    assert not gate(result, max_flagged=0, max_failed_fraction=0)["passed"]
    assert gate(result, max_flagged=0, max_failed_fraction=1)["passed"]
    assert gate(result, max_flagged=0, max_failed_fraction=result["failed_fraction"])[
        "passed"
    ]


def test_nonfinite_reference_statistic_cannot_become_an_unflagged_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undefined reference test stops aggregation rather than disappearing."""
    monkeypatch.setattr(
        calibrate, "chisquare", lambda *_a, **_kw: SimpleNamespace(pvalue=np.nan)
    )
    with pytest.raises(ValueError, match="nonfinite p-values"):
        aggregate({"theta": np.linspace(0, 1, 1000)})


@pytest.mark.parametrize(
    "arguments",
    [
        ["--bins", "1"],
        ["--alpha", "nan"],
        ["--power-floor", "-1"],
        ["--max-flagged", "-1"],
    ],
)
def test_cli_cannot_write_a_passing_ledger_with_invalid_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    """Invalid CLI controls abort before a label or output ledger is produced."""
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "rep.json").write_text('{"theta": {"pit": 0.1}}')
    output = tmp_path / "ledger.json"
    with pytest.raises(ValueError, match=arguments[0][2:].replace("-", "_")):
        main(["aggregate", str(bank), "--out", str(output), *arguments])
    assert not output.exists()
    assert "gate:" not in capsys.readouterr().out
