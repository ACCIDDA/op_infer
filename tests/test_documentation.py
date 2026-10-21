"""Check the documented analytic SBC workflow without importing test fixtures."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest

from op_infer.rng import RandomKey


@pytest.mark.skipif(
    importlib.util.find_spec("trade_study") is None, reason="requires study extra"
)
def test_normal_example_analytic_posterior_and_resume(tmp_path: Path) -> None:
    """The exact posterior matches its independent moments; the bank resumes."""
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/normal_sbc.py")
    )
    fitted = namespace["fit"](
        np.array([0.5, 1.5, 1.0, 2.0]), RandomKey(seed=71, stream="example_check")
    )
    draws = np.asarray(fitted.draws["theta"])
    assert abs(draws.mean() - 1.0) < 5 * np.sqrt(0.2 / draws.size)
    assert abs(draws.var() - 0.2) < 0.04
    report = namespace["run_example"](tmp_path, replicate_ids=(2, 5, 9, 12))
    assert report["requested"] == report["resumed"] == report["n_total"] == 4
    assert report["failed"] == 0
    assert report["underpowered"] is True
