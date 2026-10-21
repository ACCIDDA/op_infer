"""Structural model references survive existing containers and guarded consumers."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_infer.posterior import Provenance
from tests.test_forward import identified_provenance
from tests.test_posterior import posterior
from tests.test_provenance import reference

if TYPE_CHECKING:
    from pathlib import Path


def test_native_posterior_retains_structural_identity_without_copying_arrays() -> None:
    """Same arrays and separate complete schema-2 provenance reach a posterior."""
    original = posterior()
    expected = identified_provenance()
    identified = replace(original, provenance=expected)
    assert identified.draws[0].values is original.draws[0].values
    assert identified.provenance.structural_model == expected.structural_model
    assert (
        Provenance.load_json(identified.provenance.model_dump_json(), expected=expected)
        == expected
    )


def test_netcdf_roundtrip_refuses_a_changed_structural_model(tmp_path: Path) -> None:
    """Real snapshot reuse must compare independently expected model structure."""
    pytest.importorskip("arviz_base")
    pytest.importorskip("netCDF4")
    adapter = importlib.import_module("op_infer.adapters.arviz")
    expected = identified_provenance()
    original = replace(posterior(), provenance=expected)
    path = tmp_path / "posterior.nc"
    adapter.to_arviz(original).to_netcdf(path, engine="netcdf4")
    restored = adapter.load_netcdf(path, expected=expected)
    assert restored.provenance.structural_model == expected.structural_model
    np.testing.assert_array_equal(restored.draws[0].values, original.draws[0].values)
    changed = expected.with_structural_model(reference("changed coordinate order"))
    with pytest.raises(ValueError, match="Provenance mismatch"):
        adapter.load_netcdf(path, expected=changed)


def test_sbc_bank_resume_refuses_a_changed_structural_model(tmp_path: Path) -> None:
    """Existing real orchestration checks complete provenance before bank resume."""
    pytest.importorskip("trade_study")
    runner = importlib.import_module("tests.test_sbc_runner")
    expected = identified_provenance()
    assert expected.structural_model is not None
    result = runner._run(tmp_path, [2, 7], provenance=expected)  # noqa: SLF001 - existing analytic synthetic consumer
    assert result.written == 2
    contract = json.loads((tmp_path / "sbc-manifest.meta").read_text())
    assert contract["provenance"][
        "structural_model"
    ] == expected.structural_model.model_dump(mode="json")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    changed = expected.with_structural_model(reference("changed latent observation"))
    with pytest.raises(ValueError, match="mismatch"):
        runner._run(tmp_path, [2, 7], provenance=changed)  # noqa: SLF001
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == before


def test_design_selection_records_keep_model_identity(tmp_path: Path) -> None:
    """Selection/calibration eligibility artifacts retain the same shared reference."""
    pytest.importorskip("trade_study")
    consumer = importlib.import_module("tests.test_design_study")
    expected = identified_provenance()
    assert expected.structural_model is not None
    study = replace(consumer._study(), provenance=expected)  # noqa: SLF001 - existing analytic synthetic consumer
    study.run(directory=tmp_path, replicate_ids=[0], seed=91)
    contracts = [json.loads((tmp_path / "design-manifest.meta").read_text())]
    assert any(
        value.get("provenance", {}).get("structural_model")
        == expected.structural_model.model_dump(mode="json")
        for value in contracts
    )
    changed = replace(
        study,
        provenance=expected.with_structural_model(
            reference("different selection model")
        ),
    )
    with pytest.raises(ValueError, match="mismatch"):
        changed.run(directory=tmp_path, replicate_ids=[0], seed=91)
