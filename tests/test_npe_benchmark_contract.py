"""Pure contract tests for the fresh-process NPE benchmark."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_NAMESPACE = runpy.run_path(
    str(Path(__file__).parents[1] / "examples" / "benchmark_npe_pipeline.py"),
    run_name="npe_benchmark_contract",
)
Case = cast("Any", _NAMESPACE["Case"])
matrix_cases = cast("Any", _NAMESPACE["_matrix_cases"])
summarize = cast("Any", _NAMESPACE["_summarize"])
validate_matrix = cast("Any", _NAMESPACE["_validate_matrix"])
run_matrix = cast("Any", _NAMESPACE["_run_matrix"])


@pytest.mark.parametrize(
    ("device", "storage", "stage", "replicate"),
    [
        ("cuda", "mapped", "train", 0),
        ("cuda:00", "mapped", "train", 0),
        ("cuda:\u0660", "mapped", "train", 0),
        ("cpu", "unknown", "train", 0),
        ("cpu", "mapped", "unknown", 0),
        ("cpu", "mapped", "train", -1),
        ("cuda:0", "mapped", "checkpoint_full", 0),
    ],
)
def test_case_refuses_ambiguous_or_unsupported_conditions(
    device: str, storage: str, stage: str, replicate: int
) -> None:
    """A result identity cannot depend on ambient device or invalid labels."""
    with pytest.raises(ValueError, match=r"Unsupported|checkpoint"):
        Case(device, storage, stage, replicate)


def test_matrix_is_serial_complete_and_rotates_train_order() -> None:
    """Every repeat has six fits and nine CPU checkpoint workers."""
    cases = matrix_cases(("cpu", "cuda:0"), 2)
    assert len(cases) == 30
    for replicate in range(2):
        selected = [case for case in cases if case.replicate == replicate]
        assert sum(case.stage == "train" for case in selected) == 6
        assert sum(case.stage != "train" for case in selected) == 9
        assert all(case.device == "cpu" for case in selected if case.stage != "train")
    assert cases[0].slug != cases[15].slug


def _result(
    device: str, storage: str, fit: float, epochs: list[float]
) -> dict[str, Any]:
    return {
        "case": {"device": device, "storage": storage, "stage": "train"},
        "phases": {"fit": {"seconds": fit}},
        "training_trace": {
            "train_epochs": [{"seconds": seconds} for seconds in epochs]
        },
    }


def test_summary_keeps_conditions_separate_and_descriptive() -> None:
    """Aggregation cannot manufacture a cross-device speedup claim."""
    summary = summarize(
        (
            _result("cpu", "mapped", 4.0, [2.0, 1.0]),
            _result("cpu", "mapped", 6.0, [3.0, 2.0]),
            _result("cuda:0", "mapped", 3.0, [1.5, 1.0]),
        )
    )
    conditions = summary["conditions"]
    assert len(conditions) == 2
    cpu = next(row for row in conditions if row["device"] == "cpu")
    assert cpu["instrumented_fit_seconds_median"] == 5.0
    assert cpu["instrumented_fit_seconds_range"] == [4.0, 6.0]
    assert all(row["descriptive_only"] is True for row in conditions)
    assert not any("speedup" in row for row in conditions)


def _validation_result(storage: str, stage: str) -> dict[str, Any]:
    completed = 2 if stage == "checkpoint_chunk" else 4
    state = "chunk" if stage == "checkpoint_chunk" else "final"
    checkpoint = (
        {"equivalence_digests": {"state": "same"}}
        if stage.startswith("checkpoint")
        else None
    )
    return {
        "case": {
            "device": "cpu",
            "storage": storage,
            "stage": stage,
            "replicate": 0,
        },
        "completed_epochs": completed,
        "fit_state_digest": f"state-{state}",
        "source_identity": {
            "reference": {"kind": "source-manifest-v1", "digest": "sha256:same"}
        },
        "checkpoint": checkpoint,
        "training_trace": {
            "split_digests": {"train": "train", "validation": "validation"},
            "train_epochs": [
                {
                    "rng_before": {"cpu": f"before-{epoch}"},
                    "rng_after": {"cpu": f"after-{epoch}"},
                }
                for epoch in range(completed)
            ],
        },
    }


def _validation_results() -> list[dict[str, Any]]:
    return [
        _validation_result(storage, stage)
        for storage in ("ordinary", "borrowed", "mapped")
        for stage in (
            "train",
            "checkpoint_full",
            "checkpoint_chunk",
            "checkpoint_resume",
        )
    ]


def test_matrix_validation_requires_fixed_work_and_storage_equivalence() -> None:
    """Same-device storage paths retain the same split, RNG trace and state."""
    results = _validation_results()
    validation = validate_matrix(results, repetitions=1, completed_epochs=4)
    assert validation["completed_epoch_counts"] == "verified for every worker"
    results[-1]["fit_state_digest"] = "changed"
    with pytest.raises(RuntimeError, match="Storage paths changed"):
        validate_matrix(results, repetitions=1, completed_epochs=4)


def test_matrix_validation_requires_one_split_and_full_budget_state() -> None:
    """Stages cannot silently change the split or full-budget CPU result."""
    changed_split = _validation_results()
    changed_split[1]["training_trace"]["split_digests"] = {"train": "changed"}
    with pytest.raises(RuntimeError, match="different native splits"):
        validate_matrix(changed_split, repetitions=1, completed_epochs=4)

    changed_state = _validation_results()
    for row in changed_state:
        if row["case"]["stage"] in {"checkpoint_full", "checkpoint_resume"}:
            row["fit_state_digest"] = "different-full-budget-state"
    with pytest.raises(RuntimeError, match="full-budget train"):
        validate_matrix(changed_state, repetitions=1, completed_epochs=4)


def test_matrix_validation_refuses_mixed_source_identities() -> None:
    """A multi-worker summary cannot combine source revisions."""
    results = _validation_results()
    results[-1]["source_identity"]["reference"]["digest"] = "sha256:changed"
    with pytest.raises(RuntimeError, match="different source identities"):
        validate_matrix(results, repetitions=1, completed_epochs=4)


def test_matrix_refuses_existing_execution_output(tmp_path: Path) -> None:
    """A retry cannot silently overwrite or combine prior evidence."""
    (tmp_path / "results").mkdir()
    args = SimpleNamespace(
        directory=tmp_path,
        rows=8,
        units=2,
        features=1,
        max_epochs=3,
        devices="cpu",
        repetitions=1,
        cpus="0",
        monitor_cpu=1,
    )
    with pytest.raises(FileExistsError, match="new prepared directory"):
        run_matrix(args)


def test_matrix_requires_distinct_compute_and_monitor_cpus(tmp_path: Path) -> None:
    """The retained summary cannot claim an undeclared monitoring placement."""
    common = {
        "directory": tmp_path,
        "rows": 8,
        "units": 2,
        "features": 1,
        "max_epochs": 3,
        "devices": "cpu",
        "repetitions": 1,
    }
    for cpus, monitor_cpu in ((None, None), ("7", None), ("7", 7)):
        args = SimpleNamespace(**common, cpus=cpus, monitor_cpu=monitor_cpu)
        with pytest.raises(ValueError, match="distinct compute and monitor"):
            run_matrix(args)
