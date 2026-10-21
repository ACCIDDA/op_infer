"""Native run identity and preflight failures, available without NumPyro/JAX."""

from __future__ import annotations

import importlib.metadata
import json

import pytest
from pydantic import ValidationError

from op_infer.adapters import numpyro as adapter
from op_infer.adapters.numpyro import NutsSettings, SiteLayout, fit_nuts
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    Coordinate,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.rng import RandomKey
from tests.test_provenance import reference


def settings(**updates: object) -> NutsSettings:
    """Validate a bounded scalar fit configuration from explicit Python metadata."""
    return NutsSettings.model_validate(
        {
            "key": RandomKey(seed=2026, stream="fit"),
            "sites": (SiteLayout(name="theta"),),
            "num_warmup": 100,
            "num_samples": 512,
            "num_chains": 2,
            **updates,
        }
    )


def fit_provenance(
    config: NutsSettings,
    *,
    versions: tuple[BackendVersion, ...] = (
        BackendVersion(name="fixture", version="metadata-only"),
    ),
) -> Provenance:
    """Record the normal-location fixture without claiming automatic qualification."""
    target = reference("theta~Normal(0,1); y_i|theta~Normal(theta,1); Lebesgue")
    return Provenance(
        schema_version=1,
        prior=reference("theta~Normal(0,1)"),
        design=reference("observed [.5,1.5,1,2]; four independent observations"),
        simulator=reference("native normal-location fixture source-v1"),
        target=LikelihoodTargetReference(
            kind="analytic",
            intended=target,
            evaluated=target,
            qualification=Qualification(
                status="unqualified",
                scope=target,
                reason="Explicit caller model identity; no automatic certification",
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="reference_fit",
        estimator_settings=config.reference,
        backend_versions=versions,
        commit=reference("immutable normal-location fixture source"),
    )


def unused_model(*_args: object, **_kwargs: object) -> None:
    """Preflight failures must occur before the model can be evaluated."""
    pytest.fail("Preflight failure reached the expensive model")


def test_settings_round_trip_and_immutable_reference() -> None:
    """All resolved settings survive serialization with the same artifact identity."""
    config = settings()
    restored = NutsSettings.model_validate_json(config.model_dump_json())
    assert config == restored
    assert config.reference == restored.reference
    assert config.reference.kind == "numpyro-nuts-settings-v1"
    assert config.reference == ArtifactReference.from_bytes(
        config.manifest_json().encode(), kind="numpyro-nuts-settings-v1"
    )
    assert (
        NutsSettings.model_validate_json(
            json.dumps(json.loads(config.manifest_json())["settings"])
        )
        == config
    )
    assert config.reference != settings(num_samples=513).reference
    with pytest.raises(ValidationError):
        config.num_chains = 9
    invalid = config.model_copy(update={"num_chains": 0})
    with pytest.raises(ValidationError):
        _ = invalid.reference


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_samples", 3),
        ("num_samples", True),
        ("num_samples", 4.5),
        ("num_chains", 0),
        ("num_warmup", 0),
        ("target_accept_prob", 0.0),
        ("target_accept_prob", 1.0),
        ("target_accept_prob", float("nan")),
        ("target_accept_prob", float("inf")),
        ("max_tree_depth", 0),
        ("dense_mass", "yes"),
        ("enable_x64", 1),
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("unexpected", "ignored"),
    ],
)
def test_invalid_native_settings(field: str, value: object) -> None:
    """Bad controls fail before native execution or a settings reference exists."""
    with pytest.raises(ValidationError):
        settings(**{field: value})


@pytest.mark.parametrize(
    "updates",
    [
        {"sites": ()},
        {"sites": (SiteLayout(name="theta"), SiteLayout(name="theta"))},
        {"sites": (SiteLayout(name="chain"),)},
        {"sites": (SiteLayout(name="theta", dims=("unit",)),)},
        {"coordinates": (Coordinate(name="unit", labels=("a", "b")),)},
        {"coordinates": (Coordinate(name="draw", labels=(0,)),)},
    ],
)
def test_incomplete_layout_fails_early(updates: dict[str, object]) -> None:
    """Portable, complete layouts are required before asking a native sampler to run."""
    with pytest.raises(ValidationError):
        settings(**updates)


def test_portable_native_names_and_repeated_event_axes() -> None:
    """A native dotted name cannot silently be renamed into a posterior identifier."""
    with pytest.raises(ValidationError):
        SiteLayout(name="native.dotted")
    with pytest.raises(ValidationError):
        settings(
            sites=(SiteLayout(name="theta", dims=("unit", "unit")),),
            coordinates=(Coordinate(name="unit", labels=("a", "b")),),
        )


def test_settings_provenance_mismatch_prevents_model_access() -> None:
    """A changed random stream or sampling budget cannot inherit the old identity."""
    config = settings()
    changed = settings(key=RandomKey(seed=2027, stream="fit"))
    with pytest.raises(ValueError, match="Estimator settings provenance mismatch"):
        fit_nuts(unused_model, settings=changed, provenance=fit_provenance(config))


@pytest.mark.parametrize("name", ["rng_key", "extra_fields", "init_params"])
def test_model_inputs_cannot_override_run_controls(name: str) -> None:
    """Dynamic model kwargs cannot silently change the declared native algorithm."""
    config = settings()
    with pytest.raises(ValueError, match="run controls"):
        fit_nuts(
            unused_model,
            settings=config,
            provenance=fit_provenance(config),
            model_kwargs={name: None},
        )


def test_augmented_target_requires_its_qualified_adapter() -> None:
    """Ordinary parameter-only NUTS cannot acquire a pseudo-marginal claim."""
    config = settings()
    original = fit_provenance(config)
    changed = original.model_copy(
        update={
            "target": original.target.model_copy(
                update={"kind": "augmented_pseudo_marginal"}
            )
        }
    )
    with pytest.raises(ValueError, match="pseudo-marginal"):
        fit_nuts(unused_model, settings=config, provenance=changed)


def test_backend_version_mismatch_prevents_model_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different installed backend cannot inherit an expected fit identity."""
    config = settings()
    monkeypatch.setattr(
        adapter,
        "numpyro_backend_versions",
        lambda: (BackendVersion(name="jax", version="changed"),),
    )
    with pytest.raises(ValueError, match="Backend versions provenance mismatch"):
        fit_nuts(unused_model, settings=config, provenance=fit_provenance(config))


def test_missing_numpyro_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core-only environments can import metadata and get a useful optional error."""
    original = importlib.metadata.version

    def version(name: str) -> str:
        if name == "numpyro":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    with pytest.raises(ImportError, match=r"op_infer\[numpyro\]"):
        adapter.numpyro_backend_versions()


def test_settings_reference_covers_event_coordinate_order() -> None:
    """Coordinate labels/order and output interpretation are part of estimator reuse."""
    original = settings(
        sites=(SiteLayout(name="theta", dims=("unit",)),),
        coordinates=(Coordinate(name="unit", labels=("b", "a")),),
    )
    changed = settings(
        sites=original.sites,
        coordinates=(Coordinate(name="unit", labels=("a", "b")),),
    )
    assert original.reference != changed.reference
