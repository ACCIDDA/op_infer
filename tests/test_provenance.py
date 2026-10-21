"""Identity changes, immutable references and refused artifact reuse for #10/#43."""

import json
import pickle
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from op_infer.forward import ForwardRunReference
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    DerivativeContract,
    ErrorComponent,
    LikelihoodTargetReference,
    NumericalExecution,
    Provenance,
    Qualification,
)

type ErrorKind = Literal[
    "time",
    "space_quadrature",
    "nonlinear_solve",
    "linear_solve",
    "unit_integration",
    "monte_carlo",
    "sampler",
]
ERROR_KINDS: tuple[ErrorKind, ...] = (
    "time",
    "space_quadrature",
    "nonlinear_solve",
    "linear_solve",
    "unit_integration",
    "monte_carlo",
    "sampler",
)


def reference(value: str, *, kind: str = "fixture-v1") -> ArtifactReference:
    """Reference exact synthetic artifact bytes, without accessing consumer data."""
    return ArtifactReference.from_bytes(value.encode(), kind=kind)


def qualification() -> Qualification:
    """Keep untested mathematical claims explicitly unqualified."""
    return Qualification(
        status="unqualified",
        scope=reference("scalar-decay-fixture"),
        reason="Metadata only; qualification requires independent evidence",
        evidence=(),
    )


def provenance() -> Provenance:
    """Build a complete shared provenance fixture for numerical output reuse."""
    numerical = NumericalExecution(
        solver=reference("rk4; max_step=0.1; float64"),
        time_mesh=reference("0,0.5,1; fixed internal partition"),
        spatial_mesh=None,
        topology=None,
        geometry="frozen",
        replay=None,
        mesh_policy=reference("static deterministic partition on every evaluation"),
        observation=reference("same physical concentration at t=0.5,1"),
        field_prior=reference("scalar concentration; fixed physical prior on rate"),
        derivative=DerivativeContract(
            mode="finite_difference",
            parameters=("rate",),
            adaptive_path="not_applicable",
            graph_scope=reference("central differences; no cross-backend graph"),
            qualification=qualification(),
        ),
        errors=tuple(
            ErrorComponent(
                kind=kind,
                status="unqualified",
                reason="Not qualified in metadata-only fixture",
                controls=None,
                evidence=None,
            )
            for kind in ERROR_KINDS
        ),
        qualification=qualification(),
    )
    return Provenance(
        schema_version=1,
        prior=reference("physical rate ~ Uniform(0.1,2)"),
        design=reference("same physical observation times and data"),
        simulator=reference("scalar exponential decay model v1"),
        target=LikelihoodTargetReference(
            kind="deterministic_approximation",
            intended=reference("analytic decay Gaussian likelihood"),
            evaluated=reference("fixed RK4 approximate likelihood"),
            qualification=qualification(),
        ),
        numerical=numerical,
        estimator_stage="qualification",
        estimator_settings=reference("no sampler; independent reference only"),
        backend_versions=(BackendVersion(name="numpy", version="fixture"),),
        commit=reference("immutable test source manifest"),
    )


def test_guarded_round_trip_and_forward_reference() -> None:
    """A forward artifact uses exactly the same identity as a posterior artifact."""
    expected = provenance()
    restored = Provenance.load_json(expected.model_dump_json(), expected=expected)
    assert restored == expected
    assert restored.identity == expected.identity
    forward = ForwardRunReference(outputs=reference("1,0.5,0.25"), provenance=restored)
    assert forward.provenance.identity == expected.identity
    assert pickle.loads(pickle.dumps(restored)) == expected  # noqa: S301
    assert hash(restored) == hash(expected)


def test_json_formatting_does_not_change_identity() -> None:
    """Whitespace and object-key ordering do not change validated behavior identity."""
    expected = provenance()
    payload = json.dumps(expected.model_dump(mode="json"), indent=4, sort_keys=True)
    assert (
        Provenance.load_json(payload, expected=expected).identity == expected.identity
    )


@pytest.mark.parametrize(
    "path",
    [
        ("prior",),
        ("design",),
        ("simulator",),
        ("commit",),
        ("estimator_settings",),
        ("target", "intended"),
        ("target", "evaluated"),
        ("numerical", "solver"),
        ("numerical", "time_mesh"),
        ("numerical", "observation"),
        ("numerical", "field_prior"),
        ("numerical", "mesh_policy"),
        ("numerical", "derivative", "graph_scope"),
    ],
)
def test_behavior_changes_are_refused(path: tuple[str, ...]) -> None:
    """Any referenced target/numerical behavior change blocks silent artifact reuse."""
    expected = provenance()
    changed: Any = expected.model_dump(mode="json")
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = reference("changed behavior").model_dump(mode="json")
    other = Provenance.model_validate_json(json.dumps(changed))
    assert other.identity != expected.identity
    with pytest.raises(ValueError, match="Provenance mismatch"):
        Provenance.load_json(json.dumps(changed), expected=expected)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("target", "kind", "frozen_random_approximation"),
        ("derivative", "mode", "discrete_autodiff"),
        ("derivative", "parameters", ["log_rate"]),
        ("derivative", "adaptive_path", "differentiate_controller"),
    ],
)
def test_target_and_derivative_semantics_are_part_of_identity(
    section: str, field: str, value: str | list[str]
) -> None:
    """Changing a label cannot preserve the prior behavior identity."""
    expected = provenance()
    changed: Any = expected.model_dump(mode="json")
    destination = (
        changed["target"] if section == "target" else changed["numerical"][section]
    )
    destination[field] = value
    with pytest.raises(ValueError, match="Provenance mismatch"):
        Provenance.load_json(json.dumps(changed), expected=expected)


@pytest.mark.parametrize("missing", ["schema_version", "prior", "target", "numerical"])
def test_missing_identity_and_unknown_schema_are_refused(missing: str) -> None:
    """Legacy records cannot inherit an identity they do not contain."""
    expected = provenance()
    payload = expected.model_dump(mode="json")
    del payload[missing]
    with pytest.raises(ValidationError):
        Provenance.load_json(json.dumps(payload), expected=expected)
    payload = expected.model_dump(mode="json")
    payload["schema_version"] = 3
    with pytest.raises(ValidationError):
        Provenance.load_json(json.dumps(payload), expected=expected)


def test_unknown_fields_and_mutable_nested_values_are_refused() -> None:
    """Metadata is detached, immutable and rejects silently ignored settings."""
    expected = provenance()
    payload = expected.model_dump(mode="json")
    payload["unrecorded_override"] = "new solver"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Provenance.load_json(json.dumps(payload), expected=expected)
    with pytest.raises(ValidationError, match="frozen_instance"):
        expected.estimator_stage = "other"
    assert expected.numerical is not None
    with pytest.raises(ValidationError, match="frozen_instance"):
        expected.numerical.derivative.mode = "custom"
    with pytest.raises(ValidationError, match="frozen_instance"):
        expected.backend_versions[0].version = "changed"


@pytest.mark.parametrize(
    "digest", ["", "abc", "sha256:" + "a" * 63, "sha256:" + "G" * 64]
)
def test_invalid_content_references_are_refused(digest: str) -> None:
    """References require a real digest format, not an arbitrary display label."""
    with pytest.raises(ValidationError):
        ArtifactReference(kind="fixture", digest=digest)


def test_qualified_claim_needs_evidence_but_metadata_does_not_prove_it() -> None:
    """A declared pass needs evidence references; the consumer still audits evidence."""
    payload = qualification().model_dump(mode="json")
    payload["status"] = "qualified"
    with pytest.raises(ValidationError, match="require scoped evidence"):
        Qualification.model_validate_json(json.dumps(payload))
    payload["evidence"] = [reference("claimed evidence").model_dump(mode="json")]
    assert Qualification.model_validate_json(json.dumps(payload)).status == "qualified"


@pytest.mark.parametrize(
    "mutation", ["missing_error", "duplicate_error", "mesh", "replay"]
)
def test_incomplete_or_contradictory_numerical_contract_is_refused(
    mutation: str,
) -> None:
    """Unknown error sources stay explicit; geometry and replay need their artifacts."""
    expected = provenance()
    payload: Any = expected.model_dump(mode="json")
    numerical = payload["numerical"]
    if mutation == "missing_error":
        numerical["errors"].pop()
    elif mutation == "duplicate_error":
        numerical["errors"][-1] = numerical["errors"][0]
    elif mutation == "mesh":
        numerical["geometry"] = "learnable_fixed_topology"
    else:
        numerical["derivative"]["adaptive_path"] = "frozen_replay"
    with pytest.raises(ValidationError):
        Provenance.load_json(json.dumps(payload), expected=expected)


def test_analytic_provenance_cannot_masquerade_as_numerical_forward_run() -> None:
    """Analytic absence is allowed; numerical solver runs need an execution contract."""
    payload = provenance().model_dump(mode="json")
    payload["numerical"] = None
    analytic = Provenance.model_validate_json(json.dumps(payload))
    assert (
        Provenance.load_json(analytic.model_dump_json(), expected=analytic) == analytic
    )
    with pytest.raises(ValidationError, match="require an execution contract"):
        ForwardRunReference(outputs=reference("outputs"), provenance=analytic)


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_schema_version_is_an_integer_not_a_coercion(version: object) -> None:
    """Boolean/float equality to one cannot select a supported artifact format."""
    expected = provenance()
    payload = expected.model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValidationError):
        Provenance.load_json(json.dumps(payload), expected=expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("estimator_stage", "reporting"),
        ("backend_versions", [{"name": "numpy", "version": "other-version"}]),
    ],
)
def test_execution_changes_block_reuse(field: str, value: object) -> None:
    """Stages and dependency versions are execution identity, not display metadata."""
    expected = provenance()
    payload = expected.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError, match="Provenance mismatch"):
        Provenance.load_json(json.dumps(payload), expected=expected)


@pytest.mark.parametrize("mutation", ["controls", "parameters", "backend", "topology"])
def test_metadata_invariants_reject_incomplete_execution(mutation: str) -> None:
    """Coordinates, settings, versions and mesh connectivity stay explicit."""
    expected = provenance()
    payload: Any = expected.model_dump(mode="json")
    numerical = payload["numerical"]
    if mutation == "controls":
        numerical["errors"][0]["status"] = "specified"
    elif mutation == "parameters":
        numerical["derivative"]["parameters"] = ["rate", "rate"]
    elif mutation == "backend":
        payload["backend_versions"] *= 2
    else:
        numerical["spatial_mesh"] = reference("coordinates").model_dump(mode="json")
    with pytest.raises(ValidationError):
        Provenance.load_json(json.dumps(payload), expected=expected)
