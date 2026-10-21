"""Named native callable outputs and explicit guarded structural provenance."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.forward import CallableForward, ForwardDesign, ForwardModel
from op_infer.posterior import Provenance
from op_infer.rng import RandomKey, SimulationStreams
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from collections.abc import Mapping


def identified_provenance() -> Provenance:
    """Declare an independent callable manifest without importing a compiler."""
    return provenance().with_structural_model(
        reference(
            "ordered scalar decay model manifest", kind="declared-callable-model-v1"
        )
    )


def test_legacy_provenance_identity_and_serialization_are_unchanged() -> None:
    """Adding explicit structure never silently rewrites existing schema-1 banks."""
    legacy = provenance()
    assert (
        legacy.identity
        == "sha256:a91d8c0739999fa068a3196c11c04ee61b6298b4aedba7775b3ef349c0816b82"
    )
    assert "structural_model" not in json.loads(legacy.model_dump_json())
    assert Provenance.load_json(legacy.model_dump_json(), expected=legacy) == legacy
    modern = identified_provenance()
    assert modern.schema_version == 2
    assert modern.structural_model is not None
    assert modern.identity != legacy.identity
    assert Provenance.load_json(modern.model_dump_json(), expected=modern) == modern
    with pytest.raises(ValueError, match="Provenance mismatch"):
        Provenance.load_json(legacy.model_dump_json(), expected=modern)
    with pytest.raises(ValueError, match="Provenance mismatch"):
        Provenance.load_json(modern.model_dump_json(), expected=legacy)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model",
        "unknown_model_schema",
        "unknown_provenance_schema",
        "legacy_model",
    ],
)
def test_unknown_or_incomplete_structural_contract_is_refused(mutation: str) -> None:
    """Versions and structural fields must agree before reuse is possible."""
    payload = identified_provenance().model_dump(mode="json")
    if mutation == "missing_model":
        del payload["structural_model"]
    elif mutation == "unknown_model_schema":
        payload["structural_model"]["kind"] = "op-system-structural-model-v2"
    elif mutation == "unknown_provenance_schema":
        payload["schema_version"] = 3
    else:
        payload["schema_version"] = 1
    with pytest.raises(ValidationError):
        Provenance.model_validate_json(json.dumps(payload))


def test_callable_adapter_borrows_inputs_and_obeys_protocol() -> None:
    """No hidden array conversion, parameter draw or observation noise is added."""
    expected = identified_provenance()
    values = np.array([1.0, 2.0])
    design_inputs = {"times": np.array([0.0, 1.0])}
    streams = SimulationStreams(seed=73, index=4)

    def evaluate(
        theta: Mapping[str, object], inputs: object, key: RandomKey
    ) -> Mapping[str, object]:
        assert theta["rate"] is values
        assert inputs is design_inputs
        assert key == streams.simulation
        return {"value": values}

    forward: ForwardModel = CallableForward(evaluate, ("rate",), ("value",), expected)
    outputs = forward(
        {"rate": values},
        ForwardDesign(expected.design, design_inputs),
        streams.simulation,
    )
    assert outputs["value"] is values
    assert streams.parameters != streams.simulation
    assert forward.parameter_names == ("rate",)
    assert forward.output_names == ("value",)
    first = ForwardDesign(expected.design, values)
    assert first != ForwardDesign(expected.design, values.copy())


@pytest.mark.parametrize("mutation", ["design", "missing_parameter", "extra_parameter"])
def test_static_contract_fails_before_calling_the_model(mutation: str) -> None:
    """A changed design or parameter interface cannot silently reuse an adapter."""
    expected = identified_provenance()

    def never(
        _theta: Mapping[str, object], _inputs: object, _key: RandomKey
    ) -> Mapping[str, object]:
        pytest.fail("Invalid forward contract reached consumer callback")

    forward = CallableForward(never, ("rate",), ("value",), expected)
    theta: dict[str, object] = {"rate": 1.0}
    design = ForwardDesign(expected.design, None)
    if mutation == "design":
        design = ForwardDesign(reference("changed physical design"), None)
    elif mutation == "missing_parameter":
        theta = {}
    else:
        theta["extra"] = 2.0
    with pytest.raises(ValueError, match=r"design identity|parameter names"):
        forward(theta, design, RandomKey(seed=1, stream="simulation"))


def test_output_order_and_legacy_adapter_claims_are_refused() -> None:
    """Output layout and explicit model identity are required declarations."""
    expected = identified_provenance()

    def reverse(
        _theta: Mapping[str, object], _inputs: object, _key: RandomKey
    ) -> Mapping[str, object]:
        return {"second": 2.0, "first": 1.0}

    with pytest.raises(ValueError, match="schema-2"):
        CallableForward(reverse, ("rate",), ("first", "second"), provenance())
    forward = CallableForward(reverse, ("rate",), ("first", "second"), expected)
    with pytest.raises(ValueError, match="output names/order"):
        forward(
            {"rate": 1.0},
            ForwardDesign(expected.design, None),
            RandomKey(seed=1, stream="simulation"),
        )
