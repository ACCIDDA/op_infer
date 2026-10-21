"""Complete deterministic forward example; metadata is explicitly unqualified.

Requires the optional model stack and the structural-identity compiler revision
documented in docs/forward-models.md. No prior, observation noise or fit is added.
"""

from __future__ import annotations

import importlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from op_infer.adapters.op import (
    FixedForwardSettings,
    compiled_forward,
    compiled_observation_reference,
    fixed_forward_mesh_reference,
    fixed_forward_settings_reference,
    structural_model_reference,
)
from op_infer.forward import ForwardDesign
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
from op_infer.rng import SimulationStreams

if TYPE_CHECKING:
    from collections.abc import Mapping


def _ref(text: str) -> ArtifactReference:
    """Reference exact illustrative metadata, without claiming qualification."""
    return ArtifactReference.from_bytes(text.encode(), kind="forward-example-v1")


def main() -> None:
    """Run compiled observations and compare them with independent analytic decay."""
    compiler = importlib.import_module("op_system")
    compiled = compiler.compile_spec(
        {
            "kind": "expr",
            "state": ["x"],
            "equations": {"x": "-rate*x"},
            "observables": {"value": {"expr": "scale*x"}},
        }
    )
    settings = FixedForwardSettings(times=(0.0, 0.5, 1.0), max_step=0.05)
    solver = fixed_forward_settings_reference(settings)
    evidence = Qualification(
        status="unqualified",
        scope=_ref("this scalar forward example only"),
        reason="Illustrative execution metadata; no general error or statistical claim",
        evidence=(),
    )
    other_kinds: tuple[
        Literal[
            "space_quadrature",
            "nonlinear_solve",
            "linear_solve",
            "unit_integration",
            "monte_carlo",
            "sampler",
        ],
        ...,
    ] = (
        "space_quadrature",
        "nonlinear_solve",
        "linear_solve",
        "unit_integration",
        "monte_carlo",
        "sampler",
    )
    numerical = NumericalExecution(
        solver=solver,
        time_mesh=fixed_forward_mesh_reference(settings),
        spatial_mesh=None,
        topology=None,
        geometry="frozen",
        replay=None,
        mesh_policy=_ref("same static partition for every evaluation"),
        observation=compiled_observation_reference(compiled, ("value",)),
        field_prior=_ref(
            "natural scalar state; forward-only, no statistical field prior"
        ),
        derivative=DerivativeContract(
            mode="none",
            parameters=(),
            adaptive_path="not_applicable",
            graph_scope=_ref("value-only example; no derivative requested"),
            qualification=evidence,
        ),
        errors=(
            ErrorComponent(
                kind="time",
                status="specified",
                reason="Step control is not an achieved global error guarantee",
                controls=solver,
                evidence=None,
            ),
            *(
                ErrorComponent(
                    kind=kind,
                    status="not_applicable",
                    reason="Not used in this scalar deterministic forward example",
                    controls=None,
                    evidence=None,
                )
                for kind in other_kinds
            ),
        ),
        qualification=evidence,
    )
    expected = Provenance(
        schema_version=2,
        structural_model=structural_model_reference(compiled),
        prior=_ref("not applicable: parameters supplied directly; no prior sampled"),
        design=_ref("latent scale*x at physical times 0,0.5,1"),
        simulator=_ref("op_infer compiled fixed forward adapter; natural coordinates"),
        target=LikelihoodTargetReference(
            kind="deterministic_approximation",
            intended=_ref("6*exp(-rate*t); no likelihood or fit in this example"),
            evaluated=solver,
            qualification=evidence,
        ),
        numerical=numerical,
        estimator_stage="forward-example",
        estimator_settings=_ref("no estimator or posterior in this example"),
        backend_versions=tuple(
            BackendVersion(name=name, version=version(name))
            for name in ("numpy", "op-engine", "op-system", "pydantic")
        ),
        commit=ArtifactReference.from_bytes(
            Path(__file__).read_bytes(), kind="forward-example-source-v1"
        ),
    )

    def initial(theta: Mapping[str, object], _design: object) -> object:
        return np.reshape(np.asarray(theta["initial"]), (1,))

    forward = compiled_forward(
        compiled,
        settings=settings,
        initial_state=initial,
        parameter_names=("rate", "scale", "initial"),
        output_names=("value",),
        provenance=expected,
    )
    streams = SimulationStreams(seed=2026, index=0)
    theta = {
        "rate": np.asarray(0.7),
        "scale": np.asarray(3.0),
        "initial": np.asarray(2.0),
    }
    outputs = forward(theta, ForwardDesign(expected.design, None), streams.simulation)
    result = np.asarray(outputs["value"])
    error = float(
        np.max(np.abs(result - 6 * np.exp(-0.7 * np.asarray(settings.times))))
    )
    if error > 3e-7:
        msg = "Compiled forward disagrees with analytic scalar decay"
        raise RuntimeError(msg)
    print(  # noqa: T201
        json.dumps(
            {
                "provenance": expected.identity,
                "model": expected.structural_model.model_dump(mode="json")
                if expected.structural_model
                else None,
                "outputs": result.tolist(),
                "max_absolute_error": error,
            }
        )
    )


if __name__ == "__main__":
    main()
