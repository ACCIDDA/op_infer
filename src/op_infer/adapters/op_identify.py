"""Optional read-only compiler reports and measured sensitivity interoperability.

No second gauge detector or ODE solver is implemented. Structural action ranks,
finite-design Gaussian spectra and posterior geometry retain separate meanings.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from array_api_compat import array_namespace

from op_infer.adapters.op import structural_model_reference
from op_infer.geometry import (
    GaussianInformation,
    InformationContext,
    ScalingAssessment,
    ScalingGenerator,
    ScalingVerification,
    gaussian_mean_information,
)
from op_infer.observe import IIDObservation
from op_infer.posterior import ArtifactReference

__all__ = [
    "information_from_sensitivities",
    "scaling_assessment",
    "sensitivity_execution_reference",
]


def scaling_assessment(
    compiled: object, *, expected: ArtifactReference | None = None
) -> ScalingAssessment:
    """Copy an existing compiler report; never rerun or invent a detector.

    The producer's parameter-base layout differs from an expanded physical
    parameter sensitivity layout. Finite probes retain their seed/tolerances;
    a reported action does not assess initial conditions or priors.

    Returns:
        Versioned unassessed, unsupported or reported scaling evidence.

    Raises:
        ValueError: The independently expected source identity differs.
    """
    artifact: Any = compiled
    source = structural_model_reference(compiled)
    if expected is not None and expected != source:
        msg = "Structural assessment source model identity mismatch"
        raise ValueError(msg)
    if (
        artifact.operators
        or artifact.history_requirements
        or artifact.meta.get("time_varying_params")
        or artifact.gauge_fix is not None
    ):
        return ScalingAssessment(
            source_model=source,
            status="unsupported",
            reason=(
                "Operator/history/table or gauge-fixed execution needs "
                "a separate structural/statistical contract"
            ),
        )
    info = artifact.gauges
    if info is None:
        return ScalingAssessment(
            source_model=source,
            status="unassessed",
            reason="No compiler scaling assessment was requested; no zero-rank claim",
        )
    verification = None
    if info.verification is not None:
        probe = info.verification
        verification = ScalingVerification(
            seed=probe.seed,
            n_probes=probe.n_probes,
            scales=probe.scales,
            rtol=probe.rtol,
            atol=probe.atol,
            max_absolute_error=probe.max_absolute_error,
            observables=probe.observables,
            structural_homogeneity_checked=probe.structural,
        )
    return ScalingAssessment(
        source_model=source,
        status="reported",
        reason=(
            "Existing shared positive-scaling analysis; initial conditions "
            "and prior effects unassessed"
        ),
        parameter_bases=info.parameter_names,
        generators=tuple(
            ScalingGenerator(name=g.name, states=g.states, parameters=g.parameters)
            for g in info.generators
        ),
        action_rank=info.rank,
        parameter_action_rank=info.parameter_rank,
        invariant_monomials=tuple(
            value.exponents for value in info.invariant_coordinates
        ),
        detection_requested=info.detection_requested,
        verification=verification,
    )


def _solution(value: object) -> Any:  # noqa: ANN401
    try:
        module = importlib.import_module("op_engine.sensitivities")
    except ImportError as error:
        msg = "Install op_engine>=0.9.0, which provides forward sensitivities"
        raise ImportError(msg) from error
    if not isinstance(value, module.SensitivitySolution):
        msg = "Expected the qualified engine SensitivitySolution contract"
        raise TypeError(msg)
    return value


def sensitivity_execution_reference(solution: object) -> ArtifactReference:
    """Identify relevant fixed-sensitivity execution/layout without copying arrays.

    Initial functions and parameter values remain separately identified caller
    inputs. This reference does not authenticate the implementation or replace
    independent qualification evidence and runtime/source records.

    Returns:
        Versioned numerical execution reference expected by InformationContext.
    """
    result = _solution(solution)
    options = result.options
    settings = options.solver
    data = {
        "format": "op-engine-forward-sensitivities",
        "schema_version": 1,
        "source_model": result.source_model,
        "states": result.state_names,
        "parameters": result.parameter_names,
        "observables": result.observable_names,
        "mesh": result.mesh,
        "retained_times": result.times,
        "dtype": str(result.states.dtype),
        "method": settings.method,
        "fixed_max_step": settings.fixed_max_step,
        "fixed_loop": settings.fixed_loop,
        "fixed_checkpoint": settings.fixed_checkpoint,
        "store_history": options.store_history,
        "max_augmented_states": options.max_augmented_states,
        "max_output_bytes": options.max_output_bytes,
    }
    return ArtifactReference.from_bytes(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode(),
        kind="op-engine-forward-sensitivities-v1",
    )


def information_from_sensitivities(
    solution: object,
    observation: IIDObservation,
    *,
    context: InformationContext,
    mask: object | None = None,
) -> GaussianInformation:
    """Apply one fixed iid Normal law to the engine's measured sensitivities.

    The affine measurement slope is included; its fixed offset has zero
    derivative. The known fixed scale supplies the conditional observation
    variance. Censoring/count/dependent laws need separate information formulas.

    Returns:
        Physical-coordinate mean information with exact source/layout guards.

    Raises:
        ValueError: Model, numerical, observation or coordinate layouts differ.
        NotImplementedError: The observation model needs a different formula.
    """
    result = _solution(solution)
    context = InformationContext.model_validate(context)
    observation = IIDObservation.model_validate(observation)
    if (
        observation.family != "normal"
        or observation.lower_limit is not None
        or observation.upper_limit is not None
    ):
        msg = "Engine information adapter requires uncensored iid Normal observations"
        raise NotImplementedError(msg)
    if (
        context.source_model.kind != "op-system-structural-model-v1"
        or context.source_model.digest != result.source_model
        or context.parameter_names != result.parameter_names
        or context.observable_names != result.observable_names
        or context.observation != observation.reference
        or context.numerical != sensitivity_execution_reference(result)
    ):
        msg = "Information model/observation/numerical/layout context mismatch"
        raise ValueError(msg)
    values = result.observable_derivatives * observation.slope
    xp = array_namespace(values)
    sigma = xp.full_like(xp.sum(values, axis=-1), observation.scale)
    return gaussian_mean_information(values, sigma, context=context, mask=mask)
