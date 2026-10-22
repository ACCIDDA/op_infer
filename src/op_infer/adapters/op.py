"""Optional compiler/fixed-engine adapter with observation-only retention.

Numerical stages and namespace loop operations are
owned by op_engine. This adapter binds named parameters and evaluates existing
compiled latent observables; it adds neither integration methods nor noise.
"""

from __future__ import annotations

import importlib
import json
import math
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from array_api_compat import array_namespace
from pydantic import BaseModel, ConfigDict, Field, model_validator

from op_infer.forward import CallableForward
from op_infer.posterior import ArtifactReference, Provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from op_infer.rng import RandomKey

__all__ = [
    "FixedForwardSettings",
    "compiled_forward",
    "compiled_observation_reference",
    "fixed_forward_mesh_reference",
    "fixed_forward_settings_reference",
    "structural_model_reference",
]


class FixedForwardSettings(BaseModel):
    """Static bounded deterministic solver and stored-observation-time choices.

    Attributes:
        times: Start time followed by requested observation times; no interpolation.
        method: Existing engine fixed explicit method.
        dtype: Declared native state precision, checked without converting arrays.
        max_step: Maximum internal step between stored observation times.
        fixed_loop: Eager stepping or namespace-native scan selection.
        checkpoint: Rematerialize a forced scan's body during reverse mode.
        max_internal_steps: Hard setup limit on the complete fixed partition.

    Examples:
        >>> settings = FixedForwardSettings(times=(0.0, 1.0), max_step=0.1)
        >>> settings.method, settings.fixed_loop
        ('rk4', 'auto')
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    times: tuple[float, ...]
    method: Literal["euler", "heun", "rk4", "dopri5"] = "rk4"
    dtype: Literal["float32", "float64"] = "float64"
    max_step: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    fixed_loop: Literal["auto", "scan", "unroll"] = "auto"
    checkpoint: bool = False
    max_internal_steps: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def _static_grid(self) -> Self:
        if (
            not self.times
            or not all(math.isfinite(t) for t in self.times)
            or any(right <= left for left, right in pairwise(self.times))
        ):
            msg = "Forward times must be finite, nonempty and strictly increasing"
            raise ValueError(msg)
        if self.checkpoint and self.fixed_loop != "scan":
            msg = "Forward checkpointing requires fixed_loop='scan'"
            raise ValueError(msg)
        return self


def _reference(value: object, *, kind: str) -> ArtifactReference:
    """Reference portable adapter metadata.

    Returns:
        Canonical content reference without retaining arrays or closures.
    """
    return ArtifactReference.from_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode(),
        kind=kind,
    )


def _modules() -> tuple[Any, Any, Any, Any, Any]:
    """Import optional libraries only when their adapter is used.

    Returns:
        Compiler identity, solver, model, observation and loop modules.

    Raises:
        ImportError: The optional packages or structural identity API are missing.
    """
    try:
        return (
            importlib.import_module("op_system.identity"),
            importlib.import_module("op_engine.core_solver"),
            importlib.import_module("op_engine.model_core"),
            importlib.import_module("op_engine.observations"),
            importlib.import_module("op_engine.loop_ops"),
        )
    except ImportError as error:
        msg = (
            "Install the op extra with an op_system revision "
            "providing structural identity"
        )
        raise ImportError(msg) from error


def structural_model_reference(compiled: object) -> ArtifactReference:
    """Reuse the compiler's exact schema-1 manifest digest.

    Args:
        compiled: Unmodified compiler-produced model, without runtime bindings.

    Returns:
        Existing generic artifact reference, not a second model identity scheme.
    """
    try:
        identity = importlib.import_module("op_system.identity")
    except ImportError as error:
        msg = (
            "Install the op extra with an op_system revision "
            "providing structural identity"
        )
        raise ImportError(msg) from error
    reference = identity.StructuralModelIdentity.from_compiled(compiled)
    return ArtifactReference(
        kind=f"op-system-structural-model-v{reference.schema_version}",
        digest=reference.digest,
    )


def fixed_forward_settings_reference(
    settings: FixedForwardSettings,
) -> ArtifactReference:
    """Return the solver/settings reference required by numerical provenance."""
    return _reference(
        settings.model_dump(mode="json"), kind="op-fixed-forward-settings-v1"
    )


def _partition(settings: FixedForwardSettings) -> tuple[tuple[float, ...], ...]:
    """Plan a static partition using the existing engine's public planner.

    Returns:
        Engine step sizes per output interval, within the declared bound.

    Raises:
        ValueError: The requested partition exceeds the explicit setup limit.
    """
    _identity, solver, *_ = _modules()
    count = 0
    partitions = []
    for left, right in pairwise(settings.times):
        # Bound setup allocation before calling the engine's tuple planner.
        if (right - left) / settings.max_step > settings.max_internal_steps - count:
            msg = "Forward partition exceeds max_internal_steps"
            raise ValueError(msg)
        partition = tuple(solver.fixed_step_sizes(left, right, settings.max_step))
        count += len(partition)
        if count > settings.max_internal_steps:
            msg = "Forward partition exceeds max_internal_steps"
            raise ValueError(msg)
        partitions.append(partition)
    return tuple(partitions)


def fixed_forward_mesh_reference(settings: FixedForwardSettings) -> ArtifactReference:
    """Reference actual engine-planned steps and requested observation times."""
    return _reference(
        {"times": settings.times, "steps": _partition(settings)},
        kind="op-fixed-forward-mesh-v1",
    )


def compiled_observation_reference(
    compiled: object, output_names: tuple[str, ...]
) -> ArtifactReference:
    """Reference the selected ordered latent outputs on stored solver nodes."""
    return _reference(
        {
            "model": structural_model_reference(compiled).model_dump(mode="json"),
            "output_names": output_names,
            "evaluation": "stored-nodes-no-interpolation",
        },
        kind="op-compiled-latent-observation-v1",
    )


def compiled_forward(  # noqa: C901, PLR0913, PLR0915 - static binding and one native evaluator
    compiled: Any,  # noqa: ANN401 - optional compiler boundary
    *,
    settings: FixedForwardSettings,
    initial_state: Callable[[Mapping[str, object], object], object],
    parameter_names: tuple[str, ...],
    output_names: tuple[str, ...],
    provenance: Provenance,
) -> CallableForward:
    """Bind compiled latent observables to existing functional fixed-step kernels.

    The native carry contains current state/FSAL only; scan outputs are ``None``.
    At requested times, existing engine observation adapters evaluate selected
    compiler observables. Only those latent series are stacked. Reverse-mode AD
    may retain its own residuals; checkpointing does not promise a memory bound.
    The returned callable is reused with dynamic parameters/initial states.

    Args:
        compiled: Natural-coordinate, operator/history/table-free compiled model.
        settings: Static fixed partition, output grid and explicit work bound.
        initial_state: Native flat state factory, given parameters/design inputs.
        parameter_names: Complete binding layout, including observable/initial inputs.
        output_names: Selected compiled observable names in returned order.
        provenance: Schema-2 identity with matching model, solver, mesh and observation
            references. Source/runtime and statistical qualification remain explicit.

    Returns:
        A callable satisfying the core forward protocol. Qualified CPU cases use
        NumPy, JAX JIT/vmap/AD and Torch AD; other devices need separate evidence.

    Raises:
        ValueError: Identity, layout, settings or declared work limits disagree.
        NotImplementedError: Operators, history, tables or gauge-fixed execution
            need a separately qualified adapter.
    """
    identity, solver_module, model_module, observations, loops = _modules()
    del identity
    settings = FixedForwardSettings.model_validate(settings)
    provenance = Provenance.model_validate(provenance)
    structural = structural_model_reference(compiled)
    if provenance.schema_version != 2 or provenance.structural_model != structural:
        msg = "Compiled forward structural model identity mismatch"
        raise ValueError(msg)
    if (
        compiled.operators
        or compiled.history_requirements
        or compiled.meta.get("time_varying_params")
        or compiled.meta.get("forcing_breakpoints")
        or compiled.gauge_fix is not None
    ):
        msg = (
            "Compiled forward supports natural fixed explicit models without "
            "operators/history/tables/gauge fixing"
        )
        raise NotImplementedError(msg)
    partitions = _partition(settings)
    numerical = provenance.numerical
    if (
        numerical is None
        or numerical.solver != fixed_forward_settings_reference(settings)
        or numerical.time_mesh != fixed_forward_mesh_reference(settings)
        or numerical.observation
        != compiled_observation_reference(compiled, output_names)
    ):
        msg = "Compiled forward numerical settings/mesh/observation identity mismatch"
        raise ValueError(msg)
    if not output_names or any(
        name not in compiled.observables for name in output_names
    ):
        msg = "Compiled forward requires declared compiled latent observables"
        raise ValueError(msg)
    specs = tuple(
        observations.observation_from_compiled(compiled.observables[name])
        for name in output_names
    )
    rhs_names = tuple(
        dict.fromkeys(
            (
                *compiled.param_names,
                *(name for name, _axes in compiled.meta.get("shaped_params", ())),
            )
        )
    )
    required = set(rhs_names) | {
        name for spec in specs for name, _shape in spec.metadata.parameter_shapes
    }
    if not required <= set(parameter_names):
        msg = "Compiled forward parameter layout omits RHS/observable inputs"
        raise ValueError(msg)
    size = len(compiled.state_names)
    core = model_module.ModelCore(
        size,
        1,
        np.asarray(settings.times),
        options=model_module.ModelCoreOptions(
            store_history=False, dtype=settings.dtype
        ),
    )
    solver = solver_module.CoreSolver(core)

    def evaluate(  # noqa: C901 - bounded output loop and existing namespace stepping
        theta: Mapping[str, object], design: object, key: RandomKey
    ) -> Mapping[str, object]:
        del key  # Deterministic latent execution consumes no simulation randomness.
        initial = initial_state(theta, design)
        xp = array_namespace(initial)
        if getattr(initial, "shape", None) != (size,) or not xp.isdtype(
            getattr(initial, "dtype", None), "real floating"
        ):
            msg = (
                "Compiled forward initial state must be a native "
                "real floating flat state"
            )
            raise ValueError(msg)
        if getattr(initial, "dtype", None) != getattr(xp, settings.dtype):
            msg = "Compiled forward initial-state dtype differs from numerical settings"
            raise ValueError(msg)
        state = xp.reshape(initial, (size, 1))
        adapter = (
            None if settings.fixed_loop == "unroll" else loops.get_loop_adapter(xp)
        )
        if settings.fixed_loop == "scan" and adapter is None:
            msg = "Compiled forward forced scan has no namespace loop adapter"
            raise ValueError(msg)
        if settings.checkpoint and (adapter is None or adapter.checkpoint is None):
            msg = "Compiled forward namespace adapter has no checkpoint operation"
            raise ValueError(msg)
        rhs_params = {name: theta[name] for name in rhs_names}

        def rhs(time: object, values: object) -> object:
            return xp.reshape(
                compiled.eval_fn(time, xp.reshape(values, (-1,)), **rhs_params),
                (size, 1),
            )

        observation_params = {name: theta[name] for name in required}

        def observe(time: object, current: object) -> tuple[object, ...]:
            values = []
            for spec in specs:
                value = spec.eval_fn(time, current, **observation_params)
                if (
                    array_namespace(value) is not xp
                    or getattr(value, "dtype", None) != state.dtype
                ):
                    msg = (
                        "Compiled observations must preserve native namespace and dtype"
                    )
                    raise TypeError(msg)
                if getattr(value, "shape", None) != spec.metadata.output_shape:
                    msg = "Compiled observation shape differs from its metadata"
                    raise ValueError(msg)
                values.append(value)
            return tuple(values)

        def advance(carry: Any, step: Any) -> tuple[Any, None]:  # noqa: ANN401
            values, first_stage = carry
            time, dt = step
            next_values, next_stage = solver.fixed_explicit_step(
                rhs,
                method=settings.method,
                t=time,
                dt=dt,
                y=values,
                first_stage=first_stage,
            )
            return (next_values, next_stage), None

        if adapter is not None:
            body = adapter.checkpoint(advance) if settings.checkpoint else advance
            outputs = _scan_outputs(
                adapter,
                xp,
                body,
                (state, None),
                times=settings.times,
                partitions=partitions,
                observe=observe,
            )
        else:
            rows = [[value] for value in observe(settings.times[0], state)]
            carry = (state, None)
            for (start, end), steps in zip(
                pairwise(settings.times), partitions, strict=True
            ):
                starts = tuple(accumulate(steps, initial=start))[:-1]
                for time, dt in zip(starts, steps, strict=True):
                    carry, _ = advance(carry, (time, dt))
                for row, value in zip(rows, observe(end, carry[0]), strict=True):
                    row.append(value)
            outputs = tuple(xp.stack(row, axis=0) for row in rows)
        return dict(zip(output_names, outputs, strict=True))

    return CallableForward(evaluate, parameter_names, output_names, provenance)


def _scan_outputs(  # noqa: PLR0913 -- native driver and static observation plan
    adapter: Any,  # noqa: ANN401
    xp: Any,  # noqa: ANN401
    body: Any,  # noqa: ANN401
    carry: Any,  # noqa: ANN401
    *,
    times: tuple[float, ...],
    partitions: tuple[tuple[float, ...], ...],
    observe: Callable[[object, object], tuple[object, ...]],
) -> tuple[object, ...]:
    """Scan equal-length observation intervals without stacking primal states.

    One bootstrap step establishes DOPRI5's FSAL carry type. Subsequent groups
    share an outer observation scan and inner step scan; ragged interval lengths
    start a new group. Only requested observations are scan outputs.
    """
    initial = observe(times[0], carry[0])
    chunks = [tuple(xp.expand_dims(value, axis=0) for value in initial)]
    if not partitions:
        return chunks[0]
    first = partitions[0]
    starts = tuple(accumulate(first, initial=times[0]))[:-1]
    carry, _ = body(carry, (starts[0], first[0]))
    if len(first) > 1:
        carry, _ = adapter.scan(
            body,
            carry,
            (
                xp.asarray(starts[1:], dtype=carry[0].dtype),
                xp.asarray(first[1:], dtype=carry[0].dtype),
            ),
        )
    chunks.append(
        tuple(xp.expand_dims(value, axis=0) for value in observe(times[1], carry[0]))
    )
    begin = 1
    while begin < len(partitions):
        stop = begin + 1
        count = len(partitions[begin])
        while stop < len(partitions) and len(partitions[stop]) == count:
            stop += 1
        step_starts = tuple(
            tuple(accumulate(steps, initial=times[index]))[:-1]
            for index, steps in enumerate(partitions[begin:stop], start=begin)
        )

        def interval(current: Any, inputs: Any) -> tuple[Any, tuple[object, ...]]:  # noqa: ANN401
            step_times, sizes, end = inputs
            current, _ = adapter.scan(body, current, (step_times, sizes))
            return current, observe(end, current[0])

        carry, output = adapter.scan(
            interval,
            carry,
            (
                xp.asarray(step_starts, dtype=carry[0].dtype),
                xp.asarray(partitions[begin:stop], dtype=carry[0].dtype),
                xp.asarray(times[begin + 1 : stop + 1], dtype=carry[0].dtype),
            ),
        )
        chunks.append(output)
        begin = stop
    return tuple(
        xp.concat(tuple(chunk[index] for chunk in chunks), axis=0)
        for index in range(len(initial))
    )
