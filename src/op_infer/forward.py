"""Forward models, designs and simulators.

Status: immutable forward-run references, a named-output protocol and a guarded
plain-callable adapter are implemented. Optional numerical adapters live outside
the core. See ``docs/design/numerical-provenance.md``.

Scope: ForwardModel protocol (theta, design, key) -> latent outputs, with adapters for
op_system CompiledRhs + op_engine solvers and plain callables; Simulator = forward +
observation + prior, compatible with trade_study.Simulator; Design objects with sampling
schedules, limits, controlled covariates and a design hash; compile-once kernels with
data passed as traced inputs.

"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from op_infer.posterior import ArtifactReference, Provenance
from op_infer.rng import RandomKey

__all__ = ["CallableForward", "ForwardDesign", "ForwardModel", "ForwardRunReference"]


@dataclass(frozen=True, slots=True, eq=False)
class ForwardDesign:
    """Keep physical design identity beside native inputs without copying arrays.

    The reference is an independently supplied content
    claim, not an automatic digest or authenticity/integrity check of ``inputs``.
    Treat borrowed inputs as immutable; retain their graphs and device placement.
    Design objects use identity equality; compare ``reference`` explicitly.

    Attributes:
        reference: Physical design/data/layout identity expected by provenance.
        inputs: Consumer-defined native design inputs, passed through unchanged.

    Examples:
        >>> inputs = {"sampling_times": (0.0, 1.0)}
        >>> reference = ArtifactReference.from_bytes(b"times=0,1", kind="design-v1")
        >>> design = ForwardDesign(reference, inputs)
        >>> design.inputs is inputs
        True
    """

    reference: ArtifactReference
    inputs: object

    def __post_init__(self) -> None:
        """Validate immutable metadata without examining native inputs."""
        ArtifactReference.model_validate(self.reference)


class ForwardModel(Protocol):
    """Named latent outputs from explicit parameters, design and simulation key.

    Statistical sampling and log density are separate.
    Implementations expose ordered parameter/output layouts and the shared
    provenance contract; backend/JIT/derivative support is adapter-specific.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the declared parameter binding order."""
        ...

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the declared latent output order."""
        ...

    @property
    def provenance(self) -> Provenance:
        """Return shared structural/design/numerical/runtime identity."""
        ...

    def __call__(
        self, theta: Mapping[str, object], design: ForwardDesign, key: RandomKey
    ) -> Mapping[str, object]:
        """Evaluate native outputs without implicitly drawing parameters or noise."""
        ...


@dataclass(frozen=True, slots=True)
class CallableForward:
    """Adapt an explicitly identified callable without optional numerical libraries.

    Callbacks receive the original native parameter/design
    inputs and simulation key. Use ``SimulationStreams`` to draw parameters on
    its separate parameter stream. The caller declares a model manifest; this
    adapter never hashes callable repr, bytecode, globals or closure contents.

    Attributes:
        function: ``(theta, design.inputs, key) -> named native latent outputs``.
        parameter_names: Complete ordered parameter names, including initial-state
            and observable-only parameters used by this callable.
        output_names: Complete ordered latent output names.
        provenance: Schema-2 shared provenance, with declared model content.
    """

    function: Callable[
        [Mapping[str, object], object, RandomKey], Mapping[str, object]
    ] = field(repr=False, compare=False)
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        """Validate declarations before invoking a consumer callback.

        Raises:
            ValueError: Layouts or structural provenance are incomplete.
            TypeError: The supplied callback is not callable.
        """
        Provenance.model_validate(self.provenance)
        if self.provenance.schema_version != 2:
            msg = "Forward adapters require schema-2 structural model provenance"
            raise ValueError(msg)
        for names in (self.parameter_names, self.output_names):
            if (
                not isinstance(names, tuple)
                or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)
            ):
                msg = "Forward parameter/output layouts must be nonempty unique tuples"
                raise ValueError(msg)
        if not callable(self.function):
            msg = "Forward function must be callable"
            raise TypeError(msg)

    def __call__(
        self, theta: Mapping[str, object], design: ForwardDesign, key: RandomKey
    ) -> Mapping[str, object]:
        """Check static identity/layout, then preserve native callback outputs.

        Args:
            theta: Complete named parameters; values are borrowed, never converted.
            design: Independently supplied expected physical design and inputs.
            key: Explicit simulation stream address, separate from parameter draws.

        Returns:
            Named native outputs in the declared order, without array copies.

        Raises:
            ValueError: Design, parameter names or output names differ.
            TypeError: The callback does not return named outputs.
        """
        if design.reference != self.provenance.design:
            msg = "Forward design identity mismatch"
            raise ValueError(msg)
        if set(theta) != set(self.parameter_names):
            msg = "Forward parameter names differ from the declared binding layout"
            raise ValueError(msg)
        RandomKey.model_validate(key)
        outputs = self.function(theta, design.inputs, key)
        if not isinstance(outputs, Mapping):
            msg = "Forward callback must return a mapping of named outputs"
            raise TypeError(msg)
        if tuple(outputs) != self.output_names:
            msg = "Forward output names/order differ from the declared layout"
            raise ValueError(msg)
        return outputs


class ForwardRunReference(BaseModel):
    """Attach shared provenance to numerical outputs.

    This is metadata for immutable outputs, not an array container or solver adapter.

    Attributes:
        outputs: Reference to numerical output bytes, including dtype/shape/layout.
        provenance: Shared target/numerical identity used by posterior artifacts.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    outputs: ArtifactReference
    provenance: Provenance

    @model_validator(mode="after")
    def _require_numerical_execution(self) -> Self:
        if self.provenance.numerical is None:
            msg = "Numerical forward references require an execution contract"
            raise ValueError(msg)
        return self
