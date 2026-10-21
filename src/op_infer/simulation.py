"""Identified forward-to-observation composition without implicit array transfers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from op_infer.posterior import ArtifactReference
from op_infer.priors import JointPrior
from op_infer.rng import RandomKey, SimulationStreams

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from op_infer.forward import ForwardDesign, ForwardModel
    from op_infer.observe import ObservationModel

__all__ = ["ObservationChannel", "SimulationResult", "StatisticalModel"]


@dataclass(frozen=True, slots=True)
class ObservationChannel:
    """Bind one named measurement stream to a selected latent output."""

    name: str
    latent_output: str
    model: ObservationModel


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Keep parameters, latent values and random measurements distinguishable.

    Arrays are borrowed native values, not immutable snapshots. The pipeline
    reference identifies declarations/execution provenance, not the result bytes.
    Retain streams and serialize/harden arrays at an explicit artifact boundary.
    """

    parameters: Mapping[str, object]
    latent: Mapping[str, object]
    measurements: Mapping[str, object]
    streams: SimulationStreams
    pipeline: ArtifactReference


@dataclass(frozen=True, slots=True)
class StatisticalModel:
    """Reuse an identified forward, ordered prior and statistical observations.

    Calling ``simulate`` uses explicitly supplied native parameters. The separate
    ``sample_numpy`` convenience draws the declared prior on its parameter stream;
    it does not move NumPy draws into JAX/Torch implicitly. ``log_prob`` returns
    per-channel per-cell terms; consumers choose their own unit/event reduction.
    """

    forward: ForwardModel
    prior: JointPrior
    channels: tuple[ObservationChannel, ...]

    def __post_init__(self) -> None:
        """Refuse prior/layout or latent-channel mismatches before simulation."""
        JointPrior.model_validate(self.prior)
        if self.prior.reference != self.forward.provenance.prior:
            msg = "Statistical model prior identity differs from forward provenance"
            raise ValueError(msg)
        if self.prior.parameter_names != self.forward.parameter_names:
            msg = "Statistical model parameter order differs from forward layout"
            raise ValueError(msg)
        names = [channel.name for channel in self.channels]
        if (
            not isinstance(self.channels, tuple)
            or not names
            or len(set(names)) != len(names)
        ):
            msg = "Declare nonempty unique observation channel names"
            raise ValueError(msg)
        for channel in self.channels:
            if (
                not isinstance(channel.name, str)
                or not channel.name.strip()
                or channel.latent_output not in self.forward.output_names
            ):
                msg = "Observation channels require names and declared latent outputs"
                raise ValueError(msg)
            ArtifactReference.model_validate(channel.model.reference)

    @property
    def reference(self) -> ArtifactReference:
        """Identify the complete ordered pipeline, including statistical outputs.

        This supplements existing provenance; it never silently relabels an old
        bank, replaces its likelihood target or certifies supplied identities.
        """
        data = {
            "schema_version": 1,
            "forward_provenance": self.forward.provenance.identity,
            "prior": self.prior.reference.model_dump(mode="json"),
            "channels": [
                {
                    "name": channel.name,
                    "latent_output": channel.latent_output,
                    "model": channel.model.reference.model_dump(mode="json"),
                }
                for channel in self.channels
            ],
        }
        payload = json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return ArtifactReference.from_bytes(
            payload.encode(), kind="statistical-pipeline-v1"
        )

    def simulate(
        self,
        parameters: Mapping[str, object],
        design: ForwardDesign,
        streams: SimulationStreams,
    ) -> SimulationResult:
        """Evaluate latent outputs and independently address each noise channel."""
        SimulationStreams.model_validate(streams)
        latent = self.forward(parameters, design, streams.simulation)
        measurements = {}
        for channel in self.channels:
            key = RandomKey(
                seed=streams.seed,
                index=streams.index,
                stream=f"simulation/observation/{channel.name.encode('utf-8').hex()}",
            )
            measurements[channel.name] = channel.model.sample(
                latent[channel.latent_output], key
            )
        return SimulationResult(
            parameters, latent, measurements, streams, self.reference
        )

    def sample_numpy(
        self, design: ForwardDesign, streams: SimulationStreams
    ) -> SimulationResult:
        """Draw the prior then execute a NumPy-compatible forward; never retry."""
        parameters: dict[str, np.ndarray[Any, Any]] = self.prior.sample(
            streams.parameters
        )
        return self.simulate(parameters, design, streams)

    def log_prob(
        self,
        parameters: Mapping[str, object],
        design: ForwardDesign,
        key: RandomKey,
        observed: Mapping[str, object],
        *,
        masks: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Evaluate the same observation target used by simulation, without draws."""
        names = {channel.name for channel in self.channels}
        if set(observed) != names or (masks is not None and set(masks) != names):
            msg = "Observed and mask channel names must match the declared layout"
            raise ValueError(msg)
        latent = self.forward(parameters, design, key)
        return {
            channel.name: channel.model.log_prob(
                latent[channel.latent_output],
                observed[channel.name],
                mask=None if masks is None else masks[channel.name],
            )
            for channel in self.channels
        }
