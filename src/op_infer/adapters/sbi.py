"""Bounded native sbi NPE adapter for masked cohorts and seed ensembles.

This adapter wraps native sbi density estimation;
it implements neither flows nor posterior calibration. CPU float32, single-round
training and full-real parameter support are the qualified initial scope.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from op_infer.adapters._sbi_training import (
    bank_digest,
    borrow_simulations,
    require_checkpoint_runtime,
    restore_training,
    training_checkpoints,
)
from op_infer.adapters.rng import torch_generators
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    Coordinate,
    Posterior,
    Provenance,
    SampledVariable,
)
from op_infer.rng import RandomKey  # noqa: TC001 - Pydantic field resolution

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from torch import Tensor
    from torch.distributions import Distribution
    from torch.nn import Module


__all__ = [
    "NpeEnsemble",
    "NpeFit",
    "NpeSettings",
    "build_masked_set_embedding",
    "fit_npe",
    "load_npe",
    "npe_settings_reference",
    "save_npe",
    "sbi_backend_versions",
]


class NpeSettings(BaseModel):
    """Immutable architecture, training controls, named parameters and task key.

    Examples:
        >>> settings = NpeSettings(
        ...     key=RandomKey(seed=2026, stream="training"),
        ...     parameter_names=("theta", "latent"),
        ...     n_max=3,
        ...     feature_dim=2,
        ... )
        >>> settings.parameter_names, settings.n_max
        (('theta', 'latent'), 3)
        >>> npe_settings_reference(settings).kind
        'sbi-npe-settings-v1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    key: RandomKey
    parameter_names: tuple[str, ...]
    n_max: int = Field(ge=1)
    feature_dim: int = Field(ge=1)
    latent_dim: int = Field(default=16, ge=1)
    embedding_dim: int = Field(default=16, ge=1)
    hidden_features: int = Field(default=32, ge=1)
    num_transforms: int = Field(default=3, ge=1)
    mask_probability: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    batch_size: int = Field(default=128, ge=2)
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0, allow_inf_nan=False)
    max_epochs: int = Field(default=200, ge=1)
    stop_after_epochs: int = Field(default=20, ge=1)
    learning_rate: float = Field(default=5e-4, gt=0.0, allow_inf_nan=False)
    model: Literal["maf"] = "maf"

    @model_validator(mode="after")
    def _names(self) -> NpeSettings:
        """Return settings with a complete distinct scalar parameter layout.

        Raises:
            ValueError: Parameter names are missing or repeated.
        """
        if not self.parameter_names or len(set(self.parameter_names)) != len(
            self.parameter_names
        ):
            msg = "NPE requires distinct nonempty parameter names"
            raise ValueError(msg)
        for name in self.parameter_names:
            Coordinate(name=name, labels=(0,))
        return self


def _native() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load optional native Torch and sbi dependencies on first use.

    Returns:
        Torch, sbi inference, and sbi neural network modules.

    Raises:
        ImportError: The optional sbi extra is missing.
    """
    try:
        return (
            importlib.import_module("torch"),
            importlib.import_module("sbi.inference"),
            importlib.import_module("sbi.neural_nets"),
        )
    except ImportError as error:
        msg = "Install op_infer[sbi] for the native NPE adapter"
        raise ImportError(msg) from error


def sbi_backend_versions() -> tuple[BackendVersion, ...]:
    """Return exact runtime identities for guarded fitting and loading."""
    return tuple(
        BackendVersion(
            name=name,
            version=platform.python_version()
            if name == "python"
            else importlib.metadata.version(name),
        )
        for name in ("numpy", "python", "sbi", "torch")
    )


def npe_settings_reference(settings: NpeSettings) -> ArtifactReference:
    """Return the canonical architecture/control/key identity expected in provenance."""
    return ArtifactReference.from_bytes(
        settings.model_dump_json().encode(), kind="sbi-npe-settings-v1"
    )


@contextmanager
def _torch_scope(key: RandomKey) -> Iterator[None]:
    """Temporarily install one native CPU stream and restore global Torch state."""
    torch, _inference, _nets = _native()
    generator = torch_generators((key,))[0]
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(generator.get_state())
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            yield
        finally:
            torch.set_default_dtype(dtype)


def _presence(features: Tensor, settings: NpeSettings) -> Tensor:
    """Return per-cohort presence while refusing partial NaN rows and infinities.

    Raises:
        ValueError: The design shape or absence encoding is invalid.
    """
    torch, _inference, _nets = _native()
    if features.ndim != 3 or tuple(features.shape[-2:]) != (
        settings.n_max,
        settings.feature_dim,
    ):
        msg = "Expected (batch, design units, features) for this NPE design"
        raise ValueError(msg)
    missing = torch.isnan(features)
    absent = missing.all(-1)
    if bool((missing.any(-1) & ~absent).any()) or bool(torch.isinf(features).any()):
        msg = (
            "Missing units require all-NaN rows; "
            "partial NaNs and infinities are invalid"
        )
        raise ValueError(msg)
    return cast("Tensor", ~absent)


def build_masked_set_embedding(settings: NpeSettings) -> Module:
    """Build per-cohort mean/count pooling with training-only unit removal.

    Returns:
        A native Torch module mapping (batch, units, features) to embeddings.
        State dictionaries are portable through this factory; no local class
        object is serialized.
    """
    torch, _inference, _nets = _native()

    class MaskedSetEmbedding(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            options = {"dtype": torch.float32, "device": "cpu"}
            self.unit = torch.nn.Sequential(
                torch.nn.Linear(
                    settings.feature_dim, settings.hidden_features, **options
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    settings.hidden_features, settings.latent_dim, **options
                ),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(
                    settings.latent_dim + 1, settings.embedding_dim, **options
                ),
                torch.nn.ReLU(),
            )

        def forward(self, features: Tensor) -> Tensor:
            """Embed each cohort using its own presence count."""
            present = _presence(features, settings)
            if self.training and settings.mask_probability > 0:
                remove = (
                    torch.rand(
                        len(features), device=features.device, dtype=torch.float32
                    )
                    < settings.mask_probability
                ) & (present.sum(-1) > 1)
                scores = torch.rand(
                    present.shape, device=features.device, dtype=torch.float32
                ).masked_fill(~present, -1.0)
                victim = scores.argmax(-1)
                present = present.clone()
                present[
                    torch.arange(len(features), device=features.device)[remove],
                    victim[remove],
                ] = False
            weights = present.unsqueeze(-1).to(features.dtype)
            encoded = self.unit(torch.nan_to_num(features, nan=0.0)) * weights
            count = weights.sum(1)
            pooled = encoded.sum(1) / count.clamp_min(1.0)
            return cast(
                "Tensor", self.head(torch.cat((pooled, count / settings.n_max), dim=-1))
            )

    return cast("Module", MaskedSetEmbedding())


@dataclass(frozen=True)
class NpeFit:
    """Native density estimator with its exact fit identity; no gate verdict."""

    estimator: object = field(repr=False)
    settings: NpeSettings
    provenance: Provenance
    n_simulations: int
    epochs: int


def _weights_digest(weights: dict[str, Any]) -> str:
    """Return a complete native tensor-state integrity digest."""
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        array = value.detach().cpu().numpy()
        digest.update(str((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def _state_digest(estimator: object) -> str:
    """Return a deterministic native state-dictionary integrity digest."""
    return _weights_digest(cast("Any", estimator).state_dict())


def _require_provenance(settings: NpeSettings, provenance: Provenance) -> None:
    """Check declared estimator settings and exact installed backend identities.

    Raises:
        ValueError: Runtime or settings differ from the caller's provenance.
    """
    if (
        provenance.estimator_settings != npe_settings_reference(settings)
        or provenance.backend_versions != sbi_backend_versions()
    ):
        msg = "NPE provenance settings or backend versions mismatch"
        raise ValueError(msg)


def _prepare_bank(
    theta: Tensor,
    observations: Tensor,
    prior: Distribution,
    settings: NpeSettings,
    *,
    borrow_bank: bool,
) -> tuple[Tensor, Tensor]:
    """Validate the effective training bank in bounded chunks.

    Raises:
        ValueError: Array layout, prior support or declared absence encoding differ.
    """
    torch, _inference, _nets = _native()
    if type(borrow_bank) is not bool:
        msg = "borrow_bank must be a boolean"
        raise ValueError(msg)
    if borrow_bank and any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or value.requires_grad
        for value in (theta, observations)
    ):
        msg = "Borrowed banks require contiguous CPU float32 tensors without gradients"
        raise ValueError(msg)
    theta = torch.as_tensor(theta, dtype=torch.float32, device="cpu")
    observations = torch.as_tensor(observations, dtype=torch.float32, device="cpu")
    if (
        theta.ndim != 2
        or theta.shape != (len(observations), len(settings.parameter_names))
        or len(theta) < 10
    ):
        msg = "NPE needs at least ten finite, aligned named parameter/observation rows"
        raise ValueError(msg)
    support: Any = prior.support
    while hasattr(support, "base_constraint"):
        support = support.base_constraint
    if support is not torch.distributions.constraints.real:
        msg = (
            "Masked direct NPE requires full-real prior support; "
            "transform bounded parameters first"
        )
        raise ValueError(msg)
    # Validation must not allocate boolean masks for the whole mapped bank.
    rows = max(
        1,
        (8 * 1024 * 1024)
        // (
            (settings.n_max * settings.feature_dim + len(settings.parameter_names)) * 4
        ),
    )
    for start in range(0, len(theta), rows):
        parameters = theta[start : start + rows]
        _presence(observations[start : start + rows], settings)
        if not bool(torch.isfinite(parameters).all()) or not bool(
            torch.isfinite(prior.log_prob(parameters)).all()
        ):
            msg = "Training parameters must lie in the supplied prior's finite support"
            raise ValueError(msg)
    return theta, observations


def fit_npe(  # noqa: PLR0913
    theta: Tensor,
    observations: Tensor,
    *,
    prior: Distribution,
    settings: NpeSettings,
    provenance: Provenance,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    until_epoch: int | None = None,
    checkpoint_interval: int = 1,
    borrow_bank: bool = False,
) -> NpeFit:
    """Train one native sbi NPE on a prespecified, prior-coherent simulation bank.

    Used by masked-cohort design workflows. The caller
    owns the prior/observation/acceptance provenance. Full-real parameter support
    is required because masked inference samples the estimator directly, bypassing
    DirectPosterior's prior-support rejection. Transform bounded parameters before
    fitting and record that transformation in prior/parameter provenance.

    ``checkpoint_path`` saves resumable native training boundaries, distinct from
    ``save_npe`` inference snapshots. ``resume_from`` requires the same ordered
    effective training bank, complete provenance, settings and runtime. The
    optional ``until_epoch`` is a cumulative completed-epoch limit for this call;
    the original native budget permits ``settings.max_epochs + 1`` epochs.
    Chunk boundaries do not change the declared full budget or early-stop rule.
    Checkpoints require sbi 0.27.0 and serialized CPU calls, like native fitting.
    ``borrow_bank=True`` retains contiguous CPU float32 tensors (including native
    tensors backed by copy-on-write NumPy memory maps) without full observation
    copies. Keep their storage immutable for the duration of fitting. Native
    parameter standardization still materializes the training theta subset.

    Args:
        theta: Aligned simulation-by-parameter training targets.
        observations: Simulation-by-unit-by-feature data, with all-NaN absent units.
        prior: Native full-real-support prior used to generate the training bank.
        settings: Fixed architecture, full native epoch budget and training key.
        provenance: Independently supplied scientific/settings/runtime contract.
        checkpoint_path: Optional atomic output for resumable training boundaries.
        resume_from: Previous training checkpoint, requiring the same effective bank.
        until_epoch: Optional cumulative completed-epoch limit, at least two.
        checkpoint_interval: Positive interval between checkpointed complete epochs;
            the first epoch and requested chunk limit are also saved.
        borrow_bank: Retain original CPU float32 observation storage without full
            bank copies; requires contiguous tensors and qualified sbi 0.27.0.

    Returns:
        Native CPU float32 fit, retaining named variables and exact provenance.

    Raises:
        ValueError: Data, prior support or provenance do not meet this contract.
    """
    _torch, inference, nets = _native()
    _require_provenance(settings, provenance)
    limit = settings.max_epochs + 1 if until_epoch is None else until_epoch
    if (
        type(limit) is not int
        or not 2 <= limit <= settings.max_epochs + 1
        or type(checkpoint_interval) is not int
        or checkpoint_interval < 1
        or (until_epoch is not None and checkpoint_path is None)
    ):
        msg = (
            "Use integer epoch controls within the full budget, until_epoch >= 2 "
            "and a checkpoint path"
        )
        raise ValueError(msg)
    checkpointed = checkpoint_path is not None or resume_from is not None
    if checkpointed or borrow_bank:
        require_checkpoint_runtime()
    theta, observations = _prepare_bank(
        theta, observations, prior, settings, borrow_bank=borrow_bank
    )
    bank = bank_digest(theta, observations) if checkpointed else ""
    with (
        tempfile.TemporaryDirectory(prefix="op-infer-npe-") as logs,
        _torch_scope(settings.key),
        closing(
            importlib.import_module("torch.utils.tensorboard").SummaryWriter(logs)
        ) as writer,
    ):
        embedding = build_masked_set_embedding(settings)
        builder = nets.posterior_nn(
            model=settings.model,
            embedding_net=embedding,
            hidden_features=settings.hidden_features,
            num_transforms=settings.num_transforms,
            z_score_theta="independent",
            z_score_x="none",
        )
        native = inference.NPE(
            prior=prior,
            density_estimator=builder,
            device="cpu",
            show_progress_bars=False,
            summary_writer=writer,
        )
        if borrow_bank:
            borrow_simulations(native, theta, observations)
        else:
            native.append_simulations(theta, observations, exclude_invalid_x=False)
        if len(native.get_simulations()[0]) != len(theta):
            msg = "Native sbi unexpectedly excluded simulation rows"
            raise RuntimeError(msg)
        if resume_from is not None:
            restore_training(
                native,
                resume_from,
                settings=settings,
                provenance=provenance,
                bank=bank,
                n_simulations=len(theta),
                until_epoch=limit,
            )
        with training_checkpoints(
            native,
            path=checkpoint_path,
            settings=settings,
            provenance=provenance,
            bank=bank,
            until_epoch=limit,
            interval=checkpoint_interval,
        ):
            estimator = native.train(
                training_batch_size=settings.batch_size,
                validation_fraction=settings.validation_fraction,
                max_num_epochs=limit - 1,
                stop_after_epochs=settings.stop_after_epochs,
                learning_rate=settings.learning_rate,
                show_train_summary=False,
                resume_training=resume_from is not None,
            )
        estimator.eval()
    return NpeFit(estimator, settings, provenance, len(theta), int(native.epoch))


def _ensemble_reference(members: tuple[NpeFit, ...]) -> ArtifactReference:
    """Return the ordered native member/settings/state identities."""
    payload = [
        {
            "settings": m.settings.model_dump(mode="json"),
            "provenance": m.provenance.identity,
            "state": _state_digest(m.estimator),
        }
        for m in members
    ]
    return ArtifactReference.from_bytes(
        json.dumps(payload, sort_keys=True).encode(), kind="sbi-seed-ensemble-v1"
    )


@dataclass(frozen=True)
class NpeEnsemble:
    """Equal-weight native seed members, distinct from MCMC chains.

    Members must agree on prior, model, design, code, target and all settings
    except the training key. Gate-driven retraining creates a new ensemble and
    provenance; it never turns a failed calibration result into a pass by itself.
    """

    members: tuple[NpeFit, ...]

    def __post_init__(self) -> None:
        """Validate the complete common scientific and training contract.

        Raises:
            ValueError: Members are absent, mismatched or share effective seeds.
        """
        if not isinstance(self.members, tuple) or not self.members:
            msg = "NPE ensembles require a nonempty immutable tuple of members"
            raise ValueError(msg)
        first = self.members[0]
        controls = first.settings.model_dump(exclude={"key"})
        for member in self.members:
            _require_provenance(member.settings, member.provenance)
            if member.settings.model_dump(exclude={"key"}) != controls:
                msg = "NPE ensemble settings mismatch"
                raise ValueError(msg)
            member.provenance.require_match(
                first.provenance.model_copy(
                    update={
                        "estimator_settings": member.provenance.estimator_settings,
                    }
                )
            )
        torch_generators(tuple(m.settings.key for m in self.members))

    def sample(
        self,
        observation: Tensor,
        *,
        per_seed: int,
        key: RandomKey,
    ) -> Posterior[Tensor]:
        """Draw exactly per_seed samples from every native estimator.

        Returns:
            Native named posterior with separate seed/draw coordinates and
            conditioning/member/sampling provenance. Latent-state variables can
            be named parameters and are retained for subsequent forecasting.

        Raises:
            ValueError: Observation layout or draw count is invalid.
        """
        torch, _inference, _nets = _native()
        settings = self.members[0].settings
        if isinstance(per_seed, bool) or not isinstance(per_seed, int) or per_seed < 2:
            msg = "per_seed must be an integer >= 2"
            raise ValueError(msg)
        condition = torch.as_tensor(observation, dtype=torch.float32, device="cpu")
        if condition.ndim != 2:
            msg = "Sample one cohort (design units, features) at a time"
            raise ValueError(msg)
        torch_generators((*tuple(m.settings.key for m in self.members), key))
        condition = condition.unsqueeze(0)
        _presence(condition, settings)
        parts = []
        with _torch_scope(key), torch.no_grad():
            for member in self.members:
                estimator = cast("Any", member.estimator).eval()
                parts.append(
                    estimator.sample((per_seed,), condition=condition).reshape(
                        per_seed, len(settings.parameter_names)
                    )
                )
        values = torch.stack(parts)
        if not bool(torch.isfinite(values).all()):
            msg = "NPE produced non-finite draws; retain a failed fit/sample record"
            raise ValueError(msg)
        references = {
            "ensemble": _ensemble_reference(self.members).model_dump(mode="json"),
            "sample_key": key.model_dump(mode="json"),
            "per_seed": per_seed,
            "condition": hashlib.sha256(condition.numpy().tobytes()).hexdigest(),
        }
        provenance = self.members[0].provenance.model_copy(
            update={
                "estimator_stage": "reporting",
                "estimator_settings": ArtifactReference.from_bytes(
                    json.dumps(references, sort_keys=True).encode(),
                    kind="sbi-reporting-samples-v1",
                ),
            }
        )
        return Posterior(
            tuple(
                SampledVariable(name, values[:, :, i], ())
                for i, name in enumerate(settings.parameter_names)
            ),
            (
                Coordinate(
                    name="seed",
                    labels=tuple(
                        m.settings.key.model_dump_json() for m in self.members
                    ),
                ),
                Coordinate(name="draw", labels=tuple(range(per_seed))),
            ),
            provenance,
            sample_axis="seed",
        )


def save_npe(fit: NpeFit, path: str | Path) -> None:
    """Save state tensors and primitive metadata, never a native estimator object."""
    torch, _inference, _nets = _native()
    _require_provenance(fit.settings, fit.provenance)
    payload = {
        "schema_version": 1,
        "settings": fit.settings.model_dump_json(),
        "provenance": fit.provenance.model_dump_json(),
        "n_simulations": fit.n_simulations,
        "epochs": fit.epochs,
        "state_digest": _state_digest(fit.estimator),
        "state": cast("Any", fit.estimator).state_dict(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, scratch = tempfile.mkstemp(
        dir=destination.parent, prefix="npe-", suffix=".tmp.pt"
    )
    os.close(descriptor)
    try:
        torch.save(payload, scratch)
        Path(scratch).replace(destination)
    finally:
        Path(scratch).unlink(missing_ok=True)


def load_npe(path: str | Path, *, expected: Provenance) -> NpeFit:
    """Load tensor-only weights after independently supplied provenance matches.

    Returns:
        Reconstructed native estimator in evaluation mode.

    Raises:
        ValueError: Schema, provenance, runtime, settings or weight integrity differ.
    """
    torch, _inference, nets = _native()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        msg = "Unknown NPE artifact schema"
        raise ValueError(msg)
    provenance = Provenance.load_json(payload["provenance"], expected=expected)
    if _weights_digest(payload["state"]) != payload["state_digest"]:
        msg = "NPE state integrity mismatch"
        raise ValueError(msg)
    settings = NpeSettings.model_validate_json(payload["settings"])
    _require_provenance(settings, provenance)
    with _torch_scope(settings.key):
        embedding = build_masked_set_embedding(settings)
        builder = nets.posterior_nn(
            model=settings.model,
            embedding_net=embedding,
            hidden_features=settings.hidden_features,
            num_transforms=settings.num_transforms,
            z_score_theta="independent",
            z_score_x="none",
        )
        dummy_theta = torch.stack(
            (
                -torch.ones(len(settings.parameter_names)),
                torch.ones(len(settings.parameter_names)),
            )
        )
        dummy_x = torch.zeros(
            (2, settings.n_max, settings.feature_dim), dtype=torch.float32
        )
        estimator = builder(dummy_theta, dummy_x)
        estimator.load_state_dict(payload["state"], strict=True)
        estimator.eval()
    if _state_digest(estimator) != payload["state_digest"]:
        msg = "NPE state integrity mismatch"
        raise ValueError(msg)
    return NpeFit(
        estimator,
        settings,
        provenance,
        int(payload["n_simulations"]),
        int(payload["epochs"]),
    )
