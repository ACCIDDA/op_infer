"""Guarded native training boundaries for the active large-bank NPE consumer.

Consumer: OP Infer's large-bank training request. The adapter observes sbi's
epoch-summary hook; native sbi owns optimization, losses and early stopping.
The private hook/state layout is qualified for sbi 0.27.0 only.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

    from op_infer.adapters.sbi import NpeSettings
    from op_infer.posterior import Provenance


def _digest(value: Any) -> str:  # noqa: ANN401
    """Hash primitive/tensor trees without serializing arbitrary Python objects.

    Raises:
        TypeError: A value is outside the checkpoint's primitive/tensor schema.
    """
    torch = importlib.import_module("torch")
    digest = hashlib.sha256()

    def visit(item: Any) -> None:  # noqa: ANN401
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(json.dumps(["tensor", array.shape, array.dtype.str]).encode())
            digest.update(memoryview(array).cast("B"))
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(
                item, key=lambda entry: (type(entry).__name__, str(entry))
            ):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for entry in item:
                visit(entry)
            digest.update(b"]")
        elif item is None or type(item) in (str, int, float, bool):
            digest.update(json.dumps([type(item).__name__, item]).encode() + b";")
        else:
            msg = "Unsupported training checkpoint value"
            raise TypeError(msg)

    visit(value)
    return "sha256:" + digest.hexdigest()


def bank_digest(theta: Tensor, observations: Tensor) -> str:
    """Hash ordered effective float32 training data in bounded contiguous chunks."""
    digest = hashlib.sha256()
    for name, tensor in (("theta", theta), ("observations", observations)):
        digest.update(
            json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        )
        # Bound the temporary contiguous copy to roughly eight MiB, or one row.
        rows = max(1, (8 * 1024 * 1024) // (tensor[0].numel() * tensor.element_size()))
        for start in range(0, len(tensor), rows):
            array = tensor[start : start + rows].detach().contiguous().numpy()
            digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    """Replace a checkpoint only after its complete tensor payload is written."""
    torch = importlib.import_module("torch")
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


def require_checkpoint_runtime() -> None:
    """Refuse unqualified private native training layouts.

    Raises:
        ValueError: The installed sbi version has not been qualified for resume.
    """
    if importlib.metadata.version("sbi") != "0.27.0":
        msg = "Training checkpoints currently require qualified sbi 0.27.0"
        raise ValueError(msg)


def borrow_simulations(native: Any, theta: Tensor, observations: Tensor) -> None:  # noqa: ANN401
    """Register one validated prior bank without sbi's filtering/concatenation copies.

    This uses the qualified single-round sbi 0.27.0 storage layout. Native
    dataloaders still choose the split and batches; only observations used to
    infer network dimensions are sliced before initialization. Native parameter
    standardization still uses all training theta rows and allocates that subset.
    """
    torch = importlib.import_module("torch")
    masks = torch.ones((len(theta), 1), dtype=torch.bool)
    native._data_round_index = [0]  # noqa: SLF001
    native._theta_roundwise = [theta]  # noqa: SLF001
    native._x_roundwise = [observations]  # noqa: SLF001
    native._prior_masks = [masks]  # noqa: SLF001
    native._proposal_roundwise = [None]  # noqa: SLF001

    def simulations(
        _owner: Any,  # noqa: ANN401
        starting_round: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if starting_round != 0:
            msg = "Borrowed NPE banks support one prior round only"
            raise ValueError(msg)
        return theta, observations, masks

    def initialize(
        owner: Any,  # noqa: ANN401
        retrain_from_scratch: bool,  # noqa: FBT001 - native method signature
        start_idx: int,
    ) -> None:
        if owner._neural_net is None or retrain_from_scratch:  # noqa: SLF001
            parameters, features, _ = simulations(owner, start_idx)
            owner._neural_net = owner._build_neural_net(  # noqa: SLF001
                parameters[owner.train_indices], features[owner.train_indices[:2]]
            )
            checks = importlib.import_module("sbi.utils.user_input_checks")
            checks.test_posterior_net_for_multi_d_x(
                owner._neural_net,  # noqa: SLF001
                parameters[:2].unsqueeze(0),
                features[:2],
            )

    native.get_simulations = MethodType(simulations, native)
    native._initialize_neural_network = MethodType(initialize, native)  # noqa: SLF001


_SCALARS = ("_val_loss", "_best_val_loss", "_epochs_since_last_improvement")


def restore_training(  # noqa: PLR0913
    native: Any,  # noqa: ANN401
    path: str | Path,
    *,
    settings: NpeSettings,
    provenance: Provenance,
    bank: str,
    n_simulations: int,
    until_epoch: int,
) -> None:
    """Restore a pre-finalization boundary after independent contract checks.

    Raises:
        ValueError: Schema, provenance, bank, integrity or native state differ.
    """
    torch = importlib.import_module("torch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "sbi-training-v1":
        msg = "Unknown NPE training checkpoint schema"
        raise ValueError(msg)
    state = payload.get("payload")
    if not isinstance(state, dict) or _digest(state) != payload.get("digest"):
        msg = "NPE training checkpoint integrity mismatch"
        raise ValueError(msg)
    provenance.load_json(state["provenance"], expected=provenance)
    if state["settings"] != settings.model_dump_json() or state["bank"] != bank:
        msg = "NPE training checkpoint settings or bank mismatch"
        raise ValueError(msg)
    epoch = state["epoch"]
    if type(epoch) is not int or not 0 < epoch < until_epoch:
        msg = "Resume requires a later cumulative epoch limit"
        raise ValueError(msg)
    indices = torch.cat((state["train_indices"], state["val_indices"]))
    if (
        indices.dtype != torch.int64
        or indices.ndim != 1
        or len(indices) != n_simulations
        or not torch.equal(indices.sort().values, torch.arange(n_simulations))
        or len(state["summary"]["validation_loss"]) != epoch
        or len(state["summary"]["training_loss"]) != epoch
    ):
        msg = "NPE training checkpoint split or epoch history is invalid"
        raise ValueError(msg)
    dummy_theta = torch.stack(
        (
            -torch.ones(len(settings.parameter_names)),
            torch.ones(len(settings.parameter_names)),
        )
    )
    dummy_x = torch.zeros((2, settings.n_max, settings.feature_dim))
    native._neural_net = native._build_neural_net(dummy_theta, dummy_x)  # noqa: SLF001
    native._neural_net.load_state_dict(state["network"], strict=True)  # noqa: SLF001
    native.optimizer = torch.optim.Adam(
        native._neural_net.parameters(),  # noqa: SLF001
        lr=settings.learning_rate,
    )
    native.optimizer.load_state_dict(state["optimizer"])
    native.epoch = epoch
    native.train_indices = state["train_indices"]
    native.val_indices = state["val_indices"]
    native._best_model_state_dict = state["best_network"]  # noqa: SLF001
    native._summary = state["summary"]  # noqa: SLF001
    for name in _SCALARS:
        setattr(native, name, state[name])
    torch.random.set_rng_state(state["torch_rng"])


@contextmanager
def training_checkpoints(  # noqa: PLR0913
    native: Any,  # noqa: ANN401
    *,
    path: str | Path | None,
    settings: NpeSettings,
    provenance: Provenance,
    bank: str,
    until_epoch: int,
    interval: int,
) -> Iterator[None]:
    """Observe native epoch summaries before best-weight restoration at the cap."""
    if path is None:
        yield
        return
    torch = importlib.import_module("torch")
    original = native._summarize_epoch  # noqa: SLF001

    def summarize(owner: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        original(*args, **kwargs)
        completed = int(owner.epoch) + 1
        if completed != 1 and completed % interval and completed != until_epoch:
            return
        state = {
            "settings": settings.model_dump_json(),
            "provenance": provenance.model_dump_json(),
            "bank": bank,
            "epoch": completed,
            "train_indices": owner.train_indices,
            "val_indices": owner.val_indices,
            "network": owner._neural_net.state_dict(),  # noqa: SLF001
            "best_network": owner._best_model_state_dict,  # noqa: SLF001
            "optimizer": owner.optimizer.state_dict(),
            "summary": owner._summary,  # noqa: SLF001
            "torch_rng": torch.random.get_rng_state(),
            **{name: getattr(owner, name) for name in _SCALARS},
        }
        atomic_save(
            {"schema": "sbi-training-v1", "payload": state, "digest": _digest(state)},
            path,
        )

    native._summarize_epoch = MethodType(summarize, native)  # noqa: SLF001
    try:
        yield
    finally:
        native._summarize_epoch = original  # noqa: SLF001
