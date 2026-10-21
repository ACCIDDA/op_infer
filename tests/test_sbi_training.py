"""Native training continuation against uninterrupted optimization, not a mock loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from op_infer.adapters._sbi_training import atomic_save, borrow_simulations
from op_infer.adapters.sbi import fit_npe
from tests.test_provenance import reference
from tests.test_sbi_npe import _provenance, _settings

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")


@pytest.fixture
def bank() -> tuple[Any, Any, Any]:
    """Create a small bank with unequal cohort sizes and training augmentation."""
    generator = torch.Generator().manual_seed(19)
    theta = torch.randn((96, 2), generator=generator)
    observations = theta[:, :1, None] + torch.randn((96, 3, 1), generator=generator)
    observations[::3, 1] = float("nan")
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    return theta, observations, prior


@pytest.mark.parametrize("borrow_bank", [False, True])
def test_checkpoint_continuation_matches_uninterrupted_native_training(
    bank: tuple[Any, Any, Any], tmp_path: Path, *, borrow_bank: bool
) -> None:
    """Weights, optimizer moments, split, losses and RNG agree across boundaries."""
    theta, observations, prior = bank
    settings = _settings(mask_probability=0.5).model_copy(update={"max_epochs": 5})
    inputs = {
        "prior": prior,
        "settings": settings,
        "provenance": _provenance(settings),
        "borrow_bank": borrow_bank,
    }
    full_path, chunk_path = tmp_path / "full.pt", tmp_path / "chunks.pt"
    before = torch.random.get_rng_state().clone()
    full = fit_npe(theta, observations, **inputs, checkpoint_path=full_path)
    chunk = fit_npe(
        theta, observations, **inputs, checkpoint_path=chunk_path, until_epoch=2
    )
    assert chunk.epochs == 2
    # Ambient draws and inference from a chunk cannot change training continuation.
    torch.rand(7)
    ambient = torch.random.get_rng_state().clone()
    resumed = fit_npe(
        theta,
        observations,
        **inputs,
        resume_from=chunk_path,
        checkpoint_path=chunk_path,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), ambient)
    for name, tensor in cast("Any", full.estimator).state_dict().items():
        torch.testing.assert_close(
            tensor, cast("Any", resumed.estimator).state_dict()[name], rtol=0, atol=0
        )
    a = torch.load(full_path, weights_only=True)["payload"]
    b = torch.load(chunk_path, weights_only=True)["payload"]
    for name in ("train_indices", "val_indices", "torch_rng"):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert a["epoch"] == b["epoch"] == 6
    assert a["summary"]["training_loss"] == b["summary"]["training_loss"]
    assert a["summary"]["validation_loss"] == b["summary"]["validation_loss"]
    assert a["_best_val_loss"] == b["_best_val_loss"]
    assert a["_epochs_since_last_improvement"] == b["_epochs_since_last_improvement"]
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    torch.random.set_rng_state(before)


def test_checkpoint_refuses_changed_bank_provenance_and_corruption(
    bank: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    """Content identity covers the whole ordered bank and optimizer state."""
    theta, observations, prior = bank
    settings = _settings()
    inputs = {"prior": prior, "settings": settings, "provenance": _provenance(settings)}
    path = tmp_path / "state.pt"
    fit_npe(theta, observations, **inputs, checkpoint_path=path, until_epoch=2)
    changed = observations.clone()
    changed[-1, -1, -1] += 1
    with pytest.raises(ValueError, match="bank mismatch"):
        fit_npe(theta, changed, **inputs, resume_from=path)
    with pytest.raises(ValueError, match="settings or bank"):
        fit_npe(theta.flip(0), observations.flip(0), **inputs, resume_from=path)
    changed_inputs = inputs | {
        "provenance": inputs["provenance"].model_copy(
            update={"design": reference("another design")}
        )
    }
    with pytest.raises(ValueError, match="Provenance mismatch"):
        fit_npe(theta, observations, **changed_inputs, resume_from=path)
    payload = torch.load(path, weights_only=True)
    optimizer = payload["payload"]["optimizer"]["state"]
    next(iter(optimizer.values()))["exp_avg"].add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="integrity"):
        fit_npe(theta, observations, **inputs, resume_from=path)


def test_borrowed_memory_map_matches_native_storage(
    bank: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    """Borrowed observation storage preserves native masks, batches and training."""
    theta, observations, prior = bank
    settings = _settings(mask_probability=0.5)
    inputs = {"prior": prior, "settings": settings, "provenance": _provenance(settings)}
    path = tmp_path / "observations.npy"
    np.save(path, observations.numpy())
    mapped = np.load(path, mmap_mode="c")
    borrowed = torch.from_numpy(mapped)
    native_fit = fit_npe(theta, observations, **inputs)
    borrowed_fit = fit_npe(theta, borrowed, **inputs, borrow_bank=True)
    for name, tensor in cast("Any", native_fit.estimator).state_dict().items():
        torch.testing.assert_close(
            tensor,
            cast("Any", borrowed_fit.estimator).state_dict()[name],
            rtol=0,
            atol=0,
        )
    np.testing.assert_array_equal(mapped, observations.numpy())
    assert borrowed.data_ptr() == mapped.ctypes.data
    # The native dataloader must read those original buffers, rather than a cat.
    from sbi.inference import NPE  # noqa: PLC0415

    writer = torch.utils.tensorboard.SummaryWriter(str(tmp_path / "logs"))
    owner = NPE(prior=prior, show_progress_bars=False, summary_writer=writer)
    borrow_simulations(owner, theta, borrowed)
    assert owner.get_simulations()[1].data_ptr() == borrowed.data_ptr()
    train, validation = owner.get_dataloaders(training_batch_size=32)
    assert train.dataset.tensors[1].data_ptr() == borrowed.data_ptr()
    assert validation.dataset.tensors[1].data_ptr() == borrowed.data_ptr()
    writer.close()


def test_interrupted_native_epoch_leaves_resumable_previous_boundary(
    bank: tuple[Any, Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete epoch cannot replace a complete atomic checkpoint."""
    from sbi.inference import NPE  # noqa: PLC0415

    theta, observations, prior = bank
    settings = _settings(mask_probability=0.5).model_copy(update={"max_epochs": 4})
    inputs = {"prior": prior, "settings": settings, "provenance": _provenance(settings)}
    full = fit_npe(theta, observations, **inputs)
    original = NPE._train_epoch  # noqa: SLF001

    def interrupt(owner: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if owner.epoch == 2:
            msg = "synthetic preemption"
            raise InterruptedError(msg)
        return original(owner, *args, **kwargs)

    path = tmp_path / "preempted.pt"
    before = torch.random.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(NPE, "_train_epoch", interrupt)
        with pytest.raises(InterruptedError, match="preemption"):
            fit_npe(theta, observations, **inputs, checkpoint_path=path)
    torch.testing.assert_close(before, torch.random.get_rng_state())
    assert torch.load(path, weights_only=True)["payload"]["epoch"] == 2
    resumed = fit_npe(theta, observations, **inputs, resume_from=path)
    for name, tensor in cast("Any", full.estimator).state_dict().items():
        torch.testing.assert_close(
            tensor, cast("Any", resumed.estimator).state_dict()[name], rtol=0, atol=0
        )


def test_native_early_stopping_is_preserved_across_checkpoint(
    bank: tuple[Any, Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native patience counter, rather than chunk count, controls stopping."""
    from sbi.inference import NPE  # noqa: PLC0415

    theta, observations, prior = bank
    settings = _settings().model_copy(update={"max_epochs": 10, "stop_after_epochs": 2})
    inputs = {"prior": prior, "settings": settings, "provenance": _provenance(settings)}

    def worsening(owner: Any, *_args: Any) -> float:  # noqa: ANN401
        return float(owner.epoch + 1)

    monkeypatch.setattr(NPE, "_validate_epoch", worsening)
    full = fit_npe(theta, observations, **inputs)
    path = tmp_path / "early.pt"
    fit_npe(theta, observations, **inputs, checkpoint_path=path, until_epoch=2)
    resumed = fit_npe(theta, observations, **inputs, resume_from=path)
    assert full.epochs == resumed.epochs == 3
    for name, tensor in cast("Any", full.estimator).state_dict().items():
        torch.testing.assert_close(
            tensor, cast("Any", resumed.estimator).state_dict()[name], rtol=0, atol=0
        )


def test_failed_checkpoint_write_retains_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk or interrupted writer cannot replace the complete checkpoint."""
    path = tmp_path / "state.pt"
    atomic_save({"epoch": 2}, path)
    original = path.read_bytes()

    def failed_save(_payload: object, scratch: str) -> None:
        from pathlib import Path  # noqa: PLC0415

        Path(scratch).write_bytes(b"incomplete")
        msg = "synthetic storage error"
        raise OSError(msg)

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="storage error"):
        atomic_save({"epoch": 3}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["float64", "strided", "gradient"])
def test_borrowed_banks_refuse_implicit_conversion_or_gradients(
    bank: tuple[Any, Any, Any], kind: str
) -> None:
    """Borrowing cannot silently allocate a converted full bank."""
    theta, observations, prior = bank
    if kind == "float64":
        observations = observations.double()
    elif kind == "strided":
        observations = observations.repeat_interleave(2, dim=-1)[..., ::2]
    else:
        observations = observations.requires_grad_()
    settings = _settings()
    with pytest.raises(ValueError, match="contiguous CPU float32"):
        fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
            borrow_bank=True,
        )


@pytest.mark.parametrize("limit", [0, 1, True, 1.5, 10])
def test_invalid_cumulative_epoch_limits_fail(
    bank: tuple[Any, Any, Any],
    tmp_path: Path,
    limit: Any,  # noqa: ANN401
) -> None:
    """Epoch limits cannot silently change the prespecified total budget."""
    theta, observations, prior = bank
    settings = _settings()
    with pytest.raises(ValueError, match="epoch controls"):
        fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
            until_epoch=limit,
            checkpoint_path=tmp_path / "unused.pt",
        )
