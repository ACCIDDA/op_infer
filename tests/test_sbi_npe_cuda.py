"""Actual single-device CUDA qualification for masked-cohort NPE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from op_infer.adapters.sbi import NpeEnsemble, fit_npe, load_npe, save_npe
from op_infer.rng import RandomKey
from tests.test_sbi_npe import _provenance, _settings

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an actual CUDA device"
)


def _bank(*, prior_device: str = "cpu") -> tuple[Any, Any, Any]:
    """Return a fixed CPU bank with unequal cohort sizes."""
    generator = torch.Generator().manual_seed(912)
    theta = torch.randn((96, 2), generator=generator)
    observations = (
        theta[:, :1, None]
        + 0.25 * theta[:, 1:, None]
        + torch.randn((96, 3, 1), generator=generator)
    )
    observations[::3, 1] = float("nan")
    prior = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(2, device=prior_device),
            torch.ones(2, device=prior_device),
        ),
        1,
    )
    return theta, observations, prior


@pytest.mark.parametrize("borrow_bank", [False, True])
def test_cuda_fit_sample_snapshot_and_rng_scope(
    tmp_path: Path, *, borrow_bank: bool
) -> None:
    """Training, draws and snapshots stay on CUDA and replay addressed streams."""
    theta, observations, prior = _bank(prior_device="cuda:0")
    settings = _settings(
        device="cuda:0",
        mask_probability=0.5,
    )
    expected = _provenance(settings)
    assert [version.name for version in expected.backend_versions[:2]] == [
        "cuda-device",
        "cuda-runtime",
    ]
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(0).clone()

    fit = fit_npe(
        theta,
        observations,
        prior=prior,
        settings=settings,
        provenance=expected,
        borrow_bank=borrow_bank,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_before)
    torch.testing.assert_close(torch.cuda.get_rng_state(0), cuda_before)
    assert {
        value.device.type for value in cast("Any", fit.estimator).state_dict().values()
    } == {"cuda"}

    key = RandomKey(seed=912, stream="cuda-reporting")
    ensemble = NpeEnsemble((fit,))
    first = ensemble.sample(observations[0], per_seed=8, key=key)
    replay = ensemble.sample(observations[0].to("cuda:0"), per_seed=8, key=key)
    assert all(variable.values.device.type == "cuda" for variable in first.draws)
    for left, right in zip(first.draws, replay.draws, strict=True):
        torch.testing.assert_close(left.values, right.values, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_before)
    torch.testing.assert_close(torch.cuda.get_rng_state(0), cuda_before)

    path = tmp_path / f"cuda-{borrow_bank}.pt"
    save_npe(fit, path)
    restored = load_npe(path, expected=expected)
    after = NpeEnsemble((restored,)).sample(observations[0], per_seed=8, key=key)
    for left, right in zip(first.draws, after.draws, strict=True):
        torch.testing.assert_close(left.values, right.values, rtol=0, atol=0)


def test_cuda_training_checkpoint_is_explicitly_out_of_scope(tmp_path: Path) -> None:
    """CPU-only continuation cannot silently write an incomplete CUDA boundary."""
    theta, observations, prior = _bank()
    settings = _settings(device="cuda:0")
    with pytest.raises(ValueError, match="checkpoints currently require"):
        fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
            checkpoint_path=tmp_path / "unsupported.pt",
        )


def test_cuda_fit_requires_a_device_matched_prior() -> None:
    """A CPU prior cannot be silently mixed with CUDA native training."""
    theta, observations, prior = _bank()
    settings = _settings(device="cuda:0")
    with pytest.raises(ValueError, match="prior must use the declared NPE device"):
        fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
        )
