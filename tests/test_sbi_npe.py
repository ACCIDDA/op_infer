"""Real CPU NPE, masked-cohort semantics and guarded tensor snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_infer.adapters.sbi import (
    NpeEnsemble,
    NpeFit,
    NpeSettings,
    build_masked_set_embedding,
    fit_npe,
    load_npe,
    npe_settings_reference,
    save_npe,
    sbi_backend_versions,
)
from op_infer.rng import RandomKey
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")


def _settings(index: int = 0, **kwargs: Any) -> NpeSettings:  # noqa: ANN401
    return NpeSettings(
        key=RandomKey(seed=73, stream="training", index=index),
        parameter_names=("theta", "latent"),
        n_max=3,
        feature_dim=1,
        hidden_features=8,
        latent_dim=4,
        embedding_dim=4,
        num_transforms=1,
        max_epochs=2,
        batch_size=32,
        **kwargs,
    )


def _provenance(settings: NpeSettings) -> Any:  # noqa: ANN401
    return provenance().model_copy(
        update={
            "estimator_settings": npe_settings_reference(settings),
            "backend_versions": sbi_backend_versions(device=settings.device),
        }
    )


def test_embedding_uses_each_cohorts_count_and_native_gradients() -> None:
    """Analytic mean+count values detect the sbi batched-count failure."""
    settings = NpeSettings(
        key=RandomKey(seed=1, stream="test"),
        parameter_names=("theta",),
        n_max=3,
        feature_dim=1,
        hidden_features=1,
        latent_dim=1,
        embedding_dim=1,
    )
    embedding = build_masked_set_embedding(settings).eval()
    with torch.no_grad():
        for module in embedding.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(1.0)
                module.bias.zero_()
    features = torch.tensor(
        [[[1.0], [3.0], [float("nan")]], [[2.0], [2.0], [2.0]]], requires_grad=True
    )
    result = embedding(features)
    np.testing.assert_allclose(
        result.detach().numpy()[:, 0], [2.0 + 2.0 / 3.0, 3.0], rtol=1e-6
    )
    torch.testing.assert_close(result, embedding(features[:, [2, 0, 1]]))
    result.sum().backward()
    np.testing.assert_allclose(features.grad.numpy()[0, :, 0], [0.5, 0.5, 0.0])


def test_augmentation_is_training_only_and_partial_missing_is_an_error() -> None:
    """A constant cohort isolates the training-only change in normalized count."""
    settings = _settings(mask_probability=1.0)
    embedding = build_masked_set_embedding(settings)
    with torch.no_grad():
        for module in embedding.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(1.0)
                module.bias.zero_()
    features = torch.ones((2, 3, 1))
    embedding.eval()
    before = embedding(features)
    embedding.train()
    augmented = embedding(features)
    assert bool((augmented < before).all())
    embedding.eval()
    torch.testing.assert_close(before, embedding(features))
    other = settings.model_copy(update={"feature_dim": 2})
    invalid = torch.ones((2, 3, 2))
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="partial NaNs"):
        build_masked_set_embedding(other)(invalid)


@pytest.fixture(scope="module")
def native_fits() -> tuple[tuple[NpeFit, ...], Tensor]:
    """Fit two small native flows; epoch limits are not calibration evidence."""
    generator = torch.Generator().manual_seed(9)
    theta = torch.randn((96, 2), generator=generator)
    observations = (
        theta[:, 0, None, None]
        + 0.3 * theta[:, 1, None, None]
        + torch.randn((96, 3, 1), generator=generator)
    )
    observations[::2, 1, :] = float("nan")
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    fits = []
    for index in (0, 1):
        settings = _settings(index)
        rng_before = torch.random.get_rng_state().clone()
        fit = fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
        )
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        assert fit.n_simulations == 96
        fits.append(fit)
    return tuple(fits), observations[0]


def test_equal_weight_seed_posterior_retains_latent_variables(
    native_fits: tuple[tuple[NpeFit, ...], Tensor],
) -> None:
    """Every training seed contributes exactly the declared independent draws."""
    fits, observed = native_fits
    ensemble = NpeEnsemble(fits)
    key = RandomKey(seed=73, stream="reporting", index=8)
    rng_before = torch.random.get_rng_state().clone()
    posterior = ensemble.sample(observed, per_seed=12, key=key)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    assert posterior.sample_axis == "seed"
    assert [v.name for v in posterior.draws] == ["theta", "latent"]
    assert all(v.values.shape == (2, 12) for v in posterior.draws)
    assert posterior.estimator_stage == "reporting"
    replay = ensemble.sample(observed, per_seed=12, key=key)
    for left, right in zip(posterior.draws, replay.draws, strict=True):
        torch.testing.assert_close(left.values, right.values, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="collide"):
        NpeEnsemble((fits[0], fits[0]))


def test_tensor_artifacts_replay_and_refuse_provenance_or_integrity_changes(
    native_fits: tuple[tuple[NpeFit, ...], Tensor],
    tmp_path: Path,
) -> None:
    """Weights-only reconstruction preserves draws and checks caller expectations."""
    fits, observed = native_fits
    fit = fits[0]
    path = tmp_path / "fit.pt"
    save_npe(fit, path)
    restored = load_npe(path, expected=fit.provenance)
    key = RandomKey(seed=73, stream="reporting", index=0)
    before = NpeEnsemble((fit,)).sample(observed, per_seed=8, key=key)
    after = NpeEnsemble((restored,)).sample(observed, per_seed=8, key=key)
    torch.testing.assert_close(
        before.draws[0].values, after.draws[0].values, rtol=0.0, atol=0.0
    )
    changed = fit.provenance.model_copy(update={"design": reference("another design")})
    with pytest.raises(ValueError, match="Provenance mismatch"):
        load_npe(path, expected=changed)
    payload = torch.load(path, weights_only=True)
    name = next(iter(payload["state"]))
    payload["state"][name] = payload["state"][name] + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="integrity"):
        load_npe(path, expected=fit.provenance)


def test_bounded_prior_and_incompatible_members_fail(
    native_fits: tuple[tuple[NpeFit, ...], Tensor],
) -> None:
    """Direct estimator sampling cannot silently bypass bounded prior support."""
    fits, _observed = native_fits
    settings = _settings()
    prior = torch.distributions.Independent(
        torch.distributions.Uniform(-torch.ones(2), torch.ones(2)), 1
    )
    with pytest.raises(ValueError, match="full-real"):
        fit_npe(
            torch.zeros((16, 2)),
            torch.zeros((16, 3, 1)),
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
        )
    different = fits[1].provenance.model_copy(
        update={"simulator": reference("different model")}
    )
    with pytest.raises(ValueError, match="Provenance mismatch"):
        NpeEnsemble((fits[0], replace(fits[1], provenance=different)))


@pytest.mark.parametrize(
    "device",
    [
        "cuda",
        "cuda:-1",
        "cuda:00",
        "cuda:1.5",
        "cuda:\u0660",
        "cuda:\uff10",
        "mps",
    ],
)
def test_settings_require_cpu_or_indexed_cuda(device: str) -> None:
    """Device identity cannot depend on an ambient CUDA selection."""
    with pytest.raises(ValueError, match="device"):
        _settings(device=device)


def test_default_cpu_serialization_remains_backward_compatible() -> None:
    """The new CPU default preserves existing settings and artifact identities."""
    assert "device" not in _settings().model_dump()
    assert _settings(device="cuda:0").model_dump()["device"] == "cuda:0"


def test_oversized_cuda_index_has_a_stable_adapter_error() -> None:
    """Backend parser limits cannot leak a backend-specific exception."""
    settings = _settings(device=f"cuda:{'9' * 100}")
    with pytest.raises(ValueError, match="Invalid NPE device"):
        sbi_backend_versions(device=settings.device)


def test_unavailable_cuda_device_fails_before_training() -> None:
    """An indexed but absent CUDA target cannot fall back to CPU."""
    index = torch.cuda.device_count()
    settings = _settings(device=f"cuda:{index}")
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        fit_npe(
            torch.zeros((16, 2)),
            torch.zeros((16, 3, 1)),
            prior=prior,
            settings=settings,
            provenance=_provenance(settings),
        )


def test_fixed_work_benchmark_executes_a_real_cpu_worker(tmp_path: Path) -> None:
    """The retained worker must execute native fitting rather than a skipped stub."""
    root = Path(__file__).parents[1]
    output = tmp_path / "worker.json"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(root / "examples" / "benchmark_npe_pipeline.py"),
            str(tmp_path / "bank"),
            "--worker",
            "--device",
            "cpu",
            "--storage",
            "mapped",
            "--stage",
            "train",
            "--worker-output",
            str(output),
            "--rows",
            "96",
            "--units",
            "3",
            "--features",
            "1",
            "--max-epochs",
            "1",
            "--batch-size",
            "32",
            "--sample-draws",
            "8",
            "--calibration-replicates",
            "2",
            "--calibration-draws",
            "8",
        ],
        check=True,
        cwd=root,
    )
    record = json.loads(output.read_text())
    assert record["completed_epochs"] == 2
    assert len(record["training_trace"]["train_epochs"]) == 2
    assert record["training_trace"]["split_digests"]
    assert record["workload"]["sample_draws"] == 8
    assert "op_infer/rng.py" in record["source_identity"]["manifest"]
    assert record["source_identity"]["reference"]["kind"] == "source-manifest-v1"
    assert record["draw_digest"].startswith("sha256:")
    assert record["calibration_timing_only"]["replicates"] == 2
    assert record["bank_digests_verified_after_execution"] == {
        "theta": record["manifest"]["theta"]["digest"],
        "observations": record["manifest"]["observations"]["digest"],
    }
