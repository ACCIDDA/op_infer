"""Existing native fitting/checkpoint/posterior paths preserve structural identity."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from op_infer.adapters.sbi import NpeEnsemble, fit_npe, load_npe, save_npe
from op_infer.rng import RandomKey
from tests.test_provenance import reference
from tests.test_sbi_npe import _provenance, _settings

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")


def test_model_identity_survives_fit_resume_snapshot_and_posterior(
    tmp_path: Path,
) -> None:
    """A different model is refused by both native weight and training loaders."""
    generator = torch.Generator().manual_seed(29)
    theta = torch.randn((48, 2), generator=generator)
    observations = theta[:, :1, None] + torch.randn((48, 3, 1), generator=generator)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    settings = _settings().model_copy(update={"max_epochs": 3})
    model = reference(
        "synthetic two-parameter masked Gaussian bank model",
        kind="declared-callable-model-v1",
    )
    expected = _provenance(settings).with_structural_model(model)
    checkpoint = tmp_path / "training.pt"
    inputs = {"prior": prior, "settings": settings, "provenance": expected}
    first = fit_npe(
        theta, observations, **inputs, checkpoint_path=checkpoint, until_epoch=2
    )
    assert first.provenance.structural_model == model
    stored = torch.load(checkpoint, weights_only=True)
    assert json.loads(stored["payload"]["provenance"])[
        "structural_model"
    ] == model.model_dump(mode="json")
    resumed = fit_npe(theta, observations, **inputs, resume_from=checkpoint)
    path = tmp_path / "fit.pt"
    save_npe(resumed, path)
    restored = load_npe(path, expected=expected)
    assert restored.provenance.structural_model == model
    posterior = NpeEnsemble((restored,)).sample(
        observations[0], per_seed=4, key=RandomKey(seed=91, stream="reporting")
    )
    assert posterior.provenance.structural_model == model
    changed = expected.with_structural_model(
        reference("same dimensions, changed model")
    )
    with pytest.raises(ValueError, match="Provenance mismatch"):
        load_npe(path, expected=changed)
    with pytest.raises(ValueError, match="Provenance mismatch"):
        fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=changed,
            resume_from=checkpoint,
        )
