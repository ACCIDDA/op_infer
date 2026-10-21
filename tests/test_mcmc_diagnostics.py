"""Named MCMC diagnostics against the mature reference, retaining failed quantities."""

from __future__ import annotations

import hashlib
import importlib
import json
import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.adapters.arviz import MCMCDiagnostics, diagnose_mcmc
from op_infer.posterior import ArtifactReference, Coordinate, Posterior, SampledVariable
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

pytest.importorskip("arviz_stats")
array_stats = importlib.import_module("arviz_stats.base").array_stats

DEFINITION = reference("reported raw theta and ordered vector; all post-warmup draws")


def sample_posterior(
    values: NDArray[np.generic], *, axis: Literal["chain", "seed"] = "chain"
) -> Posterior[NDArray[np.generic]]:
    """Build one scalar/vector variable with explicit non-sorted coordinates."""
    dims = () if values.ndim == 2 else ("component",)
    coordinates: tuple[Coordinate, ...] = (
        Coordinate(name=axis, labels=tuple(range(values.shape[0], 0, -1))),
        Coordinate(name="draw", labels=tuple(range(values.shape[1]))),
    )
    if dims:
        coordinates += (
            Coordinate(
                name="component", labels=tuple(f"c{i}" for i in range(values.shape[2]))
            ),
        )
    return Posterior(
        draws=(SampledVariable("theta", values, dims),),
        coordinates=coordinates,
        provenance=provenance(),
        sample_axis=axis,
    )


def report(values: NDArray[np.generic]) -> MCMCDiagnostics:
    """Use the public explicit quantity/definition boundary."""
    return diagnose_mcmc(
        sample_posterior(values), quantities=("theta",), definition=DEFINITION
    )


def test_named_vector_axes_reference_and_snapshot_identity() -> None:
    """Every event element matches the reference with original chain/draw axes."""
    rng = np.random.default_rng(235)
    values = np.asfortranarray(rng.normal(size=(4, 501, 2)))
    original = values.copy()
    posterior = sample_posterior(values)
    result = diagnose_mcmc(posterior, quantities=("theta",), definition=DEFINITION)
    assert result.coordinates == posterior.coordinates
    assert result.quantities[0].dims == ("component",)
    assert tuple(c.index for c in result.quantities[0].components) == ((0,), (1,))
    assert result.n_components == 2
    assert result.n_incomplete == 0
    assert result.adequacy_status == "not_assessed"
    for i, component in enumerate(result.quantities[0].components):
        trace = values[:, :, i]
        assert component.rank_rhat.value == pytest.approx(
            float(array_stats.rhat(trace, chain_axis=0, draw_axis=1, method="rank"))
        )
        assert component.bulk_ess.value == pytest.approx(
            float(
                array_stats.ess(
                    trace, chain_axis=0, draw_axis=1, method="bulk", relative=False
                )
            )
        )
        assert component.tail_ess.value == pytest.approx(
            float(
                array_stats.ess(
                    trace,
                    chain_axis=0,
                    draw_axis=1,
                    method="tail",
                    relative=False,
                    prob=(0.05, 0.95),
                )
            )
        )
    expected = ArtifactReference(
        kind="numpy-c-little-endian-v1",
        digest="sha256:"
        + hashlib.sha256(
            values.astype(values.dtype.newbyteorder("<")).tobytes(order="C")
        ).hexdigest(),
    )
    assert result.quantities[0].content == expected
    np.testing.assert_array_equal(values, original)
    values[:] = np.nan
    assert result.n_incomplete == 0
    assert result.quantities[0].content == expected
    versions = {v.name: v.version for v in result.backend_versions}
    assert set(versions) == {"op_infer", "arviz-stats", "numpy", "scipy", "python"}
    assert result.implementation_reference.kind == "op-infer-arviz-adapter-source-v1"
    assert (
        MCMCDiagnostics.load_json(
            result.model_dump_json(),
            expected=posterior.provenance,
            definition=DEFINITION,
        )
        == result
    )


def test_original_single_chain_and_insufficient_draws() -> None:
    """Splitting a single original chain cannot make R-hat available."""
    values = np.random.default_rng(47).normal(size=(1, 100))
    single = report(values).quantities[0].components[0]
    assert single.rank_rhat.value is None
    assert single.rank_rhat.reason == "single_original_chain"
    assert single.bulk_ess.value is not None
    assert single.tail_ess.value is not None
    few = report(values[:, :3])
    assert few.n_incomplete == 1
    assert few.quantities[0].components[0].bulk_ess.reason == "fewer_than_four_draws"


def test_constants_atoms_and_nonfinite_elements_stay_in_denominator() -> None:
    """Exact point masses, NaNs and Infs cannot disappear or produce a pass."""
    values = np.random.default_rng(14).normal(size=(4, 100, 4))
    values[:, :, 1] = 0
    values[0, 0, 2] = np.nan
    values[1, 5, 2] = np.inf
    values[:, :, 3] = np.arange(4)[
        :, None
    ]  # Chains individually stuck at different constants.
    result = report(values)
    assert result.n_components == 4
    assert result.n_incomplete == 3
    good, constant, nonfinite, stuck = result.quantities[0].components
    assert good.rank_rhat.status == "available"
    assert constant.rank_rhat.status == "undefined"
    assert constant.rank_rhat.reason == "constant_quantity"
    assert constant.bulk_ess.value is None
    assert nonfinite.rank_rhat.status == "unavailable"
    assert nonfinite.nonfinite_values == 2
    assert stuck.rank_rhat.value is None
    assert stuck.rank_rhat.status == "undefined"
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_location_scale_disagreement_and_sticky_tails() -> None:
    """Rank folding detects scale differences; sticky tail excursions lower ESS."""
    iid = np.random.default_rng(156).normal(size=(4, 1000))
    shifted, scaled, sticky = iid.copy(), iid.copy(), iid.copy()
    shifted[0] += 5
    scaled[0] *= 10
    sticky[0, -250:] += 10
    clean, location, scale, tails = [
        report(x).quantities[0].components[0] for x in (iid, shifted, scaled, sticky)
    ]
    for values, component in zip(
        (iid, shifted, scaled, sticky), (clean, location, scale, tails), strict=True
    ):
        assert component.rank_rhat.value == pytest.approx(
            float(array_stats.rhat(values, chain_axis=0, draw_axis=1, method="rank"))
        )
        assert component.tail_ess.value == pytest.approx(
            float(
                array_stats.ess(
                    values, chain_axis=0, draw_axis=1, method="tail", prob=(0.05, 0.95)
                )
            )
        )
    assert clean.rank_rhat.value is not None
    assert location.rank_rhat.value is not None
    assert scale.rank_rhat.value is not None
    assert location.rank_rhat.value > clean.rank_rhat.value + 0.1
    assert scale.rank_rhat.value > clean.rank_rhat.value + 0.1
    assert tails.tail_ess.value is not None
    assert clean.tail_ess.value is not None
    assert tails.tail_ess.value < clean.tail_ess.value / 5


@pytest.mark.parametrize("dtype", ["float16", "int64"])
def test_unsupported_precision_is_explicit(dtype: str) -> None:
    """Unsupported half precision and large integer ranks are never silently cast."""
    values = np.arange(32).reshape(2, 16).astype(dtype)
    if dtype == "int64":
        values += 2**53
    result = report(values)
    assert result.n_components == result.n_incomplete == 1
    assert result.quantities[0].components[0].rank_rhat.value is None


def test_seed_members_are_not_mcmc_chains() -> None:
    """Training seeds keep their axis meaning and all unavailable rows."""
    posterior = sample_posterior(np.ones((3, 8, 2)), axis="seed")
    result = diagnose_mcmc(posterior, quantities=("theta",), definition=DEFINITION)
    assert result.sample_axis == "seed"
    assert result.n_components == result.n_incomplete == 2
    assert (
        result.quantities[0].components[0].rank_rhat.reason
        == "seed_axis_is_not_mcmc_chains"
    )


def test_explicit_host_transfer_shape_and_quantity_selection() -> None:
    """Bad conversion and unknown/duplicate/empty selections cannot pass."""
    posterior = sample_posterior(np.arange(64.0).reshape(2, 32))
    converted = diagnose_mcmc(
        posterior,
        quantities=("theta",),
        definition=DEFINITION,
        to_numpy=lambda x: x.ravel(),
    )
    assert converted.n_incomplete == 1
    assert "preserve original shape" in (
        converted.quantities[0].components[0].rank_rhat.reason or ""
    )
    for names in ((), ("typo",), ("theta", "theta")):
        with pytest.raises(ValueError, match="distinct nonempty"):
            diagnose_mcmc(posterior, quantities=names, definition=DEFINITION)
    with pytest.raises(TypeError, match="not one string"):
        diagnose_mcmc(posterior, quantities="theta", definition=DEFINITION)
    first = report(np.arange(64.0).reshape(2, 32))
    with pytest.raises(ValueError, match="definition differs"):
        MCMCDiagnostics.load_json(
            first.model_dump_json(),
            expected=posterior.provenance,
            definition=reference("different interpretation"),
        )
    changed = posterior.provenance.model_copy(update={"estimator_stage": "other"})
    with pytest.raises(ValueError, match="Provenance mismatch"):
        MCMCDiagnostics.load_json(
            first.model_dump_json(), expected=changed, definition=DEFINITION
        )
    payload = first.model_dump(mode="json")
    payload["quantities"][0]["components"] = []
    with pytest.raises(ValidationError, match="Every event component"):
        MCMCDiagnostics.model_validate_json(json.dumps(payload))


def test_multiple_sites_and_tensor_event_order() -> None:
    """Requested sites and tensor indices retain deliberately non-sorted labels."""
    rng = np.random.default_rng(241)
    tensor = rng.normal(size=(3, 200, 2, 2))
    posterior = Posterior(
        draws=(
            SampledVariable("matrix", tensor, ("row", "column")),
            SampledVariable("scalar", tensor[:, :, 0, 0]),
        ),
        coordinates=(
            Coordinate(name="chain", labels=(9, 3, 1)),
            Coordinate(name="draw", labels=tuple(range(200))),
            Coordinate(name="row", labels=("b", "a")),
            Coordinate(name="column", labels=(7, 2)),
        ),
        provenance=provenance(),
    )
    result = diagnose_mcmc(
        posterior, quantities=("scalar", "matrix"), definition=DEFINITION
    )
    assert tuple(q.name for q in result.quantities) == ("scalar", "matrix")
    assert result.coordinates == posterior.coordinates
    assert result.n_components == 5
    assert result.n_incomplete == 0
    assert tuple(c.index for c in result.quantities[1].components) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    assert (
        result.quantities[0].components[0].rank_rhat
        == result.quantities[1].components[0].rank_rhat
    )


def test_reference_failure_retains_other_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One reference failure remains undefined without hiding successful metrics."""

    def failed(*_args: object, **_kwargs: object) -> object:
        msg = "synthetic backend failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(array_stats, "rhat", failed)
    result = report(np.random.default_rng(16).normal(size=(2, 100)))
    component = result.quantities[0].components[0]
    assert result.n_components == result.n_incomplete == 1
    assert component.rank_rhat.status == "undefined"
    assert "reference_error:RuntimeError" in (component.rank_rhat.reason or "")
    assert component.bulk_ess.status == component.tail_ess.status == "available"


def test_masked_conversion_cannot_silently_discard_mask() -> None:
    """The host boundary refuses masks rather than exposing hidden data."""
    posterior = sample_posterior(np.random.default_rng(14).normal(size=(2, 100)))
    factory = cast("Callable[..., NDArray[np.generic]]", np.ma.array)
    result = diagnose_mcmc(
        posterior,
        quantities=("theta",),
        definition=DEFINITION,
        to_numpy=lambda x: factory(x, mask=True),
    )
    assert result.n_components == result.n_incomplete == 1
    assert result.quantities[0].host_dtype is None
    assert result.quantities[0].content is None
    assert "unmasked NumPy" in (
        result.quantities[0].components[0].rank_rhat.reason or ""
    )


@pytest.mark.parametrize("change", ["seed", "schema", "tail", "nonfinite", "single"])
def test_serialized_report_refuses_inconsistent_diagnostic_inputs(change: str) -> None:
    """Finite results cannot be attached to seed axes or incompatible inputs."""
    result = report(np.random.default_rng(90).normal(size=(2, 100)))
    payload = result.model_dump(mode="json")
    if change == "seed":
        payload["sample_axis"] = "seed"
        payload["coordinates"][0]["name"] = "seed"
    elif change == "schema":
        payload["schema_version"] = True
    elif change == "tail":
        payload["tail_probabilities"] = [0.1, 0.9]
    elif change == "nonfinite":
        payload["quantities"][0]["components"][0]["nonfinite_values"] = 1
    else:
        payload["coordinates"][0]["labels"] = [1]
    with pytest.raises(ValidationError):
        MCMCDiagnostics.model_validate_json(json.dumps(payload))


def test_standalone_example_keeps_all_rows_in_table() -> None:
    """The runnable guide needs no test fixtures and retains unavailable rows."""
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/mcmc_diagnostics.py")
    )
    result = namespace["run_example"]()
    table = namespace["render_table"](result)
    assert result.n_components == 5
    assert result.n_incomplete == 2
    assert result.adequacy_status == "not_assessed"
    assert "reported[component=shifted]" in table
    assert "reported[component=scaled]" in table
    assert "constant_quantity" in table
    assert "nonfinite_input" in table
    assert len(table.splitlines()) == 7
