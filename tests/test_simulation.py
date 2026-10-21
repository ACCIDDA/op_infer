"""Forward/prior/observation integration with separate randomness and identities."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from op_infer.forward import CallableForward, ForwardDesign
from op_infer.observe import IIDObservation
from op_infer.priors import JointPrior, LogNormal, PriorParameter
from op_infer.rng import RandomKey, SimulationStreams
from op_infer.simulation import ObservationChannel, StatisticalModel
from tests.test_forward import identified_provenance


def _array(value: object) -> NDArray[np.float64]:
    """Assert the NumPy boundary before independent mathematical checks."""
    assert isinstance(value, np.ndarray)
    return value


def make_model() -> tuple[StatisticalModel, ForwardDesign]:
    """Use a positive physical decay rate and an independent Gaussian measurement."""
    prior = JointPrior(
        parameters=(
            PriorParameter(name="rate", distribution=LogNormal(loc=-0.5, scale=0.3)),
        )
    )
    expected = identified_provenance().model_copy(update={"prior": prior.reference})
    times = np.array([0.0, 0.5, 1.0])

    def predict(
        theta: Mapping[str, object], inputs: object, key: RandomKey
    ) -> Mapping[str, object]:
        del key
        return {"signal": np.exp(-_array(theta["rate"]) * _array(inputs))}

    forward = CallableForward(predict, ("rate",), ("signal",), expected)
    model = StatisticalModel(
        forward,
        prior,
        (
            ObservationChannel(
                "sensor", "signal", IIDObservation(family="normal", scale=0.2)
            ),
        ),
    )
    return model, ForwardDesign(expected.design, times)


def test_complete_simulation_and_likelihood_share_one_target() -> None:
    """Analytic decay and Gaussian scoring independently check the composition."""
    model, design = make_model()
    streams = SimulationStreams(seed=73, index=4)
    run = model.sample_numpy(design, streams)
    np.testing.assert_allclose(
        _array(run.latent["signal"]),
        np.exp(-_array(run.parameters["rate"]) * _array(design.inputs)),
    )
    np.testing.assert_array_equal(
        _array(run.measurements["sensor"]),
        _array(model.sample_numpy(design, streams).measurements["sensor"]),
    )
    assert not np.array_equal(
        _array(run.latent["signal"]), _array(run.measurements["sensor"])
    )
    terms = model.log_prob(run.parameters, design, streams.simulation, run.measurements)
    np.testing.assert_allclose(
        _array(terms["sensor"]),
        stats.norm.logpdf(
            _array(run.measurements["sensor"]),
            loc=_array(run.latent["signal"]),
            scale=0.2,
        ),
    )
    assert run.pipeline == model.reference
    changed = replace(
        model,
        channels=(
            ObservationChannel(
                "sensor", "signal", IIDObservation(family="normal", scale=0.3)
            ),
        ),
    )
    assert changed.reference != model.reference


def test_channel_streams_do_not_depend_on_other_channels_or_order() -> None:
    """A new sensor does not change the existing sensor or parameter draws."""
    model, design = make_model()
    other = ObservationChannel(
        "other", "signal", IIDObservation(family="normal", scale=0.5)
    )
    more = replace(model, channels=(other, *model.channels))
    streams = SimulationStreams(seed=73, index=4)
    a = model.sample_numpy(design, streams)
    b = more.sample_numpy(design, streams)
    np.testing.assert_array_equal(
        _array(a.parameters["rate"]), _array(b.parameters["rate"])
    )
    np.testing.assert_array_equal(
        _array(a.measurements["sensor"]), _array(b.measurements["sensor"])
    )
    assert not np.array_equal(
        _array(b.measurements["other"]), _array(b.measurements["sensor"])
    )


def test_mismatched_prior_channel_and_data_fail_before_reuse() -> None:
    """Statistical changes cannot silently inherit an incompatible forward target."""
    model, design = make_model()
    with pytest.raises(ValueError, match="prior identity"):
        replace(
            model,
            prior=JointPrior(
                parameters=(PriorParameter(name="rate", distribution=LogNormal()),)
            ),
        )
    with pytest.raises(ValueError, match="declared latent"):
        replace(
            model,
            channels=(
                ObservationChannel("sensor", "typo", IIDObservation(family="normal")),
            ),
        )
    with pytest.raises(ValueError, match="channel names"):
        model.log_prob(
            {"rate": np.array(0.5)},
            design,
            RandomKey(seed=73, stream="s"),
            {"typo": np.array(1.0)},
        )
