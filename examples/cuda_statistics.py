"""Require actual CUDA execution of the shared iid statistical interfaces.

Run each backend in its separately installed CUDA environment. This checks
native devices, independent density/moment references and local RNG replay.
It does not train an estimator or qualify an application calibration policy.
"""

# Optional native namespaces are deliberately dynamic at this standalone boundary.
# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import resource
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import stats

from op_infer.adapters.rng import jax_key, torch_generators
from op_infer.observe import IIDObservation
from op_infer.priors import (
    Beta,
    Gamma,
    JointPrior,
    LogNormal,
    Normal,
    PriorParameter,
    Uniform,
)
from op_infer.rng import RandomKey

type _Family = Literal[
    "normal", "lognormal", "poisson", "nb2", "binomial", "beta_binomial", "bernoulli"
]


def _native(module: Any, value: object, dtype: str, backend: str) -> Any:
    if backend == "torch":
        return module.as_tensor(value, dtype=getattr(module, dtype), device="cuda:0")
    return module.asarray(value, dtype=getattr(module, dtype))


def _host(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _device(value: Any, backend: str) -> None:
    if backend == "torch":
        if value.device.type != "cuda":
            msg = "Expected native CUDA output"
            raise RuntimeError(msg)
    elif any(device.platform != "gpu" for device in value.devices()):
        msg = "Expected native JAX GPU output"
        raise RuntimeError(msg)


def _fixture(family: _Family) -> tuple[IIDObservation, Any, Any, Any, Any]:
    law = IIDObservation(
        family=family,
        scale=0.4,
        concentration=3.0,
        trials=5,
        sensitivity=0.9,
        specificity=0.8,
    )
    latent = (
        np.array([2.0, 6.0]) if family in {"poisson", "nb2"} else np.array([0.2, 0.6])
    )
    observed = np.array([1.0, 2.0]) if family == "lognormal" else np.array([0.0, 1.0])
    if family == "normal":
        mean, variance = latent, np.full(2, 0.16)
    elif family == "lognormal":
        mean = np.exp(latent + 0.08)
        variance = math.expm1(0.16) * np.exp(2 * latent + 0.16)
    elif family in {"poisson", "nb2"}:
        mean = latent
        variance = latent if family == "poisson" else latent + latent**2 / 3
    elif family == "bernoulli":
        mean = 0.2 + 0.7 * latent
        variance = mean * (1 - mean)
    else:
        mean = 5 * latent
        variance = 5 * latent * (1 - latent) * (2 if family == "beta_binomial" else 1)
    return law, latent, observed, mean, variance


def _sample(law: IIDObservation, latent: Any, key: RandomKey, backend: str) -> Any:
    if backend == "jax":
        jax = importlib.import_module("jax")
        return jax.jit(lambda value, rng: law.sample(value, rng, sample_shape=(8192,)))(
            latent, jax_key(key)
        )
    return law.sample(latent, key, sample_shape=(8192,))


def _observation_case(
    module: Any, backend: str, family: _Family, dtype: str
) -> dict[str, object]:
    law, location, observed, mean, variance = _fixture(family)
    latent = _native(module, location, dtype, backend)
    data = _native(module, observed, dtype, backend)
    score = law.log_prob(latent, data)
    _device(score, backend)
    tolerance = 3e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(
        _host(score), law.log_prob(location, observed), rtol=tolerance, atol=tolerance
    )
    key = RandomKey(seed=2026, stream=f"cuda/{family}/{dtype}")
    draws = _sample(law, latent, key, backend)
    _device(draws, backend)
    if draws.dtype != latent.dtype:
        msg = "CUDA sampling changed the latent dtype"
        raise RuntimeError(msg)
    np.testing.assert_array_equal(
        _host(draws), _host(_sample(law, latent, key, backend))
    )
    empirical_mean, empirical_variance = (
        _host(draws).mean(axis=0),
        _host(draws).var(axis=0),
    )
    np.testing.assert_array_less(
        np.abs(empirical_mean - mean), 8 * np.sqrt(variance / 8192)
    )
    np.testing.assert_array_less(np.abs(empirical_variance - variance), 0.2 * variance)
    return {
        "family": family,
        "dtype": dtype,
        "samples": 8192,
        "density_and_replay": "passed",
        "sample_mean": empirical_mean.tolist(),
        "sample_variance": empirical_variance.tolist(),
        "reference_mean": mean.tolist(),
        "reference_variance": variance.tolist(),
    }


def _priors(module: Any, backend: str, dtype: str) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for law, value in (
        (Normal(), 0.3),
        (LogNormal(), 1.2),
        (Gamma(concentration=2.0, rate=3.0), 0.7),
        (Beta(concentration1=2.0, concentration0=3.0), 0.4),
        (Uniform(), 0.3),
    ):
        prior = JointPrior(
            parameters=(PriorParameter(name="parameter", distribution=law),)
        )
        coordinate = _native(module, value, dtype, backend)
        score = prior.log_prob({"parameter": coordinate})
        _device(score, backend)
        tolerance = 3e-5 if dtype == "float32" else 1e-10
        np.testing.assert_allclose(
            _host(score), law.scipy().logpdf(value), rtol=tolerance, atol=tolerance
        )
        cases.append({"prior_family": law.family, "dtype": dtype, "density": "passed"})
    return cases


def _masked_censored_gradient(module: Any, backend: str) -> None:
    law = IIDObservation(family="normal", scale=2.0, lower_limit=-1.0, upper_limit=3.0)
    latent = _native(module, [0.0, 0.0, 0.0, 0.0], "float64", backend)
    data = _native(module, [-1.0, 0.0, 3.0, float("nan")], "float64", backend)
    mask = _native(module, [True, True, True, False], "bool", backend)
    expected = [
        stats.norm.logcdf(-0.5),
        stats.norm.logpdf(0.0, scale=2.0),
        stats.norm.logsf(1.5),
        0.0,
    ]
    np.testing.assert_allclose(
        _host(law.log_prob(latent, data, mask=mask)), expected, rtol=1e-10, atol=1e-10
    )
    if backend == "jax":
        jax = importlib.import_module("jax")
        gradient = jax.jit(
            jax.grad(lambda x: module.sum(law.log_prob(x, data, mask=mask)))
        )(latent)
    else:
        latent.requires_grad_()
        law.log_prob(latent, data, mask=mask).sum().backward()
        gradient = latent.grad
    _device(gradient, backend)
    expected_gradient = [
        -stats.norm.pdf(-0.5) / stats.norm.cdf(-0.5) / 2,
        0.0,
        stats.norm.pdf(1.5) / stats.norm.sf(1.5) / 2,
        0.0,
    ]
    np.testing.assert_allclose(
        _host(gradient), expected_gradient, rtol=1e-10, atol=1e-10
    )


def _qualify(module: Any, backend: str) -> dict[str, object]:
    families: tuple[_Family, ...] = (
        "normal",
        "lognormal",
        "poisson",
        "nb2",
        "binomial",
        "beta_binomial",
        "bernoulli",
    )
    if backend == "torch":
        keys = tuple(
            RandomKey(seed=2026, stream=f"cuda/{family}/{dtype}")
            for family in families
            for dtype in ("float32", "float64")
        )
        torch_generators(keys, device="cuda:0")
        before = (module.get_rng_state().clone(), module.cuda.get_rng_state().clone())
    cases = [
        _observation_case(module, backend, family, dtype)
        for family in families
        for dtype in ("float32", "float64")
    ]
    for dtype in ("float32", "float64"):
        cases.extend(_priors(module, backend, dtype))
    _masked_censored_gradient(module, backend)
    if backend == "torch":
        if not (
            module.equal(before[0], module.get_rng_state())
            and module.equal(before[1], module.cuda.get_rng_state())
        ):
            msg = "Addressed CUDA sampling changed a global RNG state"
            raise RuntimeError(msg)
        try:
            IIDObservation(family="normal").sample(
                _native(module, [0.0], "float64", backend),
                module.Generator(device="cpu"),
            )
        except ValueError as error:
            if "device must match" not in str(error):
                raise
        else:
            msg = "A CPU generator was accepted for CUDA sampling"
            raise RuntimeError(msg)
    return {
        "backend": backend,
        "status": "passed",
        "cases": cases,
        "masked_censored_gradient": "passed",
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "versions": {
            name: version(name) for name in ("numpy", "scipy", "op-infer", backend)
        },
        "source_sha256": {
            obj.__name__: hashlib.sha256(
                Path(inspect.getfile(obj)).read_bytes()
            ).hexdigest()
            for obj in (IIDObservation, JointPrior, torch_generators)
        },
        "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    """Fail without CUDA; print successful numerical checks as a JSON receipt.

    Raises:
        RuntimeError: The selected backend did not initialize an actual GPU.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "torch"), required=True)
    backend = parser.parse_args().backend
    module = importlib.import_module(backend)
    context = nullcontext()
    if backend == "torch":
        if not module.cuda.is_available():
            msg = "Torch CUDA is required; CPU fallback is not qualification"
            raise RuntimeError(msg)
        device = module.cuda.get_device_name(0)
        module.cuda.reset_peak_memory_stats()
    else:
        if module.default_backend() != "gpu":
            msg = "JAX CUDA is required; CPU fallback is not qualification"
            raise RuntimeError(msg)
        device = str(module.devices()[0].device_kind)
        enable = getattr(module, "enable_x64", None)
        if enable is None:
            enable = module.experimental.enable_x64
        context = enable()
        module = importlib.import_module("jax.numpy")
    with context:
        result = _qualify(module, backend)
    result["device"] = device
    if backend == "torch":
        result["device_peak_allocated_bytes"] = module.cuda.max_memory_allocated()
        result["device_peak_reserved_bytes"] = module.cuda.max_memory_reserved()
    print(json.dumps(result, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
