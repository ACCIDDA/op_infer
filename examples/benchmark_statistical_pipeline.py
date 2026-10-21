"""Fresh-process synthetic simulation/likelihood/gradient and memory baseline.

No fitted consumer, training network or scientific default is changed. Timing is
advisory; every successful worker first passes an independent analytic reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np

from op_infer.adapters.op import (
    FixedForwardSettings,
    compiled_forward,
    compiled_observation_reference,
    fixed_forward_mesh_reference,
    fixed_forward_settings_reference,
    structural_model_reference,
)
from op_infer.adapters.rng import jax_key, torch_generators
from op_infer.forward import CallableForward, ForwardDesign
from op_infer.observe import IIDObservation
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    DerivativeContract,
    ErrorComponent,
    LikelihoodTargetReference,
    NumericalExecution,
    Provenance,
    Qualification,
)
from op_infer.priors import JointPrior, Normal, PriorParameter
from op_infer.rng import RandomKey


@dataclass(frozen=True)
class Case:
    """Declare one bounded numerical task before any measurement."""

    backend: Literal["numpy", "jax", "torch"]
    size: Literal["small", "medium"]
    phase: Literal["simulate", "likelihood", "gradient"]
    checkpoint: bool = False

    def __post_init__(self) -> None:
        """Refuse unsupported cases before loading a backend or allocating arrays."""
        if (
            self.backend not in {"numpy", "jax", "torch"}
            or self.size not in {"small", "medium"}
            or self.phase not in {"simulate", "likelihood", "gradient"}
            or type(self.checkpoint) is not bool
        ):
            msg = "Unsupported benchmark case"
            raise ValueError(msg)
        if self.checkpoint and (self.backend != "jax" or self.phase != "gradient"):
            msg = "Checkpoint cases require JAX gradient execution"
            raise ValueError(msg)

    @property
    def dimensions(self) -> tuple[int, int, int, int]:
        """Return states, internal steps, observation intervals and batch size."""
        return (4, 16, 4, 1) if self.size == "small" else (32, 64, 16, 8)


def _ref(value: object, kind: str = "synthetic-pipeline-v1") -> ArtifactReference:
    return ArtifactReference.from_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), kind=kind
    )


def _host(value: Any) -> np.ndarray[Any, Any]:  # noqa: ANN401
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _synchronize(value: Any) -> None:  # noqa: ANN401
    if isinstance(value, tuple):
        for item in value:
            _synchronize(item)
    elif hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _memory() -> dict[str, object]:
    result: dict[str, object] = {
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024
    }
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.exists():
        values = {
            parts[0].rstrip(":"): int(parts[1]) * 1024
            for line in rollup.read_text().splitlines()
            if len(parts := line.split()) == 3 and parts[1].isdigit()
        }
        result.update(
            {key: values[key] for key in ("Rss", "Pss", "Anonymous") if key in values}
        )
        result["uss_bytes"] = sum(
            values.get(key, 0)
            for key in ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
        )
    status = Path("/proc/self/status")
    if status.exists():
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in status.read_text().splitlines()
            if ":" in line
        }
        result["rss_components_bytes"] = {
            key: int(fields[key].split()[0]) * 1024
            for key in ("RssAnon", "RssFile", "RssShmem")
            if key in fields
        }
    cgroup_root = Path("/sys/fs/cgroup")
    cgroup: dict[str, object] = {
        "scope": "visible cgroup; may include unrelated processes"
    }
    for filename in (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "memory.pressure",
    ):
        try:
            cgroup[filename] = (cgroup_root / filename).read_text().strip()
        except OSError as exc:
            cgroup[filename] = {"status": "unavailable", "reason": type(exc).__name__}
    result["cgroup_memory"] = cgroup
    return result


def _build(
    case: Case,
    xp: Any,  # noqa: ANN401
) -> tuple[
    CallableForward, ForwardDesign, JointPrior, IIDObservation, dict[str, object]
]:
    compiler = importlib.import_module("op_system")
    states, steps, intervals, batch = case.dimensions
    names = [f"x{i}" for i in range(states)]
    compiled = compiler.compile_spec(
        {
            "kind": "expr",
            "state": names,
            "equations": {name: f"-rate*{name}" for name in names},
            "observables": {"total": {"expr": "+".join(names)}},
        }
    )
    settings = FixedForwardSettings(
        times=tuple(float(t) for t in np.linspace(0.0, 1.0, intervals + 1)),
        max_step=1 / steps,
        fixed_loop="scan" if case.checkpoint else "auto",
        checkpoint=case.checkpoint,
    )
    prior = JointPrior(
        parameters=(
            PriorParameter(name="rate", distribution=Normal(loc=0.7, scale=0.1)),
        )
    )
    initial_sum = float((1 + np.arange(states) / states).sum())
    observation = IIDObservation(family="normal", scale=0.1 * initial_sum)
    evidence = Qualification(
        status="unqualified",
        scope=_ref(asdict(case)),
        reason="Worker checks a bounded analytic reference; no application claim",
        evidence=(),
    )
    solver = fixed_forward_settings_reference(settings)
    numerical = NumericalExecution(
        solver=solver,
        time_mesh=fixed_forward_mesh_reference(settings),
        spatial_mesh=None,
        topology=None,
        geometry="frozen",
        replay=None,
        mesh_policy=_ref(
            {"partition": "fixed", "settings": settings.model_dump(mode="json")}
        ),
        observation=compiled_observation_reference(compiled, ("total",)),
        field_prior=prior.reference,
        derivative=DerivativeContract(
            mode="finite_difference"
            if case.backend == "numpy"
            else "discrete_autodiff",
            parameters=("rate",),
            adaptive_path="not_applicable",
            graph_scope=_ref(
                {
                    "backend": case.backend,
                    "coordinate": "physical-rate",
                    "fixed_mesh": True,
                }
            ),
            qualification=evidence,
        ),
        errors=tuple(
            ErrorComponent.model_validate(
                {
                    "kind": kind,
                    "status": "specified" if kind == "time" else "not_applicable",
                    "reason": "Fixed RK4 steps"
                    if kind == "time"
                    else "Not used in this fixed Gaussian target",
                    "controls": solver if kind == "time" else None,
                    "evidence": None,
                }
            )
            for kind in (
                "time",
                "space_quadrature",
                "nonlinear_solve",
                "linear_solve",
                "unit_integration",
                "monte_carlo",
                "sampler",
            )
        ),
        qualification=evidence,
    )
    expected = Provenance(
        schema_version=2,
        structural_model=structural_model_reference(compiled),
        prior=prior.reference,
        design=_ref(
            {
                "times": settings.times,
                "initial": "1+i/states",
                "states": states,
                "data_rate": 0.8,
                "residual": "0.1*sigma*sin(t)",
            }
        ),
        simulator=_ref(
            {
                "adapter": "compiled_forward",
                "noise": observation.reference.model_dump(mode="json"),
            }
        ),
        target=LikelihoodTargetReference(
            kind="deterministic_approximation",
            intended=_ref(
                {"target": "analytic decay; Normal measurements and rate prior"}
            ),
            evaluated=solver,
            qualification=evidence,
        ),
        numerical=numerical,
        estimator_stage="synthetic-benchmark",
        estimator_settings=_ref(asdict(case)),
        backend_versions=tuple(
            BackendVersion(name=name, version=version(name))
            for name in sorted(
                (
                    "op-infer",
                    "op-system",
                    "op-engine",
                    "numpy",
                    "scipy",
                    "pydantic",
                )
            )
        ),
        commit=ArtifactReference.from_bytes(
            Path(__file__).read_bytes(), kind="pipeline-benchmark-source-v1"
        ),
    )

    def initial(_theta: Mapping[str, object], _design: object) -> object:
        return 1 + xp.arange(states, dtype=xp.float64) / states

    forward = compiled_forward(
        compiled,
        settings=settings,
        initial_state=initial,
        parameter_names=("rate",),
        output_names=("total",),
        provenance=expected,
    )
    return (
        forward,
        ForwardDesign(expected.design, None),
        prior,
        observation,
        {
            "states": states,
            "internal_steps": steps,
            "observation_count": intervals + 1,
            "batch": batch,
            "dtype": "float64",
            "initial_sum": initial_sum,
            "settings": settings.model_dump(mode="json"),
            "provenance": expected.identity,
            "structural_model": expected.structural_model.model_dump(mode="json")
            if expected.structural_model
            else None,
        },
    )


def _functions(  # noqa: C901 -- explicit native phase/AD dispatch
    case: Case,
    xp: Any,  # noqa: ANN401
    backend: Any,  # noqa: ANN401
) -> tuple[Any, Any, np.ndarray[Any, Any], dict[str, object]]:
    forward, design, prior, observation, metadata = _build(case, xp)
    states, _, intervals, batch = case.dimensions
    times = np.linspace(0.0, 1.0, intervals + 1)
    initial_sum = float((1 + np.arange(states) / states).sum())
    scale = 0.1 * initial_sum
    observed = initial_sum * np.exp(-0.8 * times) + 0.1 * scale * np.sin(times)
    data = xp.asarray(observed, dtype=xp.float64)
    rates = xp.asarray(np.linspace(0.6, 0.8, batch), dtype=xp.float64)
    host_key = RandomKey(seed=2026, stream="simulation")
    native_keys = (
        backend.random.split(jax_key(host_key), batch)
        if case.backend == "jax"
        else torch_generators(
            tuple(
                RandomKey(seed=2026, stream="simulation", index=i) for i in range(batch)
            )
        )
        if case.backend == "torch"
        else tuple(
            RandomKey(seed=2026, stream="simulation", index=i) for i in range(batch)
        )
    )

    def one(rate: Any, key: Any) -> Any:  # noqa: ANN401
        latent = forward({"rate": rate}, design, host_key)["total"]
        if case.phase == "simulate":
            return xp.stack((latent, observation.sample(latent, key)))
        return xp.sum(observation.log_prob(latent, data)) + prior.log_prob(
            {"rate": rate}
        )

    def values(parameters: Any, dynamic_keys: Any = None) -> Any:  # noqa: ANN401
        if case.backend == "jax":
            return backend.vmap(one)(parameters, dynamic_keys)
        # Reconstruct addressed Torch generators for exact replay; no global seed.
        keys = (
            torch_generators(
                tuple(
                    RandomKey(seed=2026, stream="simulation", index=i)
                    for i in range(batch)
                )
            )
            if case.backend == "torch"
            else native_keys
        )
        return xp.stack(
            [one(rate, key) for rate, key in zip(parameters, keys, strict=True)]
        )

    compiled = None
    if case.phase == "gradient":
        if case.backend == "jax":
            function = backend.jit(
                backend.grad(
                    lambda parameter, dynamic_keys: xp.sum(
                        values(parameter, dynamic_keys)
                    ),
                    argnums=0,
                )
            )
            compiled = function
        elif case.backend == "torch":

            def function(parameters: Any) -> Any:  # noqa: ANN401
                leaf = parameters.detach().requires_grad_()
                return backend.autograd.grad(values(leaf).sum(), leaf)[0]
        else:

            def function(parameters: Any) -> Any:  # noqa: ANN401
                epsilon = 1e-5
                return (values(parameters + epsilon) - values(parameters - epsilon)) / (
                    2 * epsilon
                )
    else:
        function = backend.jit(values) if case.backend == "jax" else values
        compiled = function if case.backend == "jax" else None
    metadata["rng_contract"] = (
        "RandomKey v1; NumPy PCG64DXSM, JAX Threefry2x32, Torch CPU MT19937"
    )
    metadata["dynamic_jax_key_inputs"] = (
        case.backend == "jax" and case.phase == "simulate"
    )
    metadata["observed_sha256"] = hashlib.sha256(observed.tobytes()).hexdigest()
    metadata["primal_history"] = (
        "current state/FSAL and requested scalar outputs; AD residuals additional"
    )
    metadata["gradient_method"] = (
        "central finite difference, two target evaluations"
        if case.phase == "gradient" and case.backend == "numpy"
        else "native reverse-mode"
        if case.phase == "gradient"
        else "none"
    )
    analytic = initial_sum * np.exp(
        -np.asarray(rates.detach() if case.backend == "torch" else rates)[:, None]
        * times
    )
    if case.phase == "simulate":
        reference = analytic
    elif case.phase == "likelihood":
        reference = (
            (
                -0.5 * ((observed - analytic) / scale) ** 2
                - np.log(scale * np.sqrt(2 * np.pi))
            ).sum(-1)
            - 0.5 * ((np.asarray(_host(rates)) - 0.7) / 0.1) ** 2
            - np.log(0.1 * np.sqrt(2 * np.pi))
        )
    else:
        reference = ((observed - analytic) / scale**2 * (-times * analytic)).sum(-1) - (
            np.asarray(_host(rates)) - 0.7
        ) / 0.1**2
    metadata["illustrative_full_history_bytes"] = batch * states * (intervals + 1) * 8
    return (
        function,
        ((rates, native_keys) if case.backend == "jax" else (rates,), compiled),
        reference,
        metadata,
    )


def _sources() -> dict[str, str]:
    """Fingerprint installed Python source bytes without exposing local paths."""
    result = {}
    for name in ("op_infer", "op_system", "op_engine"):
        module_file = importlib.import_module(name).__file__
        if module_file is None:
            msg = "Cannot fingerprint a namespace package without source files"
            raise RuntimeError(msg)
        root = Path(module_file).parent
        manifest = [
            (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.rglob("*.py"))
        ]
        result[name] = hashlib.sha256(
            json.dumps(manifest, separators=(",", ":")).encode()
        ).hexdigest()
    return result


def worker(case: Case, repeats: int) -> dict[str, object]:
    """Run one phase in isolation, with independent analytic qualification."""
    if repeats < 1:
        msg = "Worker warm repetitions must be positive"
        raise ValueError(msg)
    os.environ["JAX_PLATFORMS"] = "cpu"
    if not sys.platform.startswith("linux"):
        msg = "This RSS/affinity qualification protocol requires Linux"
        raise RuntimeError(msg)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[0]})
    backend_start = time.perf_counter()
    xp = (
        np
        if case.backend == "numpy"
        else importlib.import_module("jax.numpy" if case.backend == "jax" else "torch")
    )
    backend = importlib.import_module(case.backend) if case.backend != "numpy" else np
    if case.backend == "torch":
        backend.set_num_threads(1)
        backend.set_num_interop_threads(1)
    if case.backend == "jax":
        backend.config.update("jax_enable_x64", val=True)
    backend_setup = time.perf_counter() - backend_start
    start = time.perf_counter()
    function, (arguments, compiled), reference, metadata = _functions(case, xp, backend)
    setup = time.perf_counter() - start
    memory_before = _memory()
    start = time.perf_counter()
    result = function(*arguments)
    _synchronize(result)
    first = time.perf_counter() - start
    actual = _host(result[:, 0] if case.phase == "simulate" else result)
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)
    if case.phase == "simulate":
        if not np.isfinite(_host(result)).all():
            msg = "Synthetic measurements are nonfinite"
            raise RuntimeError(msg)
        np.testing.assert_array_equal(_host(result), _host(function(*arguments)))
    warm = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*arguments)
        _synchronize(result)
        warm.append(time.perf_counter() - start)
    memory_after = _memory()
    xla = None
    if compiled is not None:
        analysis = compiled.lower(*arguments).compile().memory_analysis()
        xla = {
            name: int(getattr(analysis, name))
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
            if getattr(analysis, name, None) is not None
        }
    pools = importlib.import_module("threadpoolctl").threadpool_info()
    return {
        "case": asdict(case),
        "status": "success",
        "metadata": metadata,
        "backend_setup_seconds": backend_setup,
        "setup_seconds": setup,
        "source_content_sha256": _sources(),
        "first_seconds": first,
        "warm_seconds": warm,
        "warm_median_seconds": float(np.median(warm)),
        "max_absolute_reference_error": float(np.max(np.abs(actual - reference))),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "xla_memory_analysis": xla,
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_pools": [
            {
                key: pool.get(key)
                for key in (
                    "internal_api",
                    "num_threads",
                    "architecture",
                    "threading_layer",
                    "version",
                )
            }
            for pool in pools
        ],
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "array-api-compat",
                "op-system",
                "op-engine",
                "op-infer",
                *(() if case.backend == "numpy" else (case.backend,)),
                *(("jaxlib", "numpyro") if case.backend == "jax" else ()),
            )
        },
        "result_shape": list(_host(result).shape),
        "result_bytes": _host(result).nbytes,
        "runtime_controls": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "JAX_PLATFORMS",
                "JAX_ENABLE_COMPILATION_CACHE",
            )
        },
        "result_sha256": hashlib.sha256(_host(result).tobytes()).hexdigest(),
    }


def main() -> None:
    """Run serial fresh workers and preserve failures and censored timeouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help="Internal JSON case declaration")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(Case(**json.loads(args.worker)), args.warm_repeats)))  # noqa: T201
        return
    if (
        args.output is None
        or args.output.exists()
        or args.repetitions < 1
        or args.warm_repeats < 1
        or args.timeout <= 0
    ):
        parser.error("Supply a new output path, positive repeat counts and timeout")
    cases = [
        Case(backend, size, phase)
        for backend in ("numpy", "jax", "torch")
        for size in ("small", "medium")
        for phase in ("simulate", "likelihood", "gradient")
    ]
    cases += [
        Case("jax", size, "gradient", checkpoint=True) for size in ("small", "medium")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal_path = args.output.with_suffix(".workers.jsonl")
    if journal_path.exists():
        parser.error("Worker journal already exists")
    with journal_path.open("x"):
        pass
    records = []
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    for repetition in range(args.repetitions):
        for case in cases if repetition % 2 == 0 else list(reversed(cases)):
            try:
                completed = subprocess.run(  # noqa: S603 -- fixed interpreter/script and JSON argv
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        json.dumps(asdict(case)),
                        "--warm-repeats",
                        str(args.warm_repeats),
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                record = (
                    json.loads(completed.stdout.splitlines()[-1])
                    if completed.returncode == 0
                    else {
                        "case": asdict(case),
                        "status": "error",
                        "returncode": completed.returncode,
                        "stderr": completed.stderr[-6000:],
                    }
                )
            except subprocess.TimeoutExpired:
                record = {
                    "case": asdict(case),
                    "status": "timeout",
                    "timeout_seconds": args.timeout,
                }
            record["repetition"] = repetition
            records.append(record)
            with journal_path.open("a") as journal:
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            print(  # noqa: T201 -- progress on the benchmark CLI
                json.dumps(
                    {
                        "case": asdict(case),
                        "status": record["status"],
                        "repetition": repetition,
                    }
                ),
                flush=True,
            )
    # Raw failure stderr stays in the caller's receipt, never implicitly published.
    result = {
        "schema_version": 1,
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unavailable",
        ),
        "machine": platform.machine(),
        "system": platform.system(),
        "records": records,
        "limitations": [
            "CPU synthetic fixed-mesh tasks only; no training or application inference",
            "process RSS/PSS includes libraries and allocator state",
            "XLA estimates differ from measured process memory",
            "NumPy gradient uses two finite-difference evaluations",
            "no CUDA/CuPy qualification or measured state-history savings",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    if any(record["status"] != "success" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
