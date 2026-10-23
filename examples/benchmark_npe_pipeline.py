"""Run fresh-process fixed-work NPE timing and memory measurements.

This benchmark is descriptive engineering evidence. It does not establish a
speedup, posterior calibration, application adequacy, or a production setting.
Run the matrix only on a deliberately controlled machine and retain contention
and placement records with every result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from op_infer.adapters.sbi import (
    NpeEnsemble,
    NpeSettings,
    fit_npe,
    load_npe,
    npe_settings_reference,
    save_npe,
    sbi_backend_versions,
)
from op_infer.posterior import (
    ArtifactReference,
    LikelihoodTargetReference,
    Provenance,
    Qualification,
)
from op_infer.rng import RandomKey

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

Storage = Literal["ordinary", "borrowed", "mapped"]
Stage = Literal["train", "checkpoint_full", "checkpoint_chunk", "checkpoint_resume"]

_DEVICE = re.compile(r"^(cpu|cuda:(0|[1-9][0-9]*))$", flags=re.ASCII)
_SCHEMA = "op-infer-npe-fixed-work-v1"
_SPLIT_SEED = 51_817


@dataclass(frozen=True)
class Case:
    """One immutable fresh-process condition."""

    device: str
    storage: Storage
    stage: Stage
    replicate: int

    def __post_init__(self) -> None:
        """Reject ambiguous devices and unsupported CUDA checkpoints."""
        if (
            not _DEVICE.fullmatch(self.device)
            or self.storage not in {"ordinary", "borrowed", "mapped"}
            or self.stage
            not in {
                "train",
                "checkpoint_full",
                "checkpoint_chunk",
                "checkpoint_resume",
            }
            or isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate < 0
        ):
            msg = "Unsupported NPE benchmark case"
            raise ValueError(msg)
        if self.device != "cpu" and self.stage != "train":
            msg = "Training checkpoint/resume qualification is CPU-only"
            raise ValueError(msg)

    @property
    def slug(self) -> str:
        """Return a filesystem-safe condition identity."""
        return (
            f"{self.device.replace(':', '-')}-{self.storage}-"
            f"{self.stage}-r{self.replicate}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _reference(payload: object, kind: str) -> ArtifactReference:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return ArtifactReference.from_bytes(encoded, kind=kind)


def _prepare_bank(
    directory: Path, rows: int, units: int, features: int
) -> dict[str, Any]:
    """Create one deterministic bank and return its immutable manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "bank-manifest.json"
    theta_path = directory / "theta.npy"
    observations_path = directory / "observations.npy"
    requested = {
        "schema": _SCHEMA,
        "rows": rows,
        "units": units,
        "features": features,
        "parameters": 2,
        "dtype": "float32",
        "seed": 51,
    }
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text())
        if not isinstance(loaded, dict):
            msg = "Benchmark bank manifest must be a JSON object"
            raise ValueError(msg)
        manifest: dict[str, Any] = loaded
        if any(manifest.get(key) != value for key, value in requested.items()):
            msg = "Existing benchmark bank differs; use a new directory"
            raise ValueError(msg)
        if not theta_path.exists() or not observations_path.exists():
            msg = "Benchmark bank manifest is incomplete"
            raise ValueError(msg)
        return manifest
    if theta_path.exists() or observations_path.exists():
        msg = "Unmanifested benchmark bank exists; use a new directory"
        raise ValueError(msg)

    generator = np.random.default_rng(51)
    theta = generator.standard_normal((rows, 2), dtype=np.float32)
    np.save(theta_path, theta)
    format_module = importlib.import_module("numpy.lib.format")
    mapped = format_module.open_memmap(
        observations_path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, units, features),
    )
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        mapped[start:stop] = (
            theta[start:stop, 0, None, None]
            + 0.3 * theta[start:stop, 1, None, None]
            + generator.standard_normal(
                (stop - start, units, features), dtype=np.float32
            )
        )
    mapped[::3, -1] = np.nan
    mapped.flush()
    manifest = {
        **requested,
        "theta": {"bytes": theta_path.stat().st_size, "digest": _sha256(theta_path)},
        "observations": {
            "bytes": observations_path.stat().st_size,
            "digest": _sha256(observations_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def _memory() -> dict[str, int]:
    values: dict[str, int] = {}
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.exists():
        parsed = {
            parts[0].rstrip(":"): int(parts[1]) * 1024
            for line in rollup.read_text().splitlines()
            if len(parts := line.split()) == 3 and parts[1].isdigit()
        }
        for key in (
            "Rss",
            "Pss",
            "Anonymous",
            "Private_Clean",
            "Private_Dirty",
            "Shared_Clean",
            "Shared_Dirty",
        ):
            if key in parsed:
                values[f"{key.lower()}_bytes"] = parsed[key]
        values["uss_bytes"] = sum(
            parsed.get(key, 0)
            for key in ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
        )
    status = Path("/proc/self/status")
    if status.exists():
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in status.read_text().splitlines()
            if ":" in line
        }
        for key in ("RssAnon", "RssFile", "RssShmem"):
            if key in fields:
                values[f"{key.lower()}_bytes"] = int(fields[key].split()[0]) * 1024
    return values


def _cgroup() -> dict[str, object]:
    root = Path("/sys/fs/cgroup")
    relative: str | None = None
    cgroup_file = Path("/proc/self/cgroup")
    if cgroup_file.exists():
        for line in cgroup_file.read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                relative = parts[2].lstrip("/")
                break
    scope = root / relative if relative else root
    record: dict[str, object] = {
        "path": str(scope),
        "shared_scope": not scope.name.startswith("op-infer-npe-"),
    }
    for filename in (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "memory.pressure",
        "cpu.pressure",
    ):
        try:
            record[filename] = (scope / filename).read_text().strip()
        except OSError as error:
            record[filename] = {"unavailable": type(error).__name__}
    return record


def _mount(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    best: tuple[int, dict[str, object]] | None = None
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            left, separator, right = line.partition(" - ")
            if not separator:
                continue
            fields = left.split()
            extra = right.split()
            if len(fields) < 6 or len(extra) < 2:
                continue
            mountpoint = Path(fields[4].replace("\\040", " "))
            try:
                resolved.relative_to(mountpoint)
            except ValueError:
                continue
            candidate: dict[str, object] = {
                "mountpoint": str(mountpoint),
                "filesystem": extra[0],
                "source": extra[1],
                "device": fields[2],
                "options": fields[5],
            }
            size = len(str(mountpoint))
            if best is None or size > best[0]:
                best = (size, candidate)
    return best[1] if best is not None else {"status": "unavailable"}


def _nvidia() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,clocks.sm,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(  # noqa: S603 - fixed diagnostic command
            command, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "reason": type(error).__name__}
    return {"fields": command[1], "rows": result.stdout.strip().splitlines()}


def _directory_bytes(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


class _Sampler:
    """Poll phase-local process memory and declared output-directory bytes."""

    def __init__(
        self,
        directories: dict[str, Path],
        interval: float = 0.01,
        monitor_cpu: int | None = None,
    ) -> None:
        self.directories = directories
        self.interval = interval
        self.monitor_cpu = monitor_cpu
        self.monitor_affinity: list[int] | None = None
        self.error: BaseException | None = None
        self.initial: dict[str, int] = {}
        self.final: dict[str, int] = {}
        self.peaks: dict[str, int] = {}
        self._latest: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Self:
        self._sample()
        self.initial = dict(self._latest)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
        self.final = dict(self._latest)

    def _sample(self) -> None:
        latest = _memory()
        for name, directory in self.directories.items():
            latest[f"{name}_directory_bytes"] = _directory_bytes(directory)
        self._latest = latest
        for key, value in latest.items():
            self.peaks[key] = max(self.peaks.get(key, 0), value)

    def _run(self) -> None:
        try:
            if self.monitor_cpu is not None:
                os.sched_setaffinity(0, {self.monitor_cpu})
            if hasattr(os, "sched_getaffinity"):
                self.monitor_affinity = sorted(os.sched_getaffinity(0))
            while not self._stop.wait(self.interval):
                self._sample()
        except (OSError, RuntimeError, ValueError) as error:
            self.error = error
            self._stop.set()


def _sync(torch: Any, device: Any) -> None:  # noqa: ANN401
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure[T](  # noqa: PLR0913
    label: str,
    operation: Callable[[], T],
    *,
    torch: Any,  # noqa: ANN401
    device: Any,  # noqa: ANN401
    measured_directories: dict[str, Path],
    monitor_cpu: int | None,
) -> tuple[T, dict[str, object]]:
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    before_group = _cgroup()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    with _Sampler(measured_directories, monitor_cpu=monitor_cpu) as sampler:
        _sync(torch, device)
        started = time.perf_counter()
        value = operation()
        _sync(torch, device)
        seconds = time.perf_counter() - started
    if sampler.error is not None:
        msg = "Phase telemetry sampler failed"
        raise RuntimeError(msg) from sampler.error
    after_usage = resource.getrusage(resource.RUSAGE_SELF)
    record: dict[str, object] = {
        "label": label,
        "seconds": seconds,
        "sample_interval_seconds": sampler.interval,
        "monitor_affinity": sampler.monitor_affinity,
        "sampled_initial": sampler.initial,
        "sampled_peaks": sampler.peaks,
        "sampled_final": sampler.final,
        "minor_page_faults": after_usage.ru_minflt - before_usage.ru_minflt,
        "major_page_faults": after_usage.ru_majflt - before_usage.ru_majflt,
        "block_input_operations": after_usage.ru_inblock - before_usage.ru_inblock,
        "block_output_operations": after_usage.ru_oublock - before_usage.ru_oublock,
        "cgroup_before": before_group,
        "cgroup_after": _cgroup(),
    }
    if device.type == "cuda":
        with torch.cuda.device(device):
            record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return value, record


def _source_manifest(script: Path) -> dict[str, str]:
    package_file = importlib.import_module("op_infer").__file__
    if package_file is None:
        msg = "The op_infer package has no source location"
        raise RuntimeError(msg)
    package = Path(package_file).parent
    source = {
        str(path.relative_to(package.parent)): _sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    source["examples/benchmark_npe_pipeline.py"] = _sha256(script)
    return source


def _provenance(
    settings: NpeSettings,
    manifest: dict[str, Any],
    source_manifest: dict[str, str],
) -> Provenance:
    source_reference = _reference(source_manifest, "source-manifest-v1")
    target = _reference(
        {"law": "x[i,j,k]~Normal(theta[i,0]+0.3*theta[i,1],1)"},
        "target-v1",
    )
    scope = _reference(
        {"bank": manifest, "settings": settings.model_dump(mode="json")},
        "npe-fixed-work-scope-v1",
    )
    return Provenance(
        schema_version=1,
        prior=_reference({"theta": "iid Normal(0,1)", "dimensions": 2}, "prior-v1"),
        design=_reference(manifest, "gaussian-cohort-design-v1"),
        simulator=source_reference,
        target=LikelihoodTargetReference(
            kind="analytic",
            intended=target,
            evaluated=target,
            qualification=Qualification(
                status="unqualified",
                scope=scope,
                reason=(
                    "Synthetic fixed-work performance pilot, not calibration evidence"
                ),
                evidence=(),
            ),
        ),
        numerical=None,
        estimator_stage="training_performance_pilot",
        estimator_settings=npe_settings_reference(settings),
        backend_versions=sbi_backend_versions(device=settings.device),
        commit=source_reference,
    )


def _draw_digest(posterior: Any) -> str:  # noqa: ANN401
    digest = hashlib.sha256()
    for variable in posterior.draws:
        array = variable.values.detach().cpu().contiguous().numpy()
        digest.update(variable.name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def _checkpoint_equivalence(payload: dict[str, Any]) -> dict[str, str]:
    """Hash replay state while keeping measured epoch durations descriptive."""
    training: Any = importlib.import_module("op_infer.adapters._sbi_training")
    state = payload["payload"]
    summary = dict(state["summary"])
    summary.pop("epoch_durations_sec", None)
    components = {
        "settings": state["settings"],
        "provenance": state["provenance"],
        "bank": state["bank"],
        "epoch": state["epoch"],
        "train_indices": state["train_indices"],
        "val_indices": state["val_indices"],
        "network": state["network"],
        "best_network": state["best_network"],
        "optimizer": state["optimizer"],
        "summary_without_durations": summary,
        "torch_rng": state["torch_rng"],
        "val_loss": state["_val_loss"],
        "best_val_loss": state["_best_val_loss"],
        "epochs_since_last_improvement": state["_epochs_since_last_improvement"],
    }
    return {name: training._digest(value) for name, value in components.items()}  # noqa: SLF001


def _fixed_split(
    rows: int,
    validation_fraction: float,
    torch: Any,  # noqa: ANN401
) -> tuple[Any, Any]:
    """Create one host-defined split shared by every execution device."""
    training_rows = int((1 - validation_fraction) * rows)
    permutation = np.random.default_rng(_SPLIT_SEED).permutation(rows).copy()
    indices = torch.from_numpy(permutation)
    return indices[:training_rows], indices[training_rows:]


@contextmanager
def _instrument_training(  # noqa: C901, PLR0915
    torch: Any,  # noqa: ANN401
    device: Any,  # noqa: ANN401
    fixed_split: tuple[Any, Any],
) -> Iterator[dict[str, Any]]:
    adapter = importlib.import_module("op_infer.adapters.sbi")
    training = importlib.import_module("op_infer.adapters._sbi_training")
    inference = importlib.import_module("sbi.inference")
    adapter_native: Any = adapter
    training_native: Any = training
    original_prepare = adapter_native._prepare_bank  # noqa: SLF001
    original_loaders = inference.NPE.get_dataloaders
    original_train = inference.NPE._train_epoch  # noqa: SLF001
    original_validate = inference.NPE._validate_epoch  # noqa: SLF001
    original_save = training_native.atomic_save
    original_restore = adapter_native.restore_training
    trace: dict[str, Any] = {
        "bank_validation_seconds": [],
        "split_digests": None,
        "train_epochs": [],
        "validation_epochs": [],
        "checkpoint_saves": [],
        "checkpoint_restores": [],
    }

    def rng_digests() -> dict[str, str]:
        values = {"cpu": training_native._digest(torch.random.get_rng_state())}  # noqa: SLF001
        if device.type == "cuda":
            values["cuda"] = training_native._digest(  # noqa: SLF001
                torch.cuda.get_rng_state(device)
            )
        return values

    def prepare(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        _sync(torch, device)
        started = time.perf_counter()
        result = original_prepare(*args, **kwargs)
        _sync(torch, device)
        trace["bank_validation_seconds"].append(time.perf_counter() - started)
        return result

    def loaders(
        owner: Any,  # noqa: ANN401
        *args: object,
        **kwargs: object,
    ) -> object:
        bound = inspect.signature(original_loaders).bind(owner, *args, **kwargs)
        bound.apply_defaults()
        if not bool(bound.arguments["resume_training"]):
            owner.train_indices, owner.val_indices = fixed_split
        bound.arguments["resume_training"] = True
        return original_loaders(*bound.args, **bound.kwargs)

    def train(owner: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        loader = args[0] if args else kwargs["train_loader"]
        if trace["split_digests"] is None:
            trace["split_digests"] = {
                "train_indices": training_native._digest(owner.train_indices),  # noqa: SLF001
                "validation_indices": training_native._digest(owner.val_indices),  # noqa: SLF001
            }
        rng_before = rng_digests()
        _sync(torch, device)
        started = time.perf_counter()
        result = original_train(owner, *args, **kwargs)
        _sync(torch, device)
        trace["train_epochs"].append(
            {
                "epoch_before_summary": int(owner.epoch),
                "seconds": time.perf_counter() - started,
                "batches": len(loader),
                "batch_size": loader.batch_size,
                "rng_before": rng_before,
                "rng_after": rng_digests(),
            }
        )
        return result

    def validate(owner: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        loader = args[0] if args else kwargs["val_loader"]
        _sync(torch, device)
        started = time.perf_counter()
        result = original_validate(owner, *args, **kwargs)
        _sync(torch, device)
        trace["validation_epochs"].append(
            {
                "epoch_before_summary": int(owner.epoch),
                "seconds": time.perf_counter() - started,
                "batches": len(loader),
                "batch_size": loader.batch_size,
            }
        )
        return result

    def save(payload: dict[str, Any], path: str | Path) -> None:
        started = time.perf_counter()
        original_save(payload, path)
        destination = Path(path)
        trace["checkpoint_saves"].append(
            {
                "seconds": time.perf_counter() - started,
                "destination_bytes": destination.stat().st_size,
            }
        )

    def restore(  # noqa: PLR0913
        native: Any,  # noqa: ANN401
        path: str | Path,
        *,
        settings: NpeSettings,
        provenance: Provenance,
        bank: str,
        n_simulations: int,
        until_epoch: int,
    ) -> None:
        source = Path(path)
        before_usage = resource.getrusage(resource.RUSAGE_SELF)
        _sync(torch, device)
        started = time.perf_counter()
        original_restore(
            native,
            source,
            settings=settings,
            provenance=provenance,
            bank=bank,
            n_simulations=n_simulations,
            until_epoch=until_epoch,
        )
        _sync(torch, device)
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        trace["checkpoint_restores"].append(
            {
                "seconds": time.perf_counter() - started,
                "source_bytes": source.stat().st_size,
                "minor_page_faults": (after_usage.ru_minflt - before_usage.ru_minflt),
                "major_page_faults": (after_usage.ru_majflt - before_usage.ru_majflt),
                "block_input_operations": (
                    after_usage.ru_inblock - before_usage.ru_inblock
                ),
            }
        )

    adapter_native._prepare_bank = prepare  # noqa: SLF001
    inference.NPE.get_dataloaders = loaders
    inference.NPE._train_epoch = train  # noqa: SLF001
    inference.NPE._validate_epoch = validate  # noqa: SLF001
    training_native.atomic_save = save
    adapter_native.restore_training = restore
    try:
        yield trace
    finally:
        adapter_native._prepare_bank = original_prepare  # noqa: SLF001
        inference.NPE.get_dataloaders = original_loaders
        inference.NPE._train_epoch = original_train  # noqa: SLF001
        inference.NPE._validate_epoch = original_validate  # noqa: SLF001
        training_native.atomic_save = original_save
        adapter_native.restore_training = original_restore


def _load_bank(
    directory: Path, case: Case
) -> tuple[Any, Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    torch = importlib.import_module("torch")
    mapped = case.storage == "mapped"
    theta_array = np.load(directory / "theta.npy", mmap_mode="c" if mapped else None)
    observation_array = np.load(
        directory / "observations.npy", mmap_mode="c" if mapped else None
    )
    return (
        torch.from_numpy(theta_array),
        torch.from_numpy(observation_array),
        theta_array,
        observation_array,
    )


def _checkpoint_paths(
    directory: Path, case: Case
) -> tuple[Path | None, Path | None, int | None]:
    if case.stage == "train":
        return None, None, None
    checkpoint_root = directory / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    suffix = "full" if case.stage == "checkpoint_full" else "resume"
    path = checkpoint_root / f"{case.storage}-r{case.replicate}-{suffix}.pt"
    if case.stage == "checkpoint_chunk":
        return path, None, 2
    if case.stage == "checkpoint_resume":
        return path, path, None
    return path, None, None


def _runtime_environment(
    directory: Path,
    torch: Any,  # noqa: ANN401
    target: Any,  # noqa: ANN401
    threads: int,
) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "numa_online": Path("/sys/devices/system/node/online").read_text().strip()
        if Path("/sys/devices/system/node/online").exists()
        else None,
        "threads": {
            "requested": threads,
            "torch": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "device": str(target),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mount": _mount(directory),
        "cgroup": _cgroup(),
        "nvidia_start": _nvidia() if target.type == "cuda" else None,
    }


def _workload(args: argparse.Namespace, case: Case | None = None) -> dict[str, object]:
    completed_target = (
        2
        if case is not None and case.stage == "checkpoint_chunk"
        else args.max_epochs + 1
    )
    return {
        "rows": args.rows,
        "units": args.units,
        "features": args.features,
        "split_seed": _SPLIT_SEED,
        "native_completed_epochs_requested": args.max_epochs + 1,
        "worker_completed_epochs_target": completed_target,
        "batch_size": args.batch_size,
        "sample_draws": args.sample_draws,
        "calibration_replicates": args.calibration_replicates,
        "calibration_draws_per_replicate": args.calibration_draws,
        "threads": args.threads,
        "compute_cpus": args.cpus,
        "monitor_cpu": args.monitor_cpu,
    }


def _worker(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901, PLR0915
    case = Case(args.device, args.storage, args.stage, args.replicate)
    available_cpus = (
        set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    if args.cpus:
        cpus = {int(value) for value in args.cpus.split(",")}
        if (
            not cpus
            or available_cpus is None
            or not cpus <= available_cpus
            or args.monitor_cpu in cpus
        ):
            msg = "Compute CPUs must be available and exclude the monitor CPU"
            raise ValueError(msg)
        os.sched_setaffinity(0, cpus)
    if args.monitor_cpu is not None and (
        not args.cpus
        or available_cpus is None
        or args.monitor_cpu not in available_cpus
    ):
        msg = "A monitor CPU requires a distinct available compute affinity"
        raise ValueError(msg)
    torch = importlib.import_module("torch")
    importlib.import_module("sbi")
    if importlib.metadata.version("sbi") != "0.27.0":
        msg = "The fixed-work harness requires the qualified sbi 0.27.0 boundary"
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    if case.device.startswith("cuda:") and not torch.cuda.is_available():
        msg = "CUDA is unavailable; a skipped worker is not qualification"
        raise RuntimeError(msg)
    target = torch.device(case.device)
    if args.calibration_replicates > args.rows:
        msg = "Calibration timing replicates cannot exceed bank rows"
        raise ValueError(msg)
    manifest = _prepare_bank(args.directory, args.rows, args.units, args.features)
    artifact_directory = args.directory / "artifacts" / case.slug
    artifact_directory.mkdir(parents=True, exist_ok=True)
    measured_directories = {
        "artifacts": artifact_directory,
        "checkpoints": args.directory / "checkpoints",
    }
    environment = _runtime_environment(args.directory, torch, target, args.threads)
    loaded, load_record = _measure(
        "bank_load",
        lambda: _load_bank(args.directory, case),
        torch=torch,
        device=target,
        measured_directories=measured_directories,
        monitor_cpu=args.monitor_cpu,
    )
    theta, observations, theta_array, observation_array = loaded
    if theta.shape != (args.rows, 2) or observations.shape != (
        args.rows,
        args.units,
        args.features,
    ):
        msg = "Bank shape differs from the declared worker condition"
        raise ValueError(msg)
    settings = NpeSettings(
        key=RandomKey(seed=2026, stream="npe_fixed_work_pilot"),
        parameter_names=("theta", "latent"),
        n_max=args.units,
        feature_dim=args.features,
        hidden_features=8,
        latent_dim=4,
        embedding_dim=4,
        num_transforms=1,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        mask_probability=0.5,
        device=case.device,
    )
    source_manifest = _source_manifest(Path(__file__))
    provenance = _provenance(settings, manifest, source_manifest)
    distributions = importlib.import_module("torch.distributions")
    prior = distributions.Independent(
        distributions.Normal(
            torch.zeros(2, device=target), torch.ones(2, device=target)
        ),
        1,
    )
    checkpoint_path, resume_from, until_epoch = _checkpoint_paths(args.directory, case)

    def train() -> Any:  # noqa: ANN401
        return fit_npe(
            theta,
            observations,
            prior=prior,
            settings=settings,
            provenance=provenance,
            checkpoint_path=checkpoint_path,
            resume_from=resume_from,
            until_epoch=until_epoch,
            borrow_bank=case.storage != "ordinary",
        )

    fixed_split = _fixed_split(args.rows, settings.validation_fraction, torch)
    with _instrument_training(torch, target, fixed_split) as trace:
        fit, fit_record = _measure(
            "fit",
            train,
            torch=torch,
            device=target,
            measured_directories=measured_directories,
            monitor_cpu=args.monitor_cpu,
        )
    snapshot = artifact_directory / "fit.pt"
    _, save_record = _measure(
        "snapshot_save",
        lambda: save_npe(fit, snapshot),
        torch=torch,
        device=target,
        measured_directories=measured_directories,
        monitor_cpu=args.monitor_cpu,
    )
    restored, load_snapshot_record = _measure(
        "snapshot_load",
        lambda: load_npe(snapshot, expected=provenance),
        torch=torch,
        device=target,
        measured_directories=measured_directories,
        monitor_cpu=args.monitor_cpu,
    )
    key = RandomKey(seed=90210, stream="npe_fixed_work_sampling")
    posterior, sample_record = _measure(
        "posterior_sample",
        lambda: NpeEnsemble((fit,)).sample(
            observations[0], per_seed=args.sample_draws, key=key
        ),
        torch=torch,
        device=target,
        measured_directories=measured_directories,
        monitor_cpu=args.monitor_cpu,
    )
    replay, replay_record = _measure(
        "posterior_replay",
        lambda: NpeEnsemble((restored,)).sample(
            observations[0], per_seed=args.sample_draws, key=key
        ),
        torch=torch,
        device=target,
        measured_directories=measured_directories,
        monitor_cpu=args.monitor_cpu,
    )
    draw_digest = _draw_digest(posterior)
    if _draw_digest(replay) != draw_digest:
        msg = "Guarded snapshot replay changed addressed posterior draws"
        raise RuntimeError(msg)

    def calibration_smoke() -> list[list[int]]:
        ranks: list[list[int]] = []
        ensemble = NpeEnsemble((restored,))
        for index in range(args.calibration_replicates):
            result = ensemble.sample(
                observations[index],
                per_seed=args.calibration_draws,
                key=RandomKey(
                    seed=90210,
                    stream="npe_fixed_work_calibration_timing",
                    index=index,
                ),
            )
            ranks.append(
                [
                    int((variable.values[0] < theta[index, column].to(target)).sum())
                    for column, variable in enumerate(result.draws)
                ]
            )
        return ranks

    ranks, calibration_record = _measure(
        "calibration_timing_smoke",
        calibration_smoke,
        torch=torch,
        device=target,
        measured_directories=measured_directories,
        monitor_cpu=args.monitor_cpu,
    )
    verified_bank = {
        "theta": _sha256(args.directory / "theta.npy"),
        "observations": _sha256(args.directory / "observations.npy"),
    }
    if (
        verified_bank["theta"] != manifest["theta"]["digest"]
        or verified_bank["observations"] != manifest["observations"]["digest"]
    ):
        msg = "Bank content changed relative to its immutable manifest"
        raise RuntimeError(msg)
    snapshot_payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    checkpoint_record = None
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        checkpoint_record = {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "payload_digest": checkpoint_payload["digest"],
            "equivalence_digests": _checkpoint_equivalence(checkpoint_payload),
            "epoch": checkpoint_payload["payload"]["epoch"],
        }
    result = {
        "schema": _SCHEMA,
        "case": asdict(case),
        "case_slug": case.slug,
        "manifest": manifest,
        "bank_digests_verified_after_execution": verified_bank,
        "environment": {
            **environment,
            "nvidia_end": _nvidia() if target.type == "cuda" else None,
            "backend_versions": [
                version.model_dump(mode="json")
                for version in provenance.backend_versions
            ],
        },
        "settings": settings.model_dump(mode="json"),
        "source_identity": {
            "manifest": source_manifest,
            "reference": provenance.commit.model_dump(mode="json"),
        },
        "workload": _workload(args, case),
        "bank_storage": {
            "theta_class": type(theta_array).__name__,
            "observations_class": type(observation_array).__name__,
            "borrow_bank": case.storage != "ordinary",
            "bytes": int(theta_array.nbytes + observation_array.nbytes),
        },
        "phases": {
            "bank_load": load_record,
            "fit": fit_record,
            "snapshot_save": save_record,
            "snapshot_load": load_snapshot_record,
            "posterior_sample": sample_record,
            "posterior_replay": replay_record,
            "calibration_timing_smoke": calibration_record,
        },
        "training_trace": trace,
        "completed_epochs": fit.epochs,
        "fit_state_digest": snapshot_payload["state_digest"],
        "snapshot": {"bytes": snapshot.stat().st_size, "digest": _sha256(snapshot)},
        "checkpoint": checkpoint_record,
        "draw_digest": draw_digest,
        "calibration_timing_only": {
            "replicates": args.calibration_replicates,
            "draws_per_replicate": args.calibration_draws,
            "rank_digest": hashlib.sha256(
                json.dumps(ranks, separators=(",", ":")).encode()
            ).hexdigest(),
            "qualification": "underpowered timing smoke; not calibration evidence",
        },
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def _matrix_cases(devices: Sequence[str], repetitions: int) -> list[Case]:
    cases: list[Case] = []
    storages: tuple[Storage, ...] = ("ordinary", "borrowed", "mapped")
    for replicate in range(repetitions):
        train = [
            Case(device, storage, "train", replicate)
            for device in devices
            for storage in storages
        ]
        if train:
            offset = replicate % len(train)
            cases.extend(train[offset:] + train[:offset])
        for storage in storages:
            cases.extend(
                (
                    Case("cpu", storage, "checkpoint_full", replicate),
                    Case("cpu", storage, "checkpoint_chunk", replicate),
                    Case("cpu", storage, "checkpoint_resume", replicate),
                )
            )
    return cases


def _summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        case = result["case"]
        if case["stage"] == "train":
            groups.setdefault((case["device"], case["storage"]), []).append(result)
    conditions = []
    for (device, storage), rows in sorted(groups.items()):
        fit_seconds = [row["phases"]["fit"]["seconds"] for row in rows]
        epoch_seconds = [
            epoch["seconds"]
            for row in rows
            for epoch in row["training_trace"]["train_epochs"]
        ]
        conditions.append(
            {
                "device": device,
                "storage": storage,
                "repetitions": len(rows),
                "instrumented_fit_seconds_median": statistics.median(fit_seconds),
                "instrumented_fit_seconds_range": [
                    min(fit_seconds),
                    max(fit_seconds),
                ],
                "instrumented_train_epoch_seconds_median": statistics.median(
                    epoch_seconds
                ),
                "descriptive_only": True,
            }
        )
    return {
        "schema": _SCHEMA,
        "conditions": conditions,
        "limitations": [
            "fixed synthetic work on one host; no application or speedup claim",
            "elapsed phases include a 10 ms monitor on a distinct declared CPU",
            (
                "fixed-seed repeats measure execution variation, including runtime "
                "nondeterminism, not statistical uncertainty"
            ),
            "calibration phase is an underpowered timing smoke only",
            "CPU checkpoints and CUDA inference snapshots have different scopes",
        ],
    }


def _require_one_identity(values: Sequence[object], message: str) -> None:
    if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
        raise RuntimeError(message)


def _validate_matrix(  # noqa: C901
    results: Sequence[dict[str, Any]], repetitions: int, completed_epochs: int
) -> dict[str, object]:
    """Require declared work and within-device storage paths to be identical."""
    by_key = {
        (
            row["case"]["device"],
            row["case"]["replicate"],
            row["case"]["storage"],
            row["case"]["stage"],
        ): row
        for row in results
    }
    for row in results:
        expected = 2 if row["case"]["stage"] == "checkpoint_chunk" else completed_epochs
        if row["completed_epochs"] != expected:
            msg = "A worker did not complete its declared fixed epoch count"
            raise RuntimeError(msg)
        if row["training_trace"]["split_digests"] is None:
            msg = "A worker did not retain its native split identity"
            raise RuntimeError(msg)
    _require_one_identity(
        [row["source_identity"]["reference"] for row in results],
        "Matrix workers used different source identities",
    )
    _require_one_identity(
        [row["training_trace"]["split_digests"] for row in results],
        "Matrix workers used different native splits",
    )

    devices = sorted({row["case"]["device"] for row in results})
    stages: tuple[Stage, ...] = (
        "train",
        "checkpoint_full",
        "checkpoint_chunk",
        "checkpoint_resume",
    )
    storages: tuple[Storage, ...] = ("ordinary", "borrowed", "mapped")
    for replicate in range(repetitions):
        replicate_rows = [
            row for row in results if row["case"]["replicate"] == replicate
        ]
        for device in devices:
            for stage in stages:
                selected = [
                    by_key[(device, replicate, storage, stage)]
                    for storage in storages
                    if (device, replicate, storage, stage) in by_key
                ]
                if not selected:
                    continue
                states = {row["fit_state_digest"] for row in selected}
                rng_traces = {
                    json.dumps(
                        [
                            (epoch["rng_before"], epoch["rng_after"])
                            for epoch in row["training_trace"]["train_epochs"]
                        ],
                        sort_keys=True,
                    )
                    for row in selected
                }
                if len(states) != 1 or len(rng_traces) != 1:
                    msg = (
                        "Storage paths changed training-order RNG or final state "
                        "within one device/stage"
                    )
                    raise RuntimeError(msg)

        cpu_full_budget_states = {
            row["fit_state_digest"]
            for row in replicate_rows
            if row["case"]["device"] == "cpu"
            and row["case"]["stage"]
            in {"train", "checkpoint_full", "checkpoint_resume"}
        }
        if len(cpu_full_budget_states) != 1:
            msg = "CPU full-budget train, checkpoint and resume states differ"
            raise RuntimeError(msg)

        for storage in storages:
            full = by_key[("cpu", replicate, storage, "checkpoint_full")]
            resumed = by_key[("cpu", replicate, storage, "checkpoint_resume")]
            if (
                full["fit_state_digest"] != resumed["fit_state_digest"]
                or full["checkpoint"]["equivalence_digests"]
                != resumed["checkpoint"]["equivalence_digests"]
            ):
                msg = "Uninterrupted and resumed CPU fixed-work states differ"
                raise RuntimeError(msg)
    return {
        "completed_epoch_counts": "verified for every worker",
        "source_identity": "one exact source-byte reference across every worker",
        "split_identity": ("exact across every repetition, device, stage and storage"),
        "within_device_storage_equivalence": (
            "exact epoch RNG trace and final fit state for every stage"
        ),
        "cpu_full_budget_equivalence": (
            "exact final fit state across train, checkpoint-full and resume paths"
        ),
        "checkpoint_resume_equivalence": (
            "exact network/optimizer/split/loss/RNG state for every CPU "
            "storage/replicate pair; measured epoch durations excluded"
        ),
    }


def _append_journal(path: Path, event: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _matrix_controls(args: argparse.Namespace) -> tuple[str, ...]:
    """Validate controls that make the retained matrix interpretable."""
    if args.max_epochs != 3:
        msg = "The retained matrix requires max_epochs=3 (four native epochs)"
        raise ValueError(msg)
    if not args.cpus or args.monitor_cpu is None:
        msg = "The retained matrix requires distinct compute and monitor CPUs"
        raise ValueError(msg)
    try:
        matrix_cpus = {int(value) for value in args.cpus.split(",")}
    except ValueError as error:
        msg = "Matrix compute CPUs must be comma-separated integers"
        raise ValueError(msg) from error
    if not matrix_cpus or args.monitor_cpu in matrix_cpus:
        msg = "The retained matrix requires distinct compute and monitor CPUs"
        raise ValueError(msg)
    devices = tuple(args.devices.split(","))
    if not devices or any(not _DEVICE.fullmatch(device) for device in devices):
        msg = "Use a comma-separated list of canonical CPU/CUDA devices"
        raise ValueError(msg)
    if "cpu" not in devices:
        msg = "The fixed-work pilot requires a CPU checkpoint control"
        raise ValueError(msg)
    return devices


def _run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    devices = _matrix_controls(args)
    manifest = _prepare_bank(args.directory, args.rows, args.units, args.features)
    results_directory = args.directory / "results"
    journal = args.directory / "results.jsonl"
    execution_paths = (
        results_directory,
        args.directory / "artifacts",
        args.directory / "checkpoints",
        journal,
        args.directory / "summary.json",
    )
    if any(path.exists() for path in execution_paths):
        msg = "Matrix output already exists; use a new prepared directory"
        raise FileExistsError(msg)
    results_directory.mkdir()
    _append_journal(
        journal,
        {
            "event": "matrix_started",
            "schema": _SCHEMA,
            "manifest": manifest,
            "devices": devices,
            "repetitions": args.repetitions,
            "workload": _workload(args),
            "argv": sys.argv,
        },
    )
    results: list[dict[str, Any]] = []
    for case in _matrix_cases(devices, args.repetitions):
        output = results_directory / f"{case.slug}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.directory),
            "--worker",
            "--device",
            case.device,
            "--storage",
            case.storage,
            "--stage",
            case.stage,
            "--replicate",
            str(case.replicate),
            "--worker-output",
            str(output),
            "--rows",
            str(args.rows),
            "--units",
            str(args.units),
            "--features",
            str(args.features),
            "--max-epochs",
            str(args.max_epochs),
            "--batch-size",
            str(args.batch_size),
            "--threads",
            str(args.threads),
            "--sample-draws",
            str(args.sample_draws),
            "--calibration-replicates",
            str(args.calibration_replicates),
            "--calibration-draws",
            str(args.calibration_draws),
        ]
        if args.cpus:
            command.extend(("--cpus", args.cpus))
        if args.monitor_cpu is not None:
            command.extend(("--monitor-cpu", str(args.monitor_cpu)))
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = str(args.threads)
        _append_journal(
            journal,
            {
                "event": "worker_started",
                "schema": _SCHEMA,
                "case": asdict(case),
                "workload": _workload(args, case),
                "command": command,
            },
        )
        completed = subprocess.run(  # noqa: S603
            command, check=False, env=environment, capture_output=True, text=True
        )
        if completed.returncode:
            _append_journal(
                journal,
                {
                    "event": "worker_failed",
                    "schema": _SCHEMA,
                    "case": asdict(case),
                    "workload": _workload(args, case),
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
            msg = f"Fixed-work worker failed: {case.slug}; inspect {journal}"
            raise RuntimeError(msg)
        result = json.loads(output.read_text())
        results.append(result)
        _append_journal(
            journal,
            {
                "event": "worker_completed",
                "schema": _SCHEMA,
                "case": asdict(case),
                "workload": _workload(args, case),
                "command": command,
                "worker_output_digest": _sha256(output),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": result,
            },
        )

    try:
        validations = _validate_matrix(results, args.repetitions, args.max_epochs + 1)
    except RuntimeError as error:
        _append_journal(
            journal,
            {
                "event": "matrix_validation_failed",
                "schema": _SCHEMA,
                "reason": str(error),
            },
        )
        raise
    _append_journal(
        journal,
        {
            "event": "matrix_validated",
            "schema": _SCHEMA,
            "validations": validations,
        },
    )
    summary = {
        **_summarize(results),
        "manifest": manifest,
        "cases": len(results),
        "results_digest": _sha256(journal),
        "validations": validations,
    }
    (args.directory / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--worker", action="store_true")
    actions.add_argument("--matrix", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", default="cpu,cuda:0")
    parser.add_argument(
        "--storage", choices=("ordinary", "borrowed", "mapped"), default="mapped"
    )
    parser.add_argument(
        "--stage",
        choices=("train", "checkpoint_full", "checkpoint_chunk", "checkpoint_resume"),
        default="train",
    )
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", help="Comma-separated Linux CPU affinity")
    parser.add_argument(
        "--monitor-cpu",
        type=int,
        help="Distinct Linux CPU used by the 10 ms telemetry sampler",
    )
    parser.add_argument("--sample-draws", type=int, default=4096)
    parser.add_argument("--calibration-replicates", type=int, default=32)
    parser.add_argument("--calibration-draws", type=int, default=128)
    return parser


def main() -> None:
    """Prepare a bank, run one fresh worker, or orchestrate the serial matrix."""
    args = _parser().parse_args()
    positive = (
        args.rows,
        args.units,
        args.features,
        args.max_epochs,
        args.batch_size,
        args.threads,
        args.sample_draws,
        args.calibration_replicates,
        args.calibration_draws,
    )
    if any(isinstance(value, bool) or value < 1 for value in positive):
        msg = "Benchmark sizes, epochs, draws and thread counts must be positive"
        raise ValueError(msg)
    if args.sample_draws < 2 or args.calibration_draws < 2:
        msg = "Posterior sample counts must be at least two"
        raise ValueError(msg)
    if args.prepare:
        result = _prepare_bank(args.directory, args.rows, args.units, args.features)
    elif args.worker:
        if args.worker_output is None:
            msg = "Workers require --worker-output"
            raise ValueError(msg)
        result = _worker(args)
    else:
        if args.repetitions < 1:
            msg = "Matrix repetitions must be positive"
            raise ValueError(msg)
        result = _run_matrix(args)
    print(json.dumps(result, sort_keys=True))  # noqa: T201 - executable evidence


if __name__ == "__main__":
    main()
