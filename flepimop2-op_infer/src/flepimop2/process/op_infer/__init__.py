# flepimop2-op_infer: flepimop2 provider for op_infer
# Copyright (C) 2026  Joshua Macdonald
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Thin executable op_infer workflows in flepimop2's process namespace.

Project factories own SystemProtocol.bind() /
EngineProtocol.run() and statistical callback construction. This GPL adapter
keeps flepimop2 imports out of the MIT core and implements no sampler/executor.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Pydantic field resolution
from typing import TYPE_CHECKING, Any, Literal

from click import echo
from flepimop2.process.abc import ProcessABC
from pydantic import Field, model_validator

from op_infer.calibrate import _write_json_atomically, aggregate, gate, load_replicates
from op_infer.posterior import Provenance  # noqa: TC001 - Pydantic field resolution
from op_infer.rng import RandomKey
from op_infer.sbc import run_sbc

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import ArrayLike

    from op_infer.sbc import SBCEstimand, SBCFit

__all__ = ["InferProcess", "SBCWorkflow"]


@dataclass(frozen=True)
class SBCWorkflow:
    """Project-built simulation/fitting callbacks and exact scientific identity."""

    prior_sampler: Callable[[RandomKey], Mapping[str, ArrayLike]]
    simulator: Callable[[Mapping[str, ArrayLike], RandomKey], ArrayLike]
    estimator: Callable[[ArrayLike, RandomKey], SBCFit]
    estimands: Mapping[str, SBCEstimand]
    observed: ArrayLike
    provenance: Provenance
    mask: ArrayLike | None = None


class InferProcess(ProcessABC, module="op_infer"):
    """Execute a configured SBC, design, or SBC calibration-report process.

    A project factory (``importable.module:callable``) receives ``parameters``
    and returns SBCWorkflow or DesignStudy. It constructs any system/engine
    bindings; the provider owns only configuration/artifact plumbing.
    Dry runs print the plan without importing or calling project code.
    """

    operation: Literal["sbc", "design", "calibrate"]
    directory: Path
    replicate_ids: list[int] = Field(min_length=1)
    seed: int = Field(ge=0, lt=2**128)
    factory: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    n_jobs: int = 1
    report: Path | None = None
    provenance: Provenance | None = None
    bins: int = Field(default=20, ge=2)
    power_floor: int = Field(default=1000, ge=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    max_flagged: int = Field(default=0, ge=0)
    must_pass: list[str] = Field(default_factory=list)
    max_failed_fraction: float = Field(default=0.05, ge=0, le=1)

    @model_validator(mode="after")
    def _contract(self) -> InferProcess:
        """Require complete explicit task addresses and mode-specific settings.

        Returns:
            The validated process configuration.

        Raises:
            ValueError: Configuration is incomplete or mixes incompatible modes.
        """
        if len(set(self.replicate_ids)) != len(self.replicate_ids):
            msg = "replicate_ids must be distinct"
            raise ValueError(msg)
        for index in self.replicate_ids:
            RandomKey(seed=self.seed, stream="provider.validation", index=index)
        if self.operation == "calibrate":
            if (
                self.report is None
                or self.provenance is None
                or self.factory is not None
            ):
                msg = (
                    "Calibration needs report and expected provenance, without factory"
                )
                raise ValueError(msg)
            if self.report.resolve().parent == self.directory.resolve():
                msg = "Calibration report must be outside the replicate directory"
                raise ValueError(msg)
        elif not self.factory or ":" not in self.factory:
            msg = "SBC/design needs an importable module:callable factory"
            raise ValueError(msg)
        return self

    def _process(self, *, dry_run: bool) -> None:
        """Delegate execution; dry runs do not invoke factories or write artifacts.

        Raises:
            TypeError: A factory returns the wrong workflow type.
            ValueError: An artifact violates the explicit scientific contract.
        """
        if dry_run:
            echo(f"op_infer {self.operation}: {self.directory}")
            return
        if self.operation == "calibrate":
            self._calibrate()
            return
        module_name, name = str(self.factory).split(":", 1)
        workflow = getattr(importlib.import_module(module_name), name)(
            **self.parameters
        )
        if self.operation == "sbc":
            if not isinstance(workflow, SBCWorkflow):
                msg = "SBC factory must return SBCWorkflow"
                raise TypeError(msg)
            run_sbc(
                workflow.prior_sampler,
                workflow.simulator,
                workflow.estimator,
                estimands=workflow.estimands,
                observed=workflow.observed,
                provenance=workflow.provenance,
                mask=workflow.mask,
                directory=self.directory,
                replicate_ids=self.replicate_ids,
                seed=self.seed,
                n_jobs=self.n_jobs,
            )
        else:
            from op_infer.design import DesignStudy  # noqa: PLC0415 - optional design workflow

            if not isinstance(workflow, DesignStudy):
                msg = "Design factory must return DesignStudy"
                raise TypeError(msg)
            workflow.run(
                self.directory,
                replicate_ids=self.replicate_ids,
                seed=self.seed,
                n_jobs=self.n_jobs,
            )

    def _calibrate(self) -> None:
        """Require a complete prespecified SBC bank and write a provenance report.

        Raises:
            ValueError: Missing/extra records, seed, provenance or contract mismatch.
        """
        manifest_bytes = (self.directory / "sbc-manifest.meta").read_bytes()
        manifest = json.loads(manifest_bytes)
        expected = (
            None if self.provenance is None else self.provenance.model_dump(mode="json")
        )
        if manifest["provenance"] != expected or manifest["seed"] != self.seed:
            msg = "SBC report provenance or seed mismatch"
            raise ValueError(msg)
        contract = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        expected_paths = {
            self.directory / f"rep_{i:020d}.json" for i in self.replicate_ids
        }
        if set(self.directory.glob("*.json")) != expected_paths:
            msg = "Calibration needs exactly the complete prespecified SBC ID bank"
            raise ValueError(msg)
        for index in self.replicate_ids:
            payload = json.loads(
                (self.directory / f"rep_{index:020d}.json").read_text()
            )
            if payload.get("_meta") != {
                "runner_schema": 1,
                "contract": contract,
                "replicate": index,
            }:
                msg = "SBC replicate contract mismatch"
                raise ValueError(msg)
        pits, contraction, diagnostics = load_replicates(self.directory)
        result = aggregate(
            pits,
            contraction,
            bins=self.bins,
            power_floor=self.power_floor,
            alpha=self.alpha,
            n_failed=diagnostics["n_failed"],
        )
        verdict = gate(
            result,
            max_flagged=self.max_flagged,
            must_pass=self.must_pass,
            max_failed_fraction=self.max_failed_fraction,
        )
        _write_json_atomically(
            str(self.report),
            {
                "schema_version": 1,
                "manifest": manifest,
                "contract": contract,
                "replicate_ids": self.replicate_ids,
                "result": result,
                "gate": verdict,
            },
        )
