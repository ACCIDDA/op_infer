# Run all default tasks for local development
default: format check mypy pytest docs

format:
	uv run ruff format

check:
	uv run ruff check --fix

mypy:
	uv run mypy

pytest:
	uv run pytest

docs:
	uv run mkdocs build --strict

# Optional qualification: install op, jax and torch extras before this recipe.
# Explicit imports make missing backends a failure, never a successful skipped run.
qualify-numerical:
	uv run --no-sync python -c 'import op_engine, jax, torch'
	uv run --no-sync pytest tests/test_numerical_reference.py

# Requires the reviewed structural-identity compiler revision until its release.
qualify-forward:
	uv run --no-sync python -c 'from op_system import StructuralModelIdentity; import op_engine, jax, torch, trade_study'
	uv run --no-sync mypy
	uv run --no-sync pytest tests/test_forward.py tests/test_forward_op.py
	uv run --no-sync python examples/compiled_forward.py

# Optional snapshots: missing dependencies must fail this qualification.
qualify-posterior:
	uv run --no-sync python -c 'import arviz_base, xarray, netCDF4'
	uv run --no-sync mypy
	uv run --no-sync pytest tests/test_posterior_arviz.py
	uv run --no-sync pytest tests/test_structural_artifacts.py::test_netcdf_roundtrip_refuses_a_changed_structural_model

# Run after installing both jax and torch; no skipped native cases are accepted.
qualify-native-posterior:
	uv run --no-sync python -c 'import jax, torch'
	uv run --no-sync pytest tests/test_posterior_backends.py

# Task keys must replay on both real backends; missing extras fail this recipe.
qualify-rng:
	uv run --no-sync python -c 'import jax, torch'
	uv run --no-sync pytest tests/test_rng_backends.py

# Native fitting qualification must fail if NumPyro/JAX are not installed.
qualify-numpyro:
	uv run --no-sync python -c 'import jax, numpyro'
	uv run --no-sync mypy
	uv run --no-sync pytest tests/test_numpyro_settings.py tests/test_numpyro_nuts.py

example-numpyro:
	uv run --no-sync python examples/numpyro_normal.py

# Modern named diagnostics require the reference and an actual native posterior.
qualify-diagnostics:
	uv run --no-sync python -c 'import arviz_stats.base, jax, numpyro'
	uv run --no-sync mypy
	uv run --no-sync pytest tests/test_mcmc_diagnostics.py tests/test_mcmc_diagnostics_native.py
	uv run --no-sync python examples/mcmc_diagnostics.py

# Fixed coverage planning and reference TARP; missing extras fail qualification.
qualify-coverage:
	uv run --no-sync python -c 'import tarp, deprecation, tqdm'
	uv run --no-sync mypy
	uv run --no-sync pytest tests/test_coverage_planning.py tests/test_tarp.py src/op_infer/coverage.py
	uv run --no-sync python examples/coverage_diagnostics.py --check

# Provider package (flepimop2-op_infer); needs flepimop2 available
provider-sync:
	cd flepimop2-op_infer && uv venv --clear && uv sync --dev

pytest-provider: provider-sync
	cd flepimop2-op_infer && uv run pytest

# CI entry point (no auto-fix)
ci:
	uv run ruff format --check
	uv run ruff check
	uv run mypy
	uv run pytest
	uv run mkdocs build --strict

# Source checks plus standalone wheels; no source path or core test fixtures leak in.
qualify-process-provider:
	#!/usr/bin/env bash
	set -euo pipefail
	uv pip install --python .venv/bin/python "${FLEPIMOP2_SPEC:-flepimop2==0.3.0}"
	uv run --no-sync ruff format --check flepimop2-op_infer
	uv run --no-sync ruff check flepimop2-op_infer
	MYPYPATH=src:flepimop2-op_infer/src uv run --no-sync mypy --strict --explicit-package-bases flepimop2-op_infer/src/flepimop2/process/op_infer/__init__.py flepimop2-op_infer/tests/test_provider_scaffold.py
	INFER_CLEANROOM="$(mktemp -d)"
	trap 'rm -rf "${INFER_CLEANROOM}"' EXIT
	uv build --wheel --out-dir "${INFER_CLEANROOM}/dist"
	uv build --project flepimop2-op_infer --wheel --out-dir "${INFER_CLEANROOM}/dist"
	uv venv --python 3.12 "${INFER_CLEANROOM}/venv"
	uv pip install --python "${INFER_CLEANROOM}/venv/bin/python" --find-links "${INFER_CLEANROOM}/dist" 'flepimop2-op-infer==0.0.1' "${FLEPIMOP2_SPEC:-flepimop2==0.3.0}" pytest
	cp -R flepimop2-op_infer/tests "${INFER_CLEANROOM}/tests"
	cd "${INFER_CLEANROOM}"
	unset PYTHONPATH
	"${INFER_CLEANROOM}/venv/bin/python" -c 'from pathlib import Path; import op_infer, flepimop2.process.op_infer as provider; assert ".venv" not in str(Path.cwd()); assert "site-packages" in str(op_infer.__file__); assert "site-packages" in str(provider.__file__)'
	"${INFER_CLEANROOM}/venv/bin/python" -m pytest tests -q

# Fast core API examples; optional workflows have required recipes below.
doctest:
	uv run pytest --doctest-modules src/op_infer

# Requires the study extra; do not accept a skipped example as qualification.
qualify-study:
	uv run --no-sync python -c 'import trade_study'
	uv run --no-sync pytest tests/test_documentation.py tests/test_sbc_runner.py tests/test_design_study.py tests/test_coverage.py
	just figures-check
	just qualify-masked

# Require actual scheduling and a standalone fixed-bank example.
qualify-masked:
	#!/usr/bin/env bash
	set -euo pipefail
	uv run --no-sync python -c 'import trade_study'
	uv run --no-sync pytest tests/test_masked_sbc.py
	MASKED_SBC_EXAMPLE="$(mktemp -d)"
	trap 'rm -rf "${MASKED_SBC_EXAMPLE}"' EXIT
	uv run --no-sync python examples/masked_normal_sbc.py "${MASKED_SBC_EXAMPLE}"

example-masked directory:
	uv run --no-sync python examples/masked_normal_sbc.py {{quote(directory)}}

# Requires the study extra; rendering is optional and stays outside the core.
figures-check:
	uv run --no-sync python examples/illustrate_normal_sbc.py --check

example-sbc directory:
	uv run --no-sync python examples/normal_sbc.py {{quote(directory)}}
