# Contributing

- One issue per change; link it in the PR. Use open issues to coordinate work.
- **Rule of two:** a new module or public function needs either two existing implementations to port
  (cite public references) or an active consumer issue. Keep confidential
  consumer evidence outside this repository.
- Array-API code goes through `array_api_compat.array_namespace`; no backend-specific imports in the core
  data model. Backend-specific code lives in adapters behind optional extras.
- Every distribution and observation family is covered by the cross-backend conformance suite
  (numpy, jax, torch) before it is exported.
- Gate and calibration code is tested on a posterior calibrated by construction, including one with a point
  mass.
- `just ci` must pass: ruff (format and lint), mypy (strict), pytest with doctests, MkDocs build.
- Typed public API; no samplers reimplemented.

## Executable documentation

Keep capability claims, API docstrings and complete examples in sync. Use small
deterministic doctests and independent reference checks. Run `just doctest` for
core examples, `just qualify-study` with the study extra, then `just ci`.
See [implemented capabilities](docs/capabilities.md).
