# op_infer

`op_infer` provides inference interfaces and checks for scientific simulators. It works with
[`op_system`](https://github.com/ACCIDDA/op_system) (model specification),
[`op_engine`](https://github.com/ACCIDDA/op_engine) (simulation engines) and
[`trade-study`](https://github.com/jcm-sci/trade-study) (experiment orchestration).
The implementation includes:

- Immutable provenance/likelihood-target references and native posterior containers.
- Identified callable and optional compiled fixed-step forward adapters with named latent outputs.
- Replayable random streams with separate parameter and simulation addresses.
- Optional native NumPyro NUTS and masked-cohort sbi NPE adapters.
- Borrowed observation banks and guarded native NPE training checkpoints.
- Named rank-normalized R-hat and bulk/tail ESS through optional ArviZ Stats.
- Rank calibration with atom-aware PIT, multiplicity, power floors and counted failures.
- Central-interval coverage with marginal Wilson intervals on the study extra.
- Fixed coverage-rule replicate planning and optional reference TARP curves.
- Fixed-bank SBC and held-out design-recovery workflows on trade-study.
- Masked-unit SBC with stratum counts and paired posterior-width comparisons.
- Prespecified, seeded scalar gate records with adequacy checks and amendment disclosures.
- A separate GPL flepimop2 process provider.

The package is pre-alpha (0.0.1) and unreleased. PyPI publication is on hold. The
[documentation](https://accidda.github.io/op_infer/) distinguishes implemented
interfaces and executable examples from the planned architecture. The implemented
surface includes a deliberately bounded catalog of five independent scalar or
fixed-shaped iid prior families, seven iid observation-law families, scoped
scaling assessments, fixed-independent-variance Gaussian mean information and
local spectra. General transforms, hierarchical or dependent priors, correlated
or AR(1) observations, general interval censoring and likelihood integration,
LOO/PSIS, automatic calibration-gate qualification, general identifiability and
EVSI remain planned. Other inference libraries have reserved extras, not
implemented adapters. Implemented adapters wrap their native libraries; no
samplers are reimplemented.

## Motivation

The package draws on repeated implementation work and reported failures in several modeling projects:
- priors maintained twice, once per backend;
- simulator and likelihood noise models drifting apart;
- point masses mis-ranked in SBC;
- coverage checks at too few replicates;
- test cohorts outside the training distribution;
- mixed-provenance simulation shards;
- importance samplers with effectively infinite variance;
- −∞ units swamping comparisons.

It aims to reduce that duplication. Its implemented statistical-model surface
is deliberately bounded; the capability guide identifies remaining interfaces.

## Development principles

These guide implementation. They include planned work as well as existing
interfaces; consult the capability guide for current support.

1. Array-API polymorphism for the data model (parameters, transforms, priors, observation models, per-unit
   integrals); backend-native samplers behind adapters.
2. A new module needs implementations in at least two projects or an active consumer issue.
3. One observation object per data stream, used by simulation and likelihood alike.
4. Provenance on every artefact; loaders refuse mismatches.
5. Specify gate criteria before examining results and record later amendments.
6. Failures are counted, never dropped.
7. Explicit, independent random streams.

## Layout

| Path | Contents |
|---|---|
| `src/op_infer/` | MIT core: provenance/posteriors, `forward`, `rng`, `calibrate`, `sbc`, `compare`, `design`, optional `adapters`; other statistical modules are design placeholders |
| `flepimop2-op_infer/` | GPL-3.0-or-later flepimop2 provider (process and parameter modules), as for `flepimop2-op_system` and `flepimop2-op_engine` |
| `docs/design/proposal.md` | Package architecture and planned interfaces |

## Install (development)

```bash
uv sync --dev            # core
uv sync --dev --extra jax --extra torch   # backends
just ci                  # format check, lint, mypy, tests, docs
```

Optional extras: `jax`, `torch`, `numpyro`, `blackjax`, `sbi`, `pyabc`, `diagnostics`, `tarp`, `arviz`, `study`
(trade-study), `op` (op_system + op_engine), `julia`, `pypesto`.

This public repository is source-only pre-alpha software. Both distributions
retain `Private :: Do Not Upload`; no PyPI package, GitHub release or archive
deposit is implied. Report suspected vulnerabilities through the
[security policy](SECURITY.md).

## Licence

Core: MIT. Provider (`flepimop2-op_infer`): GPL-3.0-or-later, because it imports flepimop2.
