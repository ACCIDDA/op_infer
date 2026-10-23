# Package architecture and planned interfaces

`op_infer` connects scientific simulators to established inference libraries.
The [capability guide](../capabilities.md) lists implemented interfaces and their
qualification boundaries. This architecture note also describes planned work.

## Package boundaries

- OP System owns model specification, compiler artifacts and latent observables.
- OP Engine owns numerical simulation, observation evaluation and sensitivities.
- OP Infer owns statistical observation models, priors, inference adapters,
  posterior artifacts and calibration summaries.
- trade-study supplies replicate orchestration, scoring and experiment comparison.
- The separate GPL provider connects these interfaces to flepimop2; the MIT core
  never imports flepimop2.

Samplers and density estimators remain native library implementations. Optional
adapters wrap NumPyro and sbi; other reserved extras are not implemented adapters.
The package supplies explicit inputs and guarded outputs at those boundaries.

## Implemented foundations

Native posterior containers preserve named sample/event axes and separate seed
members from MCMC chains. Immutable provenance records structural model identity,
prior/design/observation targets, numerical settings and estimator/runtime identity.
Loaders require independently expected identities before reusing artifacts.

Identified forward callables and a bounded compiled fixed-step CPU adapter share
the same provenance. Task-addressed random streams separate parameter,
simulation and estimator randomness. Fixed-bank SBC and design studies retain
failed and missing attempts rather than silently changing their denominators.

Rank aggregation, central-interval coverage, optional TARP curves, named MCMC
diagnostics and saved scalar criteria have separate documented scopes. A finite
diagnostic or a saved criterion does not certify inference accuracy.

## Implemented and planned statistical interfaces

The implemented bounded catalog supplies five independent scalar or fixed-shaped
IID prior families and seven IID observation-law families. The same observation
objects define simulation and likelihood evaluation for their documented scopes,
with explicit backend lowerings and independent conformance checks.

General parameter spaces that declare names, shapes, support and coordinate
transforms remain planned. So do transformed, hierarchical and dependent prior or
observation structures beyond the bounded independent/IID catalog.

Further planned work includes per-unit integrated likelihoods, LOO/PSIS, profile
likelihood and optional structural-identifiability interoperability. Each addition
needs a bounded target, independent numerical/statistical references and concrete
consumers. Reserved modules and extras do not establish support.

The [likelihood-target contract](likelihood-targets.md) governs approximate and
augmented targets, adequacy and failed-unit accounting. Numerical identity and
derivative claims follow the [numerical provenance contract](numerical-provenance.md).

## Backend and execution policy

Python remains the API. Numerical data stay in their native Array API namespace;
library-specific adapters are optional. Host serialization and device transfer
are explicit boundaries. CPU qualification does not establish CUDA support, and
DLPack interchange does not preserve an autodifferentiation graph by itself.

Measure complete representative workloads before choosing additional acceleration
implementations. The [execution roadmap](execution-roadmap.md) describes interface
dependencies and measurement requirements. Julia is optional interoperability;
full language ports and new sampler implementations are outside this plan.

## Distribution

The project is pre-alpha and unreleased. Development installation uses the source
checkout. Both distributions carry a `Private :: Do Not Upload` classifier while
PyPI publication is on hold. Publishing requires a separate maintainer decision.
