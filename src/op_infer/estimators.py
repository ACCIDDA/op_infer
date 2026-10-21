"""Adapters to mature inference libraries.

Status: a bounded optional native NumPyro NUTS adapter is implemented in
``op_infer.adapters.numpyro``. A bounded CPU masked-cohort NPE adapter is
implemented in ``op_infer.adapters.sbi``. Other adapters and the registry
remain planned.
See ``docs/numpyro.md`` and ``docs/design/proposal.md``.

Scope: A registry of thin adapters, each an optional extra: numpyro NUTS/SVI/Predictive,
blackjax SMC/MCLMC, sbi NPE/NLE/MNLE/NRE/BNRE (with a masked set embedding and seed
ensembles), pyabc ABC-SMC, scipy MAP/Laplace, pseudo-marginal NUTS. Every adapter
returns a Posterior with estimator stage and provenance; non-flow posteriors carry
per-replicate MCMC adequacy. Samplers are never reimplemented.

"""

__all__: list[str] = []
