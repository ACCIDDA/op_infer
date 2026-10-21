"""Observation models shared by the likelihood and the simulator.

Status: planned (no public API yet). See ``docs/design/proposal.md`` and the issue
tracker.

Scope: Poisson, NB2 (one documented parameterisation), Normal/log-Normal,
Binomial/beta-binomial, Bernoulli with sensitivity and specificity; left/right/interval
censoring with analytic gradient and curvature; iid or AR(1) dependence; missingness
masks; per-stream affine measurement maps. Each model exposes log_prob(latent, y, mask),
sample(latent, key) and metadata recording the effective behaviour.

"""

__all__: list[str] = []
