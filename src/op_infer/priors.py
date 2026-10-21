"""Declarative priors, their backend lowerings and prior-predictive checks.

Status: planned (no public API yet). See ``docs/design/proposal.md`` and the issue
tracker.

Scope: Distribution specs with Array-API log_prob/sample; JointPrior (independent,
hierarchical, structured MVN); prior_hash; lowerings to numpyro, torch (sbi-ready,
support-validated) and scipy, covered by a cross-backend conformance suite;
PlausibilityRule with counted rejections shared by training and test simulation;
prior-coherence check.

"""

__all__: list[str] = []
