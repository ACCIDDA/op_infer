"""Log-densities and per-unit integrated likelihoods.

Status: planned (no public API yet). See ``docs/design/proposal.md`` and the issue
tracker.

Scope: LogDensity protocol (log prior + sum over units); per-unit integrals over unit
latents by Laplace, Laplace importance sampling (Levenberg-Marquardt with Gauss-Newton
curvature, moment-matched t, defensive mixture, Pareto k-hat) and prior Monte Carlo
(reference only), with a validation harness against high-sample references.


Inference adapters and per-unit likelihood/comparison workflows follow the target and
adequacy contract
in ``docs/design/likelihood-targets.md``. Frozen common random numbers define an
approximation; an exact pseudo-marginal claim requires a qualified augmented target.
Unbiasedness concerns the likelihood, not its logarithm, including the joint product
over units. Importance diagnostics and target validity are separate requirements.
"""

__all__: list[str] = []
