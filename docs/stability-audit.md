# Monte Carlo stability audit

A contrast between two models is a sum over units of paired per-unit predictive
score differences, each estimated by Monte Carlo (often importance sampling with
heavy-tailed weights). `op_infer.audit` checks the reproducibility of such a
total before it is read, on a sample of units (Vehtari, Gelman and Gabry 2017,
*Statistics and Computing* 27:1413; Sivula, Magnusson, Matamoros and Vehtari
2025, *Bayesian Analysis*, on the uncertainty of elpd differences).

1. **Choose the sample from a tail diagnostic only.**
   `audit.audit_sample(diagnostic, edges=..., per_stratum=..., seed=...)` takes a
   per-unit diagnostic (for example the largest Pareto shape over the models) and
   draws a stratified sample. It has no argument through which a score could
   enter (enforced by its signature), so the sample cannot depend on the scores. A non-finite diagnostic goes
   to the top stratum, never out of the frame.
2. **Re-score the sample** with independent random keys and larger Monte Carlo
   settings, using your own scorer.
3. **Estimate.** `audit.stability_estimate(original, audit, sample)` takes the
   per-unit contrast values of the sampled units in both scorings and returns the
   estimated shift of the total, its sampling standard error (finite population
   correction included) and `mc_sd`, a conservative standard deviation of the
   Monte Carlo noise of the total over all units. `var(audit - original)` holds
   the noise of both runs, so `mc_sd` overstates the noise of the original run.
4. **Label.** `audit.stability_label(estimate, se, max_mc_ratio=..., max_shift_ratio=...)`
   returns `stable` or `not_assessable`, never `tied`. Stable needs no failed
   audited unit, `mc_sd <= max_mc_ratio * se` and
   `|shift| + z * shift_se <= max_shift_ratio * se`, with `se` the standard error
   of the original total. An audit too small to bound the shift is not
   assessable. The ratios have no defaults: they are reading-rule choices to fix
   before the audit is read.

A nonempty stratum with fewer than two usable audited units makes the estimate
`insufficient` (NaN errors) and the label `not_assessable`. A sampled unit with a non-finite value in either scoring is counted in
`n_failed` and left out of the moments; any failure makes the contrast not
assessable. The estimates assume finite-variance, independent per-unit Monte Carlo errors
and independent random keys in the two scorings; correlated errors, shared keys
and heavy-tailed importance-sampling errors (Pareto shape above 0.5) can make
`mc_sd` understate the noise. A stable label means the total is reproducible
against an independent rerun, not that it is accurate. `z` (default 1.645) is the
only default of the rule. The package does not score units; use your estimator
and, for paired contrasts, `compare.paired_predictive_contrast`
([design studies](design-studies.md)).
