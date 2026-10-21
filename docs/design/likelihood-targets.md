# Likelihood targets and adequacy

**Contract, 2026-10-15.**
This contract specifies requirements for inference adapters, per-unit integrals
and unit-level predictive comparisons. It includes planned interfaces; consult
the [capability guide](../capabilities.md) for implemented support.
It supplies the target requirements for the [package architecture](proposal.md).
The target obligations below govern future inference implementations. Shared provenance/target/numerical
metadata and a guarded provenance JSON loader are now implemented as described in
[numerical provenance](numerical-provenance.md). Posterior draws, diagnostics and sampler APIs remain
planned; metadata declarations do not qualify those implementations.

## Target identity comes before the adapter

Write the intended posterior as `π(θ | y) ∝ p(θ) L(θ)`. For conditionally independent units,
`L(θ) = ∏ᵢ ∫ p(yᵢ | zᵢ, θ) p(zᵢ | θ) dzᵢ`.
Record which of these targets an adapter actually uses:

| Kind | Evaluated target | Claim allowed |
|---|---|---|
| `analytic` | `p(θ) L(θ)`, with an analytic likelihood | Analytic likelihood for the declared model; floating-point evaluation and sampler convergence still need qualification. |
| `deterministic_approximation` | `p(θ) L̃(θ; a)` for fixed approximation settings `a` | Inference for that approximation, with its error evidence. Quadrature, Laplace and numerical forward-model error belong here when they alter the intended likelihood. |
| `frozen_random_approximation` | `p(θ) L̂(θ, u₀; a)`, holding one draw set fixed | Inference conditional on the recorded realization. Common random numbers can improve reproducibility and smoothness; they do not integrate out `u₀`. |
| `augmented_pseudo_marginal` | `p(θ) L̂(θ, u; a) mθ(u)` on `(θ, u)` | The intended marginal only under the conditions below and a qualified invariant transition kernel. |

Exactness is relative to the declared statistical model, likelihood and prior. An unbiased estimator
of a discretized or truncated likelihood does not establish exactness for a different, ideal model.
Do not infer a target kind from an adapter name such as `NUTS` or from a finite diagnostic value.

### Augmented-target requirements

For every supported `θ`, `mθ` must be a normalized auxiliary law and the likelihood estimator must
be nonnegative, with `E[mθ][L̂(θ, U)] = L(θ)` and a proper posterior normalizer. Unbiased **log**
likelihood is insufficient. The transition must preserve the joint target; rejected proposals retain
the current auxiliary state and its estimate. Refreshing the current estimate arbitrarily is a
different algorithm. Record the law, proposal/update kernel, initialization and rejection behavior.
These conditions follow the extended-target construction of
[Andrieu and Roberts (2009)](https://arxiv.org/abs/0903.5480).

Each unit estimate being unbiased does not by itself make their product unbiased. Conditional
independence of the units' auxiliary draws is one sufficient construction; a correlated construction
needs its own joint-unbiasedness argument. Sharing draws across parameter evaluations and sharing
them across units are different choices. Raw full-support importance estimates can qualify;
self-normalization, weight smoothing/clipping, adaptive stopping and data-dependent selection need
separate arguments. Diagnostics cannot supply those arguments.

Before using a mature HMC/NUTS implementation, specify a continuous auxiliary parameterization and
the differentiable joint log density, including the auxiliary density, transforms and Jacobians.
Gradients must cover both parameters and auxiliary variables. The published pseudo-marginal HMC
construction uses standard-normal auxiliary variables and differentiable likelihood estimates;
discrete resampling and zeros need additional treatment. Nonnegativity sufficient for some MH
constructions does not automatically make the log target usable by HMC.
[Alenlöv, Doucet and Lindsten](https://arxiv.org/abs/1607.02516) provide the reference construction.

An augmented-target inference adapter must wrap and qualify a suitable optional library kernel. Merely supplying frozen or freshly
randomized per-unit log estimates to ordinary parameter-only NUTS does not implement this joint target.
Until that qualification exists, report the actual approximation and leave the augmented claim unset.
No sampler is implemented by this contract.

### Integration support

Importance proposals must cover the nonzero integrand, with evaluable densities and the correct
normalization. A finite integration box usually computes `L_B(θ)`, not `L(θ)`. Record its coordinates,
bounds, transformations and omitted-mass treatment. Renormalizing the latent prior inside a box
defines yet another likelihood; it does not restore omitted likelihood mass. A tail bound is error
evidence for an approximation, not an exact correction. An exact claim requires a valid correction
or a proof that the integrand vanishes outside the box. Pilot adaptation must preserve the stated
estimator law and proposal support, including any pilot randomness used by the joint construction.

## Importance estimates: evidence and outcomes

A per-unit integration implementation must retain a record per unit, parameter evaluation and attempt. Store the method and proposal
identity, mode-search settings/outcome, draw count, finite and zero-weight counts, invalid-weight counts,
estimate, ESS definition/value, tail-fit status and Pareto `k̂`, uncertainty estimate when available,
diagnostic library/version, and policy identity. For dependent draws include the dependence correction.
Store missing diagnostics as missing with a reason, never as zero.

The policy is declared before inspecting results: required diagnostics, minimum effective sample size,
tail threshold and its sample-size convention, uncertainty tolerance and allowed recovery actions.
For PSIS, the documented sample-size threshold is `min(1 - 1/log10(S), 0.7)`; insufficient sample sizes
and failed fits need explicit outcomes. ESS alone cannot establish reliability of heavy-tailed weights.
Use mature implementations rather than implementing PSIS here.
[Vehtari et al. (2024)](https://www.jmlr.org/papers/v25/19-556.html) and the
[loo diagnostic documentation](https://mc-stan.org/loo/reference/pareto-k-diagnostic.html) motivate
these diagnostics. Their thresholds are diagnostics, not exactness certificates.

| Outcome | Meaning | Reporting |
|---|---|---|
| `readable=true, passed=true` | All required, interpretable diagnostics meet the declared policy. | Eligible for the declared use, subject to target and provenance qualification. |
| `readable=true, passed=false` | Diagnostics are interpretable and show inadequacy, such as excessive `k̂`, low ESS or excessive MC error. | Retain the estimate for diagnosis; do not use it as an adequate reported unit score. |
| `readable=false, passed=null` | The estimate or a required diagnostic cannot be interpreted, such as invalid weights, all-zero sampled weights, an insufficient tail sample or a failed tail fit. | Retain counts and reason; no pass or adequate score. |

`-∞` log weights represent valid zero weights; they are counted. NaN and `+∞` log weights are invalid for
ordinary normalized IS. All-zero sampled weights cannot distinguish a true zero integral from a missed
mode. An independently established zero likelihood can legitimately have log likelihood `-∞`; record
that support result separately and count it as a non-finite unit score in comparisons.

A failed/unreadable PSIS attempt followed by raw IS remains failed/unreadable for reporting. Raw IS may
provide a diagnostic number, with a failed-tail marker (historically `k̂=∞`), but cannot supply the missing
reliability evidence. A new proposal, additional qualified sampling or a direct LOUO refit may create a
new attempt with its own evidence. Preserve the original attempt, recovery reason, settings and cost;
never overwrite it with a successful-looking fallback. MCMC recoveries require their own per-replicate
R-hat/ESS assessment. A valid augmented target may mix poorly even if its unbiasedness argument holds;
target validity and computational adequacy are separate requirements.

## Posterior and comparison provenance

The [posterior containers](../posteriors.md) and [numerical provenance records](numerical-provenance.md)
preserve these identities. Artifact loaders must retain their semantics,
independently of any configuration language:

| Required identity | Contents |
|---|---|
| Intended target | Prior, model/observation definition, data/design and ordered unit identities, latent integration measure/support, normalization conventions. |
| Evaluated target | One of the four kinds above; content identity of approximation algorithm/settings, numerical forward contract and any truncation/correction. |
| Randomness | Auxiliary law and joint update/retention contract for augmented targets; actual frozen draw-set content identity, generator/version, backend/dtype and stream identity for frozen approximations. A seed alone is insufficient. |
| Qualification | Referenced unbiasedness/support/differentiability argument where applicable, independent reference tests, diagnostic policy and all attempt outcomes. A declaration is not proof. |
| Execution | Estimator stage/settings, code commit, backend/library versions, precision and immutable artifact identities. |

Loaders and shard merges must refuse missing or mismatched required identity against the caller's
expected identity. Legacy unknown identity stays unknown and cannot inherit an exactness claim.
Content identities cover canonicalized settings and referenced immutable artifacts, not just display
names. The [numerical provenance guide](numerical-provenance.md) describes
the current formats, versioning and numerical derivative records.

Comparison artifacts retain **both** sides' target and approximation identities. Different models can
be compared deliberately; a loader must not require those two model identities to be equal. It must
require each to match its declared expected identity and verify a shared unit/population, observation
measure and evaluation protocol. Record deliberate differences, compatibility evidence and amendments.
Do not silently merge approximate and intended-target likelihoods or incompatible densities.

## Unit population and paired contrasts

The unit manifest and exclusions are declared before scores are inspected. Keep unit IDs, exclusion
reasons, stratum, per-side scores, support results and adequacy outcomes. Count raw non-finite scores
even for excluded units in the audit ledger. For the pre-included population report:

- `N_total`, `N_excluded` and `N_eligible = N_total - N_excluded`;
- counts non-finite on A, on B and on both (NaN and either infinity count);
- inadequate/unreadable attempts on each side, including failures with finite numeric scores;
- `N_paired_finite` and `N_paired_usable` (both finite and adequate), with retained unit IDs.

For example, five eligible units with A scores `[1, -∞, 3, NaN, 5]` and B scores
`[0, 2, -∞, NaN, 4]` have two non-finite scores on A, two on B and one on both.
Thus `N_paired_finite = 5 - 2 - 2 + 1 = 2` (units 1 and 5).
If both pairs are adequate, their mean A-minus-B contrast is 1 with denominator 2. If unit 5 is
inadequate despite finite scores, the usable denominator is 1 and the adequacy failure remains counted.

In general, `N_paired_finite = N_eligible - N_bad_A - N_bad_B + N_bad_both`.
The estimand of a finite-only contrast is conditional on both scores being finite; an adequacy-filtered
contrast is additionally conditional on both estimates being adequate. Report that conditional
estimand, the exact denominator and the eligible population denominator used for failure fractions.
Neither contrast recovers the full-population mean without further assumptions. Zero usable pairs
produce an unreadable contrast, not a zero difference. Post-result exclusions require a recorded
amendment and a new analysis identity; they cannot silently redefine the original population.
Scoring, uncertainty intervals and paired bootstrap execution belong in trade-study.
A unit-level comparison implementation must supply its ledger and qualified
inputs, including strata, to that reporting layer.

## Executable qualification

`tests/test_likelihood_contract.py` uses a synthetic Gaussian random-effects model:
`zᵢ | θ ~ Normal(θ, τ²)`, `yᵢ | zᵢ ~ Normal(zᵢ, σ²)`.
Completing the square gives the independent analytic marginal
`Lᵢ(θ) = NormalPDF(yᵢ; θ, τ² + σ²)`.
Gauss-Hermite quadrature of the original latent integral at two resolutions agrees with that result
and with the joint product across units. A bounded latent integral is independently checked by
Gauss-Legendre quadrature against the analytic marginal times the conditional latent probability
inside the box. Dividing by the prior box probability does not recover the full marginal.
No prior Monte Carlo is used for reported scores.

For a controlled auxiliary example let `s(θ) = 0.25 + θ²` and
`L̂(θ, u) = L(θ) exp(s(θ)u - s(θ)²/2)`, `u ~ Normal(0, 1)`.
The normal moment-generating identity gives `E[L̂] = L`, whereas
`E[log L̂] = log L - s(θ)²/2`. Quadrature independently verifies both.
Fixing `u` changes likelihood ratios because `s` depends on `θ`.
Conversely, `log L + s(θ)u` has an unbiased log estimate but its exponent has expectation
`L exp(s(θ)²/2)`, which distorts the target.

For a finite auxiliary box `[-T,T]`, integration of this augmented density yields
`L(θ) [Φ(T - s(θ)) - Φ(-T - s(θ))]`, not `L(θ)`. Gauss-Legendre quadrature confirms the
omitted mass and its parameter dependence; conditional-law renormalization does not remove it.
For two units sharing the same auxiliary draw, multiplying the positive unbiased estimates instead
gives expectation `L₁ L₂ exp(s₁ s₂)`. Tests compare this with independently integrated auxiliary draws.
These are mathematical qualification fixtures, not implemented distribution families, calibration gates
or evidence that an unimplemented inference kernel is correct. Future adapters must add their own
invariance, derivative, numerical and sampler-adequacy qualification under this contract.
