# Saved calibration reading policies

`op_infer.readings` applies prospectively saved point and interval readings to
the same fixed replicate bank. Both panels remain visible, with the governing
choice and its reason declared in `CalibrationPolicy`. A pass under an interval
reading means compatibility with its declared criteria; it does not establish
equivalence or calibration.

The consumer supplies acceptable bands, minimum usable replicates, a failure
fraction budget, the number of distinct quantities allowed to violate criteria,
and a must-pass quantity set. Multiple failed criteria for one quantity consume
one quantity allowance. A must-pass failure overrides that allowance. The
library supplies no application thresholds.

## Evidence and metric definitions

The saved policy names every planned replicate. `CalibrationEvidence` must
partition that exact bank into usable IDs and `ReplicateFailure` records. Failure
reasons and diagnostic references remain in the result. Each supplied metric
uses the same complete usable cohort; changing the denominator by dropping a
failed replicate is refused. Per-quantity missing data require a separately
declared cohort/policy, rather than quietly reducing one metric's count.

Each `MetricEstimate` identifies its quantity, metric, method, usable count,
point and optional interval. The method reference must match the saved rule and
identify the formula, interval family/level, assumptions and any clipping or
other options. References record content claims; the producer owns correctness
of the metrics and the independence assumptions. Missing or nonfinite values
cannot silently become zero or passing evidence.

`wilson_coverage` delegates ordinary binomial coverage intervals to SciPy. Its
method identity includes the confidence level and a fixed independent usable
trial assumption. It is unsuitable for dependent replicates or a cohort selected
according to the coverage outcome. A zero-trial cohort produces an unavailable
estimate. Consumer-specific spread statistics and their interval assumptions
remain consumer inputs.

## Point, interval and practical-limit rules

A `CalibrationRule(view="band")` compares the point estimate with an inclusive
acceptable band in the point reading. In the interval reading it fails when the
metric's entire interval excludes the band. Overlap is compatibility, which can
be common with low precision. The replicate floor is therefore mandatory.

Other views always use the declared `point`, `lower` or `upper` value, in either
reading. This supports a practical limit on a supplied spread metric, including
a limit on its lower endpoint. The length of a coverage interval is a different
quantity and is never substituted for that metric. `applies_to` can restrict a
rule to one panel; each panel must still cover the complete quantity set.

Below `minimum_usable`, beyond `max_failure_fraction`, or with an unavailable
required metric/view, the affected reading is `insufficient_evidence`. It cannot
produce a pass/fail decision. Raw criteria violations remain visible beside the
adequacy reasons. With adequate evidence, failed quantities and the must-pass
set determine `pass` or `fail`.

## Registration and reporting

Call `policy.write(path)` before generating or inspecting results. It atomically
creates complete bytes and refuses to overwrite a different policy. Then call
`record = policy.evaluate(path, evidence)`: the saved policy, method identities,
context and complete bank must agree. `record.readings()` returns both panels;
`record.governing_reading` selects the prespecified panel. Full policy/evidence
JSON round-trips and recalculates decisions, so serialized verdicts cannot be
altered independently of their inputs.

For a prospective change, construct a new policy with a `GateAmendment`
referencing the previous policy, a timezone-aware timestamp, reason and explicit
references to any evidence already seen. Save it to a separate file. Local
registration cannot authenticate chronology or prove that fresh evidence is
independent; those are consumer and external version-control responsibilities.

## Executable finite-sample qualification

```bash
python examples/calibration_readings.py /tmp/public-reading-toy
```

The example saves an arbitrary public toy policy for 80 independent Bernoulli
trials, a band of 0.4 to 0.6, and a 95% Wilson interval. It enumerates every count
and weights decisions by exact binomial probabilities, then checks independent
seeded Monte Carlo cohorts at success probabilities 0.5 and 0.8. The output
includes the saved policy, both panels, source hash and failure probabilities.
These illustrative choices are not recommended application defaults.

The [retained toy receipt](assets/calibration-readings-toy.json) gives point and
interval false-failure probabilities of 5.6664% and 0.0183% at p=0.5. Detection
probabilities at p=0.8 are 99.9967% and 97.8332%, respectively. The reduced
false-failure rate comes with reduced detection probability in this example;
neither reading has a general superiority guarantee. This qualifies the
coverage-reading logic, not the operating characteristics of an arbitrary
consumer-supplied statistic or an entire posterior calibration procedure.

Tests additionally compare Wilson endpoints with an independent closed-form
reference, and exercise floor refusal, method/context mismatch, immutable
registration, practical endpoint limits, distinct-quantity allowances and
must-pass overrides. The existing rank aggregation API remains independently
documented in [rank calibration](calibration.md); callers must explicitly adopt
the saved reading policy.
