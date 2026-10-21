# Saved gate records

`GateSet` records scalar criteria for an estimator stage, its context and a
random-stream address. Save the criteria before evaluating a result. Evaluation
checks the saved specification and context before invoking your callback with
the declared `RandomKey`; callbacks use that key for stochastic diagnostics.
The legacy `calibrate.gate` rank reading retains its existing behavior.

Each `Gate` has inclusive bounds and one role:

| Role | Meaning |
| --- | --- |
| `adequacy` | Whether the result can be read, including sampler diagnostics, replicate floors and failure limits. |
| `decision` | Whether a readable result meets the declared substantive criteria. |

Every set requires both roles. Missing and nonfinite metrics are unavailable.
`GateRecord.readable` requires every adequacy criterion to pass;
`GateRecord.passed` also requires every decision criterion. An underpowered run
cannot pass if the prespecified replicate floor is an adequacy criterion. There
are no default statistical thresholds: the project must choose and qualify them.

The executable `GateSet` docstring demonstrates registration, evaluation and
JSON replay. Import these classes from `op_infer.gates` or `op_infer.calibrate`.
Use separate paths and keys for distinct estimator stages. `write` is atomic
and refuses different criteria at an existing path. `evaluate` requires that
saved artifact and the expected context. Context and evidence are content
references; the caller must retain their exact underlying payloads.

`record.model_dump_json()` retains criteria, stage, key, input identity and all
criterion values, including unreadable or failed results. Load with
`GateRecord.model_validate_json` and read `readable`, `passed` and `statuses()`;
verdicts are derived again from the retained criteria, rather than trusting a
stored pass label. Evaluator exceptions propagate and cannot produce a pass.

## Amendments

Create a `GateAmendment` with the prior specification's `reference()`, an aware
timestamp, a reason and `seen_before_amendment`: references to every result
artifact already inspected, or an explicit empty tuple. Call `original.amend`
with that disclosure, the revised criteria and a declared key, then save the
result at a new path. The original file stays unchanged and the revision retains
its prior-artifact identity. Keep the linked prior specifications available.

The package records the author's disclosure; it cannot authenticate chronology,
prove that results were unseen or make an inspected bank independent. Version
control and study registration provide that evidence. Keep original readings
and label retrospective re-readings; use fresh independent replicates when the
prospective protocol requires them.

## Statistical scope

Saved criteria and reproducible evaluation do not establish calibration, power,
a false-failure rate or validity under optional stopping. Diagnostics, failure
policies, multiplicity, fixed or sequential sampling and threshold qualification
remain the consumer's statistical contract. A sampler-adequacy pass alone does
not establish posterior coverage. See the [coverage guide](calibration.md#central-interval-coverage)
for marginal interval precision and its limits.
