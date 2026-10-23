# Sensor calibration from independent reference readings

This worked example asks a small, concrete question: what offset and gain map
known reference values to the readings from a scalar sensor? It demonstrates how
`op_infer` keeps an observation law, independent priors, random streams and
content provenance explicit. Every value in the example is invented. The result
is neither a calibration certificate nor a recommended laboratory protocol.

## The study in plain language

Suppose a sensor has an approximately linear response over a narrow operating
range. At reference value $x_i$, its reading is

$$
y_i = b + g x_i + \epsilon_i,
\qquad \epsilon_i \sim \mathrm{Normal}(0, \sigma^2).
$$

Here, $b$ is an offset in sensor units, $g$ is the gain in sensor units per
reference unit, and the illustrative residual standard deviation is fixed at
$\sigma=0.08$ sensor units. The script generates independent readings at several
known reference levels. It reserves a second, independently generated set of
intermediate reference levels for a descriptive prediction check.

The prior treats offset and gain as independent Normal quantities. Because the
response is linear, the noise variance is known and the prior is Gaussian, the
posterior mean and covariance are available by exact conjugate algebra. The
example therefore does not introduce a new optimizer or sampler. It draws from
the resulting Normal posterior only to report how closely a finite, explicitly
seeded draw bank approximates the analytic mean.

## Run it

From the repository root, run the CPU-bounded smoke version:

```bash
uv run python examples/sensor_calibration_study.py --check
```

Run the slightly larger illustrative design with:

```bash
uv run python examples/sensor_calibration_study.py
```

Both commands print JSON. The full output records:

- the scientific question and model assumptions;
- the synthetic truth and number of calibration readings;
- posterior means, standard deviations and 95% marginal credible intervals;
- the ratio of posterior to prior standard deviation for each parameter;
- held-out standardized residuals and mean log predictive density;
- numerical diagnostics, the seed and named random streams; and
- hashes of the source, prior, observation model and generated data.

The `--check` mode uses six calibration readings, four held-out readings and 200
posterior diagnostic draws. The default uses fifteen calibration readings,
eight held-out readings and 4,000 diagnostic draws. Both modes run on a CPU.

## How `op_infer` enters the workflow

`JointPrior` and `PriorParameter` declare named, independent prior coordinates.
`IIDObservation(family="normal", scale=0.08)` supplies one model for generating
and scoring independent readings. `RandomKey` assigns distinct, replayable
streams to the calibration readings, held-out readings and posterior-draw check;
the script never seeds or consumes NumPy's global random state.

`ArtifactReference` identifies exact source and data bytes. A content hash is
useful for detecting a changed artifact, but does not prove that the model or
data are scientifically appropriate. The script therefore reports
`illustrative_success`, rather than a calibration pass.

## Reading the result

A posterior-to-prior standard-deviation ratio below one means the synthetic
readings reduced marginal uncertainty under this model. It does not establish
that the real sensor is linear or that the stated noise scale is correct. The
held-out readings answer a narrower question: were separately generated toy
readings surprising under the fitted toy model? With only four or eight
readings, interval counts are descriptive and cannot establish coverage.

The example checks that the design matrix identifies both offset and gain, that
all supplied values are finite, that the posterior precision is not severely
ill-conditioned, and that the posterior covariance is positive definite. An
anticipated failure is printed as structured JSON with a nonzero exit status;
failed inputs are not silently dropped or retried.

## Limits and next steps

The residual standard deviation is fixed rather than inferred. Readings are
independent, Gaussian and uncensored, with a linear response over normalized
reference values. The example does not implement correlated errors,
hierarchical priors, censoring, drift or automatic design optimization. Those
features require a different, explicitly supported model and new qualification.

For a real study, replace the invented units and design with a documented
reference procedure; predefine the operating range, replicate structure and
acceptance criteria; retain environmental and instrument metadata; estimate or
otherwise justify the error model; and qualify the complete analysis on the
intended range. Independent test material and enough replicates to assess the
chosen performance criterion are scientific requirements, not consequences of
this software example.
