# Scoped structural and local information reports

`op_infer.identify` separates existing compiler scaling evidence from local
Gaussian information at one design and parameter point. Neither a small
eigenvalue nor a reported scaling count supplies an identifiable parameter
dimension or a model-selection penalty.

## Compiler reports

`adapters.op_identify.scaling_assessment(compiled)` copies the compiler's existing
action basis, parameter-base order, invariant monomials and probe record. It
performs no second gauge detection. The source is the compiler's schema-1
structural identity; an independently expected identity can be required.

Unassessed and unsupported reports have no ranks. A supported analysis returning
zero generators has `status="reported"` and rank zero, with its restricted
scaling scope retained. Operator/history/table and gauge-fixed execution require
separate contracts and return unsupported. Parameter-base actions are distinct
from expanded physical parameter sensitivity coordinates.

Finite probes retain seed, scales, observables and tolerances; the producer's
structural homogeneity flag is separate. Initial conditions and prior effects
remain unassessed by this record. No coordinate map, prior pushforward or anchor
is applied. General transformations must preserve support, induced priors and
Jacobians in their separately qualified parameterization interface.

## Native finite-design Gaussian information

For derivatives J of the actual measurement mean and known fixed independent
standard deviations, `gaussian_mean_information` returns `J.T @ diag(1/sigma**2)
@ J`. All leading cells represent independent measurements. The result is a
data-only sum, with no prior curvature or nuisance elimination.

```python
import numpy as np
from op_infer.identify import (
    InformationContext,
    gaussian_mean_information,
    local_spectrum,
)
from op_infer.posterior import ArtifactReference


def ref(name):
    return ArtifactReference.from_bytes(name.encode(), kind="toy-v1")


context = InformationContext(
    source_model=ref("linear-mean"),
    design=ref("two-independent-measurements"),
    parameter_point=ref("a,b=0,0"),
    initialization=ref("no-ODE-initial-state"),
    observation=ref("known-sigma=2"),
    numerical=ref("analytic-mean-derivative"),
    qualification=ref("independent-linear-reference"),
    parameter_names=("a", "b"),
    parameter_units=("unitless", "unitless"),
    observable_names=("a+b", "a-b"),
)
information = gaussian_mean_information(
    np.array([[1.0, 1.0], [1.0, -1.0]]),
    np.array(2.0),
    context=context,
)
assert np.allclose(information.matrix, np.eye(2) / 2)
report = local_spectrum(information, rtol=1e-8, atol=1e-12)
assert report.local_rank == 2
```

Sensitivity shape is `(..., observables, parameters)`. Sigma is native scalar or
has the full measurement shape; an optional native boolean mask has that exact
shape. Missing NaNs are replaced before arithmetic. Invalid selected values,
nonpositive noise or no observed cells return native `valid=False` and a NaN
matrix. They cannot receive a spectrum or an invented rank-zero result.

The default `InformationLimits` bound is 128 parameters and 16,777,216 sensitivity
elements. Computation retains a weighted sensitivity temporary and a dense
parameter matrix, with no observation-by-observation covariance matrix. Borrowed
input storage, native AD graphs, compiler memory and whole-process RSS are not
bounded by those limits. NumPy/JAX/CPU Torch float32/64 and JAX JIT/Torch native
gradients are qualified; additional devices require separate evidence.

The native matrix remains available for AD/JIT. `local_spectrum` is an explicit
host snapshot/reporting boundary, preserving units, context, actual eigenvalues
and caller-declared absolute/relative rank tolerances. Its rank is local and
coordinate-dependent. For example, the globally injective mean `theta**3` has
zero local information at zero; that alone establishes no global symmetry.

## Consume engine measured sensitivities

`information_from_sensitivities(solution, observation, context=context)` accepts
the reviewed engine `SensitivitySolution` and an uncensored iid Normal law. The
source model, numerical execution, ordered parameter/output layouts and exact
observation-law identity must match the expected context. Known affine sensor
slopes multiply the measured sensitivities; fixed offsets have zero derivative.
The law's fixed scale supplies variance.

The engine already includes initial-condition and direct measurement-parameter
terms through `dg/dy @ S + dg/dtheta`. Keep an independent initialization-function
reference and parameter-point reference in the context. Native arrays remain
borrowed; those references are caller-supplied content claims, not automatic
hashes or authentication of numerical inputs. Numerical qualification inherits
the solver mesh and derivative producer. `sensitivity_execution_reference`
records relevant settings, actual mesh, dtype and layouts without copying arrays.

Count/censored/correlated observations, parameter-dependent variances, nuisance
profiling, general transformed coordinates, observed posterior curvature and
external structural-rank tools retain their separate scopes. An RHS state
Jacobian is not a measured-mean parameter sensitivity or a Fisher matrix.

## Qualification

`just qualify-identify` requires actual compiler derivatives, engine forward
sensitivities, JAX and Torch. The lowest-dependency CI lane installs the exact
released interface floors, `op-system==0.10.0` and `op-engine==0.9.0`.
The independent references cover unequal noise, masked NaNs, native dtype and
gradients, matrix/rank guards, source/layout mismatches and a measured ODE
response with both initial-only and measurement-only parameters. A nonlinear
counterexample checks that a local rank result never creates a structural claim.
Current/minimum statistical CI also executes the native algebra tests.
