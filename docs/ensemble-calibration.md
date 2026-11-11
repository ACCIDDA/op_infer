# Per-member and mixture calibration

For an equal-weight mixture of independently trained estimators, calibration of the
**mixture** is what is reported (Hermans et al. 2022, arXiv:2110.06581, on
ensembling; Talts et al. 2018, arXiv:1804.06788, on SBC). `op_infer.ensemble_calibration`
adds a descriptive report of how the members relate to the mixture. A mixture of
members that differ in location can be wider than every member, and the
agreement of member locations and the spread of validation losses are separate
facts about training.

The SBC runner stores PITs and contraction only, not truths, means or member
structure, so the records come from your own loop over replicates.
`MemberRecords.from_draws(truth, draws)` builds them from `(R,)` truths and
`(R, M, S)` posterior draws (equal-weight pool of the members as the mixture;
randomised PITs with `atoms=True` and an `rng`):

```python
import numpy as np
from op_infer.ensemble_calibration import MemberRecords, member_calibration

rng = np.random.default_rng(0)
truth = rng.normal(size=200)  # R simulated truths
draws = rng.normal(size=(200, 3, 500))  # R replicates, M members, S draws
records = MemberRecords.from_draws(truth, draws)
report = member_calibration({"theta": records}, validation_losses=[1.0, 1.2, 0.9])
print(report["n_members"], report["estimands"]["theta"]["z_sd_gap"])
```

`member_calibration({name: MemberRecords(...)}, validation_losses=..., n_failed=...)`
returns, per estimand:

- for each member and the mixture, the centred SD, the mean and the root mean
  square of the standardised error `(truth - mean) / sd` (read `z_sd` with
  `z_mean`: a biased member of the right width has `z_sd` near 1) and the coverage
  of the central 50% and 90% intervals from the PITs;
- `z_sd_gap`, the smallest member `z_sd` minus the mixture `z_sd` (positive when
  the mixture's standardised errors are smaller than every member's);
- `agreement`, the median over replicates of the range of member means in units
  of the mixture SD (it grows with the number of members);
- both are `None` for one member, and the standardised-error fields are `None`
  for estimands with atoms (their SDs may be zero).

Across estimands it counts rank flags (Benjamini-Hochberg, through
`calibrate.aggregate`) per member and for the mixture, passes through the
replicate count, failed count and the `underpowered` banner, and reports the
spread of the validation losses. `members_with_flags` has no multiplicity control
across members (about `1 - (1 - alpha)**M` per estimand for calibrated members
sharing the same truths), so it is not evidence of member miscalibration by
itself. **Nothing is gated**: the report is a description and does
not change any gate decision. Failed replicates are not records; count them
separately, as in the SBC runner.
