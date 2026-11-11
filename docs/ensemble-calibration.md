# Per-member and mixture calibration

For an equal-weight mixture of independently trained estimators, calibration of the
**mixture** is what is reported (Hermans et al. 2022, arXiv:2110.06581, on
ensembling; Talts et al. 2018, arXiv:1804.06788, on SBC). `op_infer.ensemble_calibration`
adds a descriptive report of how the members relate to the mixture. A mixture of
members that differ in location can be wider than every member, and the
agreement of member locations and the spread of validation losses are separate
facts about training.

`member_calibration({name: MemberRecords(...)}, validation_losses=...)` takes
per-replicate summaries that you have already produced for every member and for
the mixture (truth, posterior mean and SD, PIT; randomised PITs for atoms) and
returns, per estimand:

- for each member and the mixture, the SD and mean of the standardised error
  `(truth - mean) / sd` and the coverage of the central 50% and 90% intervals;
- `z_sd_gap`, the smallest member SD minus the mixture SD (positive when the
  mixture is wider than every member);
- `agreement`, the median over replicates of the range of member means in units
  of the mixture SD.

Across estimands it counts rank flags (Benjamini-Hochberg, through
`calibrate.aggregate`) per member and for the mixture, and reports the spread of
the validation losses. **Nothing is gated**: the report is a description and does
not change any gate decision. Failed replicates are not records; count them
separately, as in the SBC runner.
