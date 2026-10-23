# op_infer

`op_infer` connects scientific simulators to inference libraries and calibration
checks. It is unreleased pre-alpha software; PyPI publication is on hold. Start with the
[implemented capabilities and executable examples](capabilities.md); the
[architecture note](design/proposal.md) also describes planned interfaces. The
site follows reviewed public `main`; it is documentation for a development
checkout, not a package-release channel.

Current workflows include [native posterior containers](posteriors.md),
[guarded numerical provenance](design/numerical-provenance.md),
[separate random streams](random-streams.md), [NumPyro NUTS](numpyro.md),
[masked-cohort sbi NPE](sbi-npe.md), [rank calibration](calibration.md),
[fixed-bank SBC execution](sbc-runner.md) and
[held-out design recovery](design-studies.md). Optional adapters have explicitly
bounded qualification. Scoped local-information and scaling reports are
implemented, while general structural identifiability, calibration certification,
LOO/PSIS and general posterior geometry remain planned. Metadata does not
certify numerical accuracy.
