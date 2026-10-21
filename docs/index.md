# op_infer

`op_infer` connects scientific simulators to inference libraries and calibration
checks. It is unreleased pre-alpha software; PyPI publication is on hold. Start with the
[implemented capabilities and executable examples](capabilities.md); the
[architecture note](design/proposal.md) also describes planned interfaces.

Current workflows include [native posterior containers](posteriors.md),
[guarded numerical provenance](design/numerical-provenance.md),
[separate random streams](random-streams.md), [NumPyro NUTS](numpyro.md),
[masked-cohort sbi NPE](sbi-npe.md), [rank calibration](calibration.md),
[fixed-bank SBC execution](sbc-runner.md) and
[held-out design recovery](design-studies.md). Optional adapters have explicitly
bounded qualification. General calibration certification, LOO/PSIS and
identifiability remain planned; metadata does not certify numerical accuracy.
