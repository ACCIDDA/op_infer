"""Scoped structural reports and finite-design local Gaussian information.

Optional compiler/engine consumers live in ``adapters.op_identify``. General
coordinate transformations, nuisance profiling, observed posterior curvature,
profile likelihood and external structural tools retain their separate scopes.
"""

from op_infer.geometry import (
    GaussianInformation,
    InformationContext,
    InformationLimits,
    LocalSpectrum,
    ScalingAssessment,
    ScalingGenerator,
    ScalingVerification,
    gaussian_mean_information,
    local_spectrum,
)

__all__ = [
    "GaussianInformation",
    "InformationContext",
    "InformationLimits",
    "LocalSpectrum",
    "ScalingAssessment",
    "ScalingGenerator",
    "ScalingVerification",
    "gaussian_mean_information",
    "local_spectrum",
]
