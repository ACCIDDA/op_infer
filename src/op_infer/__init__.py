"""op_infer: the statistical layer of the op stack.

Parameter spaces, priors, observation models, posteriors with provenance,
calibration gates, identifiability and design studies, wrapping mature inference
libraries (numpyro, blackjax, sbi, pyabc, scipy) rather than reimplementing them.
Stands alone; ``flepimop2-op_infer`` exposes it to flepimop2.

Examples:
    >>> import op_infer
    >>> op_infer.__version__
    '0.0.1'
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
