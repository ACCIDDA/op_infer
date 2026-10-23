"""op_infer: the statistical layer of the op stack.

Provenance-bound forward and posterior records, a bounded catalog of independent
scalar or fixed-shaped iid priors and observation laws, calibration and design
workflows, and native adapters for declared subsets of mature inference
libraries. Stands alone; ``flepimop2-op-infer`` exposes it to flepimop2.

Examples:
    >>> import op_infer
    >>> op_infer.__version__
    '0.0.1'
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("op_infer")
except PackageNotFoundError:
    __version__ = "0+unknown"
