"""Top-level package for MRMU validation suite."""

from .compare import compare_dataframes
from .statistical import vif_check
from .stability import psi_check, csi_check
# from .performance import backtesting_report
from .results import ValidationResult

__version__ = "0.1.2"

__all__ = [
    "compare_dataframes",
    "vif_check",
    "psi_check",
    "csi_check",
    # "backtesting_report",
    "ValidationResult",
]