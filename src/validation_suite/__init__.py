"""Top-level package for MRMU validation suite."""

from .compare import compare_dataframes
from .performance import backtesting_report
from .results import ValidationResult
from .stability import csi_check, psi_check
from .statistical import vif_check

__version__ = "0.2.0"

__all__ = [
    "compare_dataframes",
    "vif_check",
    "psi_check",
    "csi_check",
    "backtesting_report",
    "ValidationResult",
]
