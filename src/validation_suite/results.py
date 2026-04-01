from dataclasses import dataclass, field
import pandas as pd

@dataclass
class ValidationResult:
    test_name: str
    status: str          # "PASS" | "FAIL" | "WARN"
    summary_df: pd.DataFrame
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"ValidationResult(test={self.test_name!r}, status={self.status!r}, warnings={len(self.warnings)})"

    def to_excel(self, path: str):
        """Export to Excel — útil para evidencia de auditoría."""
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.summary_df.to_excel(writer, sheet_name="Summary", index=False)
            meta = pd.DataFrame({
                "test_name": [self.test_name],
                "status": [self.status],
                "warnings": ["; ".join(self.warnings)],
            })
            meta.to_excel(writer, sheet_name="Metadata", index=False)