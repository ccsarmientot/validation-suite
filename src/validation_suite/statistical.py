import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .results import ValidationResult


def vif_check(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 10.0,
) -> ValidationResult:
    """
    Computes VIF for each feature. SR11-7 standard threshold = 10.
    Values > 10 indicate problematic multicollinearity.
    """
    X = df[feature_cols].dropna()

    if len(feature_cols) == 1:
        return ValidationResult(
            test_name="vif_check",
            status="PASS",
            summary_df=pd.DataFrame(
                {
                    "feature": feature_cols,
                    "VIF": 1,
                }
            ),
            details={"threshold": threshold, "flagged_features": [0]},
            warnings=[],
        )

    vif_data = pd.DataFrame(
        {
            "feature": feature_cols,
            "VIF": [
                variance_inflation_factor(X.values, i) for i in range(len(feature_cols))
            ],
        }
    )
    vif_data["flag"] = vif_data["VIF"].apply(
        lambda v: "HIGH" if v > threshold else ("MODERATE" if v > 5 else "OK")
    )

    flagged = vif_data[vif_data["flag"] == "HIGH"]["feature"].tolist()
    status = "PASS" if not flagged else "FAIL"
    warnings = [f"High VIF (>{threshold}) detected in: {flagged}"] if flagged else []

    return ValidationResult(
        test_name="vif_check",
        status=status,
        summary_df=vif_data,
        details={"threshold": threshold, "flagged_features": flagged},
        warnings=warnings,
    )
