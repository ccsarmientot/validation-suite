import pandas as pd

from .results import ValidationResult


def compare_dataframes(
    df_reference: pd.DataFrame,
    df_challenger: pd.DataFrame,
    key_cols: list[str],
    numeric_tol: float = 1e-6,
    label_reference: str = "Model Owner",
    label_challenger: str = "Dry Run",
) -> ValidationResult:
    """
    Compares two DataFrames (e.g. model owner output vs validator dry run).

    Returns a ValidationResult with:
      - schema diff (missing/extra columns)
      - row-level numeric differences beyond tolerance
      - a summary DataFrame with max_abs_diff per column
    """
    warnings = []

    # 1. Schema check
    cols_ref = set(df_reference.columns)
    cols_cha = set(df_challenger.columns)
    missing_in_challenger = cols_ref - cols_cha
    extra_in_challenger = cols_cha - cols_ref

    if missing_in_challenger:
        warnings.append(
            f"Columns in {label_reference} but not in {label_challenger}: {missing_in_challenger}"
        )
    if extra_in_challenger:
        warnings.append(f"Extra columns in {label_challenger}: {extra_in_challenger}")

    common_cols = list(cols_ref & cols_cha)
    common_cols = [c for c in common_cols if c not in key_cols]

    # 2. Align on key columns
    merged = df_reference[key_cols + common_cols].merge(
        df_challenger[key_cols + common_cols], on=key_cols, suffixes=("_ref", "_cha")
    )

    # 3. Numeric diff per column
    numeric_cols = (
        df_reference[common_cols].select_dtypes(include="number").columns.tolist()
    )
    diff_records = {}
    diff_records["index"] = {
        "max_abs_diff": 0,
        "mean_abs_diff": 0,
        "rows_exceeding_tol": 0,
    }
    if numeric_cols:
        for col in numeric_cols:
            diff = (merged[f"{col}_ref"] - merged[f"{col}_cha"]).abs()
            diff_records[col] = {
                "max_abs_diff": diff.max(),
                "mean_abs_diff": diff.mean(),
                "rows_exceeding_tol": int((diff > numeric_tol).sum()),
            }

    summary_df = (
        pd.DataFrame(diff_records).T.reset_index().rename(columns={"index": "column"})
    )
    status = "PASS" if summary_df["rows_exceeding_tol"].max() == 0 else "FAIL"

    return ValidationResult(
        test_name="compare_dataframes",
        status=status,
        summary_df=summary_df,
        details={
            "merged_df": merged,
            "schema_warnings": list(missing_in_challenger | extra_in_challenger),
        },
        warnings=warnings,
    )
