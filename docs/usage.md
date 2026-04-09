# Usage

To use MRMU Validation Suite in a project. Import functions from:

```python
from validation_suite import (
    compare_dataframes,
    vif_check,
    psi_check,
    csi_check,
    backtesting_report,
    ValidationResult,
)
```

## 1. compare_dataframes():

Compares two DataFrames (e.g. model owner output vs validator dry run).

Returns a ValidationResult with:
* schema diff (missing/extra columns)
* row-level numeric differences beyond tolerance
* a summary DataFrame with max_abs_diff per column

For example:
```python
from validation_suite import compare_dataframes

result_strict = compare_dataframes(
    df_reference=df_model_owner,
    df_challenger=df_dry_run,
    key_cols=["obligor_id"],
    numeric_tol=1e-6,
    label_reference="Model Owner v2.3",
    label_challenger="Validator Dry Run",
)

print(f"Status   : {result_strict.status}")
print(f"Warnings : {result_strict.warnings if result_strict.warnings else 'None'}")
print()
result_strict.summary_df.sort_values("max_abs_diff", ascending=False)
```

results in:

| column   | max_abs_diff | mean_abs_diff | rows_exceeding_tol |
|----------|--------------|---------------|--------------------|
| pd_score | 0.007647     | 0.000059      | 8.000000           |
| index    | 0.000000     | 0.000000      | 0.000000           |
| lgd      | 0.000000     | 0.000000      | 0.000000           |
| ead      | 0.000000     | 0.000000      | 0.000000           |

Columns with rows_exceeding_tol > 0 require additional analysis. A small numerical difference (e.g., max_abs_diff < 0.005) may be acceptable if it can be explained by rounding differences between platforms


