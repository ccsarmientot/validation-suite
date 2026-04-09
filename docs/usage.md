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

Example
-------
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

## 2. vif_check():

Computes VIF for each feature. SR11-7 standard threshold = 10. Values > 10 indicate problematic multicollinearity.

Example
-------

```python
from validation_suite import vif_check

rng2 = np.random.default_rng(7)
n_obs = 1000

df_features_clean = pd.DataFrame(
    {
        "ltv": rng2.uniform(0.30, 0.95, n_obs),
        "dti": rng2.uniform(0.10, 0.60, n_obs),
        "credit_age_yrs": rng2.uniform(1, 30, n_obs),
        "utilization_rate": rng2.uniform(0, 1, n_obs),
        "num_delinquencies": rng2.poisson(lam=0.4, size=n_obs).astype(float),
    }
)

result_vif_clean = vif_check(
    df_features_clean, feature_cols=FEATURE_COLS, threshold=10.0
)
```
Messing with features and correlating it will produce as output something like:

| # | Feature             | VIF          | Flag     |
|---:|---------------------|--------------|----------|
| 1 | dti                 | 20543658.327543 | HIGH     |
| 5 | dti_annualized      | 20543522.726991 | HIGH     |
| 0 | ltv                 | 6.611564     | MODERATE |
| 2 | credit_age_yrs      | 3.856893     | OK       |
| 3 | utilization_rate    | 3.500687     | OK       |
| 4 | num_delinquencies   | 1.398969     | OK       |


## 3. psi_check()

Population Stability Index (PSI) — detects distributional shift between a reference (development/prior) sample and a monitoring (current) sample.

SR 11-7 context: Monitors whether the *input* population the model is applied to has drifted from the population it was developed on. Elevated PSI (> 0.25) is a trigger for model recalibration review.

Parameters
----------
* expected : DataFrame or Series:
    Reference distribution — typically the development sample or the last approved monitoring period.
* actual : DataFrame or Series:
    Current distribution to compare against the reference.
* columns : list[str], optional:
    Columns to evaluate.  Defaults to all numeric columns when a DataFrame is passed.
* bins : int:
    Number of quantile bins. 10 (deciles) is the SR 11-7 convention.
* epsilon : float:
    Small constant to avoid log(0) on empty bins.

Returns
-------
ValidationResult
* status       : 'PASS' if all variables are STABLE, else 'FAIL'
* summary_df   : one row per variable — psi_value, flag, n_bins_shifted
* details      : {'bin_detail': {col: DataFrame}, 'thresholds': {...}}
* warnings     : list of variables with MODERATE or UNSTABLE PSI

Example
-------

```python
from validation_suite import psi_check

PSI_COLUMNS = ["pd_score", "ltv", "dti"]

result_psi = psi_check(
    expected=df_dev,
    actual=df_monitor,
    columns=PSI_COLUMNS,
    bins=10,  # deciles
)

print(f"Status general: {result_psi.status}")
print()
if result_psi.warnings:
    print("Warnings:")
    for w in result_psi.warnings:
        print(f"  ⚠  {w}")
    print()

result_psi.summary_df.sort_values("psi_value", ascending=False)
```

results in:

Status general: FAIL

Warnings
* ⚠ **PSI UNSTABLE** for **`pd_score`**: **7.3992**  
*(Thresholds: MODERATE = 0.25, UNSTABLE > 0.25)*

PSI Summary Table

| # | Variable | PSI Value | Flag     | # Bins Shifted | N Expected | N Actual |
|---:|----------|----------:|----------|----------------:|------------:|-----------:|
| 0 | pd_score | 7.399166  | UNSTABLE | 10              | 5000        | 5000       |
| 2 | dti      | 0.032043  | STABLE   | 1               | 5000        | 5000       |
| 1 | ltv      | 0.002046  | STABLE   | 0               | 5000        | 5000       |


## 3. csi_check()

Characteristic Stability Index (CSI) — measures distributional shift of the model score *within each segment* of a categorical variable.

SR 11-7 context: Where PSI measures overall population shift, CSI pinpoints *which segment* is driving instability.  Commonly applied to risk grades, product lines, or geographic segments in PD/LGD/EAD models.

Parameters
----------
* expected : DataFrame - 
    Reference sample containing both `score_col` and `segment_col`.
* actual : DataFrame - 
    Monitoring sample with the same columns.
* score_col : str - 
    Name of the model score (continuous, e.g. PD score or log-odds).
* segment_col : str - 
    Categorical column defining the segments (e.g. risk_grade, region).
* bins : int - 
    Bins for score distribution within each segment.
* epsilon : float - 
    Small constant to avoid log(0).

Returns
-------
ValidationResult
* status       : 'PASS' if all segments are STABLE, else 'FAIL'
* summary_df   : one row per segment — csi_value, flag, segment_pct_*
* details      : {'bin_detail': {segment: DataFrame}, 'score_col': ..., 'segment_col': ..., 'thresholds': {...}}
* warnings     : list of unstable segments

Example
-------

```python
from validation_suite import csi_check

result_csi = csi_check(
    expected=df_dev_seg,
    actual=df_monitor_seg,
    score_col=SCORE_COL,
    segment_col=SEGMENT_COL,
    bins=10,
)

print(f"Status general: {result_csi.status}")
print()
if result_csi.warnings:
    print("Warnings:")
    for w in result_csi.warnings:
        print(f"  ⚠  {w}")
    print()

result_csi.summary_df.sort_values("csi_value", ascending=False)
```

results in:

Status general: FAIL

Warnings
* ⚠ **CSI UNSTABLE** in segment **`BBB`** (`pd_score`): **0.8413**  
* ⚠ **CSI UNSTABLE** in segment **`CCC`** (`pd_score`): **5.1397**

CSI Summary Table

| # | Segment | CSI Value | Flag     | N Expected | N Actual | Segment % Expected | Segment % Actual |
|---:|---------|----------:|----------|------------:|-----------:|-------------------:|------------------:|
| 2 | CCC     | 5.139687  | UNSTABLE | 800         | 800        | 0.333333           | 0.333333          |
| 1 | BBB     | 0.841326  | UNSTABLE | 800         | 800        | 0.333333           | 0.333333          |
| 0 | AAA     | 0.015184  | STABLE   | 800         | 800        | 0.333333           | 0.333333          |

## 4. backtesting_report()

Comprehensive backtesting report for binary classification models (PD, fraud, prepayment) aligned with SR 11-7 performance requirements.

Metrics computed
----------------
Discrimination:
- AUC-ROC
- Gini coefficient  (= 2*AUC - 1)
- KS statistic      (max TPR - FPR separation)

Calibration:
- Brier score
- Hosmer-Lemeshow goodness-of-fit test (chi-squared)
- Portfolio-level binomial test (predicted DR vs observed DR)
- Traffic-light classification per Basel / SR 11-7 convention
- Per-decile calibration table

Parameters
----------
* y_true : array-like of int (0/1) - 
Observed binary outcomes (1 = default/event, 0 = no event).
* y_score : array-like of float in [0, 1] - 
Predicted probabilities from the model.
* model_name : str - 
Label for the model under review (used in metadata).
* n_hl_groups : int - 
Number of groups for Hosmer-Lemeshow test (default 10 = deciles).
* n_cal_buckets : int - 
Number of buckets for the calibration table (default 10).
* alpha : float - 
Significance level for pass/fail determination (default 0.05).

Returns
-------
ValidationResult
* status      : 'PASS' if discrimination and calibration are adequate
* summary_df  : one row — all scalar metrics side by side
* details     : {
                'roc_curve'          : DataFrame(fpr, tpr, threshold),
                'hosmer_lemeshow'    : {statistic, p_value, df, groups_df},
                'binomial_test'      : {predicted_dr, observed_dr, z_stat, p_value},
                'calibration_table'  : DataFrame (per-decile),
                'traffic_light'      : str,
                'alpha'              : float,
                }
* warnings    : list of metric-level findings

Example
-------

```python
from validation_suite import backtesting_report()

result_bt_good = backtesting_report(
    y_true=y_true,
    y_score=y_score_good,
    model_name="PD-RETAIL-V2.3",
    n_hl_groups=10,
    n_cal_buckets=10,
    alpha=0.05,
)

print(f"Status        : {result_bt_good.status}")
print(f"Traffic light : {result_bt_good.details['traffic_light']}")
print()
if result_bt_good.warnings:
    for w in result_bt_good.warnings:
        print(f"  ⚠  {w}")
else:
    print("  Sin warnings.")
print()

cols = [
    "model_name",
    "n_observations",
    "n_defaults",
    "auc_roc",
    "gini",
    "ks_statistic",
    "brier_score",
    "hl_p_value",
    "binomial_p_value",
    "traffic_light",
    "status",
]
result_bt_good.summary_df[cols]
```

results in:

**Status** : PASS

**Traffic light**: GREEN

* ⚠ **AUC-ROC = 0.6935** — moderate discrimination, monitor closely.

| model_name       | n_observations | n_defaults | auc_roc | gini     | ks_statistic | brier_score | hl_p_value | binomial_p_value | traffic_light | status |
|------------------|---------------:|-----------:|--------:|---------:|-------------:|------------:|-----------:|-----------------:|---------------|--------|
| PD-RETAIL-V2.3   | 3000           | 280        | 0.693480| 0.386959 | 0.306828     | 0.081629    | 0.587496   | 0.317816         | GREEN         | PASS   |