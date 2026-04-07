# validation_suite/performance.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from .results import ValidationResult

# ── Helpers ─────────────────────────────────────────────────────────────────


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of defaulters and non-defaulters.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _hosmer_lemeshow(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_groups: int = 10,
) -> dict:
    """
    Hosmer-Lemeshow goodness-of-fit test for PD calibration.

    Splits observations into `n_groups` deciles by predicted PD,
    then computes chi-squared statistic and p-value.

    Returns dict with keys: statistic, p_value, df, groups_df.
    """
    from scipy.stats import chi2

    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["decile"] = pd.qcut(df["p"], q=n_groups, duplicates="drop", labels=False)

    groups = (
        df.groupby("decile")
        .agg(
            n=("y", "count"),
            observed_defaults=("y", "sum"),
            mean_predicted_pd=("p", "mean"),
        )
        .reset_index()
    )
    groups["expected_defaults"] = groups["n"] * groups["mean_predicted_pd"]
    groups["expected_non_defaults"] = groups["n"] - groups["expected_defaults"]
    groups["observed_non_defaults"] = groups["n"] - groups["observed_defaults"]

    # Chi-squared components
    groups["chi2_defaults"] = np.where(
        groups["expected_defaults"] > 0,
        (groups["observed_defaults"] - groups["expected_defaults"]) ** 2
        / groups["expected_defaults"],
        0,
    )
    groups["chi2_non_defaults"] = np.where(
        groups["expected_non_defaults"] > 0,
        (groups["observed_non_defaults"] - groups["expected_non_defaults"]) ** 2
        / groups["expected_non_defaults"],
        0,
    )

    hl_stat = float((groups["chi2_defaults"] + groups["chi2_non_defaults"]).sum())
    df_stat = len(groups) - 2
    p_value = float(1 - chi2.cdf(hl_stat, df=df_stat))

    return {
        "statistic": hl_stat,
        "p_value": p_value,
        "df": df_stat,
        "groups_df": groups,
    }


def _binomial_test_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Portfolio-level binomial test: are predicted defaults consistent
    with observed defaults?

    H0: E[predicted PD] == observed default rate
    Uses normal approximation (valid for large n).

    Returns dict with keys: predicted_dr, observed_dr, z_stat, p_value.
    """
    from scipy.stats import norm

    n = len(y_true)
    predicted_dr = float(y_score.mean())
    observed_dr = float(y_true.mean())

    # Standard error under H0
    se = np.sqrt(predicted_dr * (1 - predicted_dr) / n)
    z_stat = (observed_dr - predicted_dr) / se if se > 0 else 0.0
    p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

    return {
        "predicted_dr": predicted_dr,
        "observed_dr": observed_dr,
        "z_stat": z_stat,
        "p_value": p_value,
    }


def _traffic_light(p_value: float) -> str:
    """
    Basel/SR 11-7 traffic light classification based on p-value.

    Green  : p >= 0.10  — model within expected range
    Yellow : 0.05 <= p < 0.10  — caution, monitor closely
    Red    : p < 0.05  — significant miscalibration
    """
    if p_value >= 0.10:
        return "GREEN"
    if p_value >= 0.05:
        return "YELLOW"
    return "RED"


def _pd_calibration_by_bucket(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Compares mean predicted PD vs observed default rate per score decile.
    Used for calibration plots and bucket-level analysis.
    """
    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["bucket"] = pd.qcut(df["p"], q=n_buckets, duplicates="drop", labels=False)

    cal = (
        df.groupby("bucket")
        .agg(
            n=("y", "count"),
            observed_dr=("y", "mean"),
            mean_predicted_pd=("p", "mean"),
            min_score=("p", "min"),
            max_score=("p", "max"),
        )
        .reset_index()
    )
    cal["ratio_pred_obs"] = np.where(
        cal["observed_dr"] > 0,
        cal["mean_predicted_pd"] / cal["observed_dr"],
        np.nan,
    )
    return cal


# ── Public API ──────────────────────────────────────────────────────────────


def backtesting_report(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    model_name: str = "Model",
    n_hl_groups: int = 10,
    n_cal_buckets: int = 10,
    alpha: float = 0.05,
) -> ValidationResult:
    """
    Comprehensive backtesting report for binary classification models
    (PD, fraud, prepayment) aligned with SR 11-7 performance requirements.

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
    y_true : array-like of int (0/1)
        Observed binary outcomes (1 = default/event, 0 = no event).
    y_score : array-like of float in [0, 1]
        Predicted probabilities from the model.
    model_name : str
        Label for the model under review (used in metadata).
    n_hl_groups : int
        Number of groups for Hosmer-Lemeshow test (default 10 = deciles).
    n_cal_buckets : int
        Number of buckets for the calibration table (default 10).
    alpha : float
        Significance level for pass/fail determination (default 0.05).

    Returns
    -------
    ValidationResult
        status      : 'PASS' if discrimination and calibration are adequate
        summary_df  : one row — all scalar metrics side by side
        details     : {
                        'roc_curve'          : DataFrame(fpr, tpr, threshold),
                        'hosmer_lemeshow'    : {statistic, p_value, df, groups_df},
                        'binomial_test'      : {predicted_dr, observed_dr, z_stat, p_value},
                        'calibration_table'  : DataFrame (per-decile),
                        'traffic_light'      : str,
                        'alpha'              : float,
                      }
        warnings    : list of metric-level findings
    """
    # ── Input validation ──────────────────────────────────────────────
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score must have the same length "
            f"(got {len(y_true)} vs {len(y_score)})."
        )
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1 values.")
    if np.any((y_score < 0) | (y_score > 1)):
        raise ValueError("y_score values must be in [0, 1].")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("y_true must contain both classes (0 and 1).")

    warnings_list = []

    # ── Discrimination metrics ────────────────────────────────────────
    auc = float(roc_auc_score(y_true, y_score))
    gini = _gini(y_true, y_score)
    ks = _ks_statistic(y_true, y_score)
    brier = float(brier_score_loss(y_true, y_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    # Discrimination warnings (SR 11-7 general thresholds)
    if auc < 0.60:
        warnings_list.append(f"AUC-ROC = {auc:.4f} — below 0.60, poor discrimination.")
    elif auc < 0.70:
        warnings_list.append(
            f"AUC-ROC = {auc:.4f} — moderate discrimination, monitor closely."
        )

    if gini < 0.20:
        warnings_list.append(f"Gini = {gini:.4f} — weak rank-ordering ability.")

    if ks < 0.20:
        warnings_list.append(
            f"KS = {ks:.4f} — low separation between defaulters and non-defaulters."
        )

    # ── Calibration metrics ───────────────────────────────────────────
    hl = _hosmer_lemeshow(y_true, y_score, n_groups=n_hl_groups)
    binom = _binomial_test_calibration(y_true, y_score)
    cal_tbl = _pd_calibration_by_bucket(y_true, y_score, n_buckets=n_cal_buckets)
    tl = _traffic_light(binom["p_value"])

    # Calibration warnings
    if hl["p_value"] < alpha:
        warnings_list.append(
            f"Hosmer-Lemeshow p={hl['p_value']:.4f} < {alpha} — "
            f"significant lack of fit (chi2={hl['statistic']:.2f}, df={hl['df']})."
        )
    if binom["p_value"] < alpha:
        direction = "over" if binom["predicted_dr"] > binom["observed_dr"] else "under"
        warnings_list.append(
            f"Binomial test p={binom['p_value']:.4f} < {alpha} — "
            f"model is {direction}-predicting defaults "
            f"(predicted DR={binom['predicted_dr']:.4%}, "
            f"observed DR={binom['observed_dr']:.4%})."
        )
    if tl == "RED":
        warnings_list.append("Traffic light: RED — significant calibration failure.")
    elif tl == "YELLOW":
        warnings_list.append("Traffic light: YELLOW — marginal calibration, monitor.")

    # ── Status ────────────────────────────────────────────────────────
    discrimination_ok = auc >= 0.60 and gini >= 0.20
    calibration_ok = hl["p_value"] >= alpha and binom["p_value"] >= alpha
    status = "PASS" if (discrimination_ok and calibration_ok) else "FAIL"

    # ── Summary DataFrame (one row, all scalar metrics) ───────────────
    summary_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n_observations": len(y_true),
                "n_defaults": int(y_true.sum()),
                "default_rate_obs": binom["observed_dr"],
                "default_rate_pred": binom["predicted_dr"],
                # Discrimination
                "auc_roc": auc,
                "gini": gini,
                "ks_statistic": ks,
                # Calibration
                "brier_score": brier,
                "hl_statistic": hl["statistic"],
                "hl_p_value": hl["p_value"],
                "binomial_z": binom["z_stat"],
                "binomial_p_value": binom["p_value"],
                "traffic_light": tl,
                "status": status,
            }
        ]
    )

    return ValidationResult(
        test_name="backtesting_report",
        status=status,
        summary_df=summary_df,
        details={
            "roc_curve": roc_df,
            "hosmer_lemeshow": hl,
            "binomial_test": binom,
            "calibration_table": cal_tbl,
            "traffic_light": tl,
            "alpha": alpha,
        },
        warnings=warnings_list,
    )
