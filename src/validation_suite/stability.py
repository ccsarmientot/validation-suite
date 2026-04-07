# sr117_validator/stability.py

import numpy as np
import pandas as pd

from .results import ValidationResult

# ── Umbrales SR 11-7 / industria estándar ──────────────────────────────────
#
#  PSI / CSI    Clasificación   Acción recomendada
#  ---------    -------------   -------------------------------------------
#  < 0.10       STABLE          Sin acción
#  0.10 – 0.25  MODERATE        Monitorear, investigar causa
#  > 0.25       UNSTABLE        Requiere remediación o recalibración
#
# ---------------------------------------------------------------------------

_PSI_STABLE = 0.10
_PSI_MODERATE = 0.25


def _compute_psi_series(
    expected: pd.Series,
    actual: pd.Series,
    bins: int | list,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Core PSI computation for a single numeric or categorical series.

    Returns a DataFrame with one row per bin containing:
      bin_label, expected_n, actual_n, expected_pct, actual_pct,
      psi_contribution
    """
    if isinstance(bins, int):
        # Build bin edges from the *expected* distribution (development sample)
        _, bin_edges = np.histogram(expected.dropna(), bins=bins)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        labels = [
            f"({bin_edges[i]:.4g}, {bin_edges[i + 1]:.4g}]"
            for i in range(len(bin_edges) - 1)
        ]
        exp_counts, _ = np.histogram(expected.dropna(), bins=bin_edges)
        act_counts, _ = np.histogram(actual.dropna(), bins=bin_edges)
    else:
        # Categorical: bins is the list of unique categories
        labels = bins
        exp_counts = np.array([(expected == cat).sum() for cat in labels])
        act_counts = np.array([(actual == cat).sum() for cat in labels])

    n_exp = exp_counts.sum()
    n_act = act_counts.sum()

    exp_pct = exp_counts / n_exp
    act_pct = act_counts / n_act

    # Replace zeros to avoid log(0)
    exp_pct_safe = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct_safe = np.where(act_pct == 0, epsilon, act_pct)

    psi_contributions = (act_pct_safe - exp_pct_safe) * np.log(
        act_pct_safe / exp_pct_safe
    )

    return pd.DataFrame(
        {
            "bin": labels,
            "expected_n": exp_counts,
            "actual_n": act_counts,
            "expected_pct": exp_pct,
            "actual_pct": act_pct,
            "psi_contribution": psi_contributions,
        }
    )


def _flag_psi(value: float) -> str:
    if value < _PSI_STABLE:
        return "STABLE"
    if value < _PSI_MODERATE:
        return "MODERATE"
    return "UNSTABLE"


# ── Public API ──────────────────────────────────────────────────────────────


def psi_check(
    expected: pd.DataFrame | pd.Series,
    actual: pd.DataFrame | pd.Series,
    columns: list[str] | None = None,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> ValidationResult:
    """
    Population Stability Index (PSI) — detects distributional shift between
    a reference (development/prior) sample and a monitoring (current) sample.

    SR 11-7 context:
        Monitors whether the *input* population the model is applied to has
        drifted from the population it was developed on.  Elevated PSI (> 0.25)
        is a trigger for model recalibration review.

    Parameters
    ----------
    expected : DataFrame or Series
        Reference distribution — typically the development sample or
        the last approved monitoring period.
    actual : DataFrame or Series
        Current distribution to compare against the reference.
    columns : list[str], optional
        Columns to evaluate.  Defaults to all numeric columns when a
        DataFrame is passed.
    bins : int
        Number of quantile bins.  10 (deciles) is the SR 11-7 convention.
    epsilon : float
        Small constant to avoid log(0) on empty bins.

    Returns
    -------
    ValidationResult
        status       : 'PASS' if all variables are STABLE, else 'FAIL'
        summary_df   : one row per variable — psi_value, flag, n_bins_shifted
        details      : {'bin_detail': {col: DataFrame}, 'thresholds': {...}}
        warnings     : list of variables with MODERATE or UNSTABLE PSI
    """
    # ── Normalise inputs to DataFrames ────────────────────────────────
    if isinstance(expected, pd.Series):
        expected = expected.to_frame(name=expected.name or "variable")
    if isinstance(actual, pd.Series):
        actual = actual.to_frame(name=actual.name or "variable")

    if columns is None:
        columns = expected.select_dtypes(include="number").columns.tolist()

    if not columns:
        raise ValueError(
            "No numeric columns found. Pass `columns=` explicitly or "
            "ensure the DataFrame contains numeric data."
        )

    # ── Per-variable PSI ──────────────────────────────────────────────
    summary_rows = []
    bin_details = {}
    warnings = []

    for col in columns:
        if col not in expected.columns or col not in actual.columns:
            warnings.append(f"Column '{col}' not found in both DataFrames — skipped.")
            continue

        detail_df = _compute_psi_series(
            expected=expected[col],
            actual=actual[col],
            bins=bins,
            epsilon=epsilon,
        )

        psi_value = detail_df["psi_contribution"].sum()
        flag = _flag_psi(psi_value)
        n_bins_shifted = int((detail_df["psi_contribution"] > 0.01).sum())

        bin_details[col] = detail_df
        summary_rows.append(
            {
                "variable": col,
                "psi_value": psi_value,
                "flag": flag,
                "n_bins_shifted": n_bins_shifted,
                "n_expected": int(expected[col].notna().sum()),
                "n_actual": int(actual[col].notna().sum()),
            }
        )

        if flag in ("MODERATE", "UNSTABLE"):
            warnings.append(
                f"PSI {flag} for '{col}': {psi_value:.4f} "
                f"(threshold MODERATE={_PSI_MODERATE}, UNSTABLE>{_PSI_MODERATE})"
            )

    summary_df = pd.DataFrame(summary_rows)
    status = "PASS" if summary_df["flag"].isin(["STABLE"]).all() else "FAIL"

    return ValidationResult(
        test_name="psi_check",
        status=status,
        summary_df=summary_df,
        details={
            "bin_detail": bin_details,
            "thresholds": {
                "stable": _PSI_STABLE,
                "moderate": _PSI_MODERATE,
            },
            "bins": bins,
        },
        warnings=warnings,
    )


def csi_check(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    score_col: str,
    segment_col: str,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> ValidationResult:
    """
    Characteristic Stability Index (CSI) — measures distributional shift
    of the model score *within each segment* of a categorical variable.

    SR 11-7 context:
        Where PSI measures overall population shift, CSI pinpoints *which
        segment* is driving instability.  Commonly applied to risk grades,
        product lines, or geographic segments in PD/LGD/EAD models.

    Parameters
    ----------
    expected : DataFrame
        Reference sample containing both `score_col` and `segment_col`.
    actual : DataFrame
        Monitoring sample with the same columns.
    score_col : str
        Name of the model score (continuous, e.g. PD score or log-odds).
    segment_col : str
        Categorical column defining the segments (e.g. risk_grade, region).
    bins : int
        Bins for score distribution within each segment.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    ValidationResult
        status       : 'PASS' if all segments are STABLE, else 'FAIL'
        summary_df   : one row per segment — csi_value, flag, segment_pct_*
        details      : {'bin_detail': {segment: DataFrame}, 'score_col': ...,
                        'segment_col': ..., 'thresholds': {...}}
        warnings     : list of unstable segments
    """
    segments = sorted(
        set(expected[segment_col].dropna()) | set(actual[segment_col].dropna())
    )

    summary_rows = []
    bin_details = {}
    warnings = []

    n_exp_total = len(expected)
    n_act_total = len(actual)

    for seg in segments:
        exp_seg = expected.loc[expected[segment_col] == seg, score_col]
        act_seg = actual.loc[actual[segment_col] == seg, score_col]

        # Skip segments with no observations in either sample
        if exp_seg.empty or act_seg.empty:
            warnings.append(
                f"Segment '{seg}' missing in one sample "
                f"(expected n={len(exp_seg)}, actual n={len(act_seg)}) — skipped."
            )
            continue

        detail_df = _compute_psi_series(
            expected=exp_seg,
            actual=act_seg,
            bins=bins,
            epsilon=epsilon,
        )

        csi_value = detail_df["psi_contribution"].sum()
        flag = _flag_psi(csi_value)

        bin_details[str(seg)] = detail_df
        summary_rows.append(
            {
                "segment": seg,
                "csi_value": csi_value,
                "flag": flag,
                "n_expected": len(exp_seg),
                "n_actual": len(act_seg),
                "segment_pct_exp": len(exp_seg) / n_exp_total,
                "segment_pct_act": len(act_seg) / n_act_total,
            }
        )

        if flag in ("MODERATE", "UNSTABLE"):
            warnings.append(
                f"CSI {flag} in segment '{seg}' ({score_col}): {csi_value:.4f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    status = "PASS" if summary_df["flag"].isin(["STABLE"]).all() else "FAIL"

    return ValidationResult(
        test_name="csi_check",
        status=status,
        summary_df=summary_df,
        details={
            "bin_detail": bin_details,
            "score_col": score_col,
            "segment_col": segment_col,
            "thresholds": {
                "stable": _PSI_STABLE,
                "moderate": _PSI_MODERATE,
            },
            "bins": bins,
        },
        warnings=warnings,
    )
