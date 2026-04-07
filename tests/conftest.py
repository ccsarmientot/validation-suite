import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df_identical():
    """Two identical DataFrames — simulates a perfect dry run match."""
    data = {
        "obligor_id": [1001, 1002, 1003, 1004, 1005],
        "pd_score": [0.023, 0.118, 0.045, 0.302, 0.011],
        "lgd": [0.45, 0.60, 0.38, 0.72, 0.30],
        "ead": [150_000, 80_000, 220_000, 45_000, 310_000],
    }
    df = pd.DataFrame(data)
    return df.copy(), df.copy()


@pytest.fixture
def df_with_diff():
    """
    Reference vs challenger with a controlled numeric difference.
    obligor 1003 has a PD delta of 0.05 — above default tolerance (1e-6).
    """
    ref = pd.DataFrame(
        {
            "obligor_id": [1001, 1002, 1003, 1004, 1005],
            "pd_score": [0.023, 0.118, 0.045, 0.302, 0.011],
            "lgd": [0.45, 0.60, 0.38, 0.72, 0.30],
        }
    )
    cha = ref.copy()
    cha.loc[cha["obligor_id"] == 1003, "pd_score"] = 0.095  # delta = 0.05
    return ref, cha


@pytest.fixture
def df_schema_mismatch():
    """Challenger is missing the 'lgd' column — common in dry-run hand-offs."""
    ref = pd.DataFrame(
        {
            "obligor_id": [1001, 1002],
            "pd_score": [0.023, 0.118],
            "lgd": [0.45, 0.60],
        }
    )
    cha = ref.drop(columns=["lgd"])
    return ref, cha


@pytest.fixture
def df_low_collinearity():
    """Four near-orthogonal features — expected VIF close to 1.0."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "ltv": rng.uniform(0.3, 0.95, n),
            "dti": rng.uniform(0.1, 0.6, n),
            "credit_age": rng.uniform(1, 30, n),
            "util_rate": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def df_high_collinearity():
    """
    'income' and 'income_scaled' are nearly identical — VIF will be >> 10.
    Mimics the real pattern of including both raw and normalized versions
    of the same variable in a regression.
    """
    rng = np.random.default_rng(0)
    n = 200
    income = rng.lognormal(mean=10, sigma=1, size=n)
    return pd.DataFrame(
        {
            "income": income,
            "income_scaled": income / 1000 + rng.normal(0, 10, n),  # near-duplicate
            "dti": rng.uniform(0.1, 0.6, n),
            "credit_age": rng.uniform(1, 30, n),
        }
    )


# ── Fixtures stability ────────────────────────────────────────────────────────


@pytest.fixture
def dist_stable():
    """
    Expected and actual drawn from the same distribution.
    PSI should be near zero — STABLE.
    """
    rng = np.random.default_rng(42)
    n = 2000
    expected = pd.DataFrame({"pd_score": rng.beta(2, 18, n)})
    actual = pd.DataFrame({"pd_score": rng.beta(2, 18, n)})
    return expected, actual


@pytest.fixture
def dist_moderate_shift():
    """
    Actual distribution shifted moderately (beta params changed slightly).
    PSI should land in the MODERATE band (0.10 – 0.25).
    """
    rng = np.random.default_rng(0)
    n = 2000
    expected = pd.DataFrame({"pd_score": rng.beta(2, 18, n)})
    actual = pd.DataFrame({"pd_score": rng.beta(3, 14, n)})  # shifted right
    return expected, actual


@pytest.fixture
def dist_unstable_shift():
    """
    Actual distribution severely shifted — simulates a portfolio
    quality deterioration between development and monitoring periods.
    PSI should be > 0.25 — UNSTABLE.
    """
    rng = np.random.default_rng(1)
    n = 2000
    expected = pd.DataFrame({"pd_score": rng.beta(2, 18, n)})
    actual = pd.DataFrame({"pd_score": rng.beta(8, 4, n)})  # heavy shift
    return expected, actual


@pytest.fixture
def dist_multivar():
    """
    Multi-variable DataFrame: one stable variable, one unstable.
    Used to validate per-variable summary and mixed statuses.
    """
    rng = np.random.default_rng(7)
    n = 2000
    expected = pd.DataFrame(
        {
            "pd_score": rng.beta(2, 18, n),
            "ltv": rng.uniform(0.3, 0.9, n),  # stable
        }
    )
    actual = pd.DataFrame(
        {
            "pd_score": rng.beta(8, 4, n),  # unstable
            "ltv": rng.uniform(0.3, 0.9, n),  # stable
        }
    )
    return expected, actual


@pytest.fixture
def dist_with_nans():
    """Expected and actual with ~5% NaNs — should not crash."""
    rng = np.random.default_rng(99)
    n = 1000
    scores_exp = rng.beta(2, 18, n).astype(float)
    scores_act = rng.beta(2, 18, n).astype(float)
    scores_exp[rng.choice(n, size=50, replace=False)] = np.nan
    scores_act[rng.choice(n, size=50, replace=False)] = np.nan
    return (
        pd.DataFrame({"pd_score": scores_exp}),
        pd.DataFrame({"pd_score": scores_act}),
    )


@pytest.fixture
def csi_frames():
    """
    Segmented DataFrames for CSI tests.

    Segments:
      - 'Low'    : stable across expected / actual
      - 'Medium' : moderate shift
      - 'High'   : severe shift (simulates grade migration)
    """
    rng = np.random.default_rng(3)
    n_per_seg = 600

    def _make(segment, a, b):
        return pd.DataFrame(
            {
                "pd_score": rng.beta(a, b, n_per_seg),
                "risk_grade": segment,
            }
        )

    expected = pd.concat(
        [
            _make("Low", a=1, b=30),
            _make("Medium", a=3, b=15),
            _make("High", a=8, b=5),
        ],
        ignore_index=True,
    )

    actual = pd.concat(
        [
            _make("Low", a=1, b=30),  # stable
            _make("Medium", a=4, b=12),  # moderate shift
            _make("High", a=15, b=3),  # severe shift
        ],
        ignore_index=True,
    )

    return expected, actual


# ── Fixtures performance ──────────────────────────────────────────────────────
@pytest.fixture
def good_model():
    """
    Well-discriminating, well-calibrated model.
    AUC ~ 0.80, default rate predicted ≈ observed.
    Represents a model that should PASS all checks.
    """
    rng = np.random.default_rng(42)
    n = 2000
    true_pd = rng.beta(2, 18, n)
    y_true = rng.binomial(1, true_pd).astype(float)
    # Score correlated with true_pd + small noise → good discrimination
    y_score = np.clip(true_pd + rng.normal(0, 0.01, n), 0, 1)
    return y_true, y_score


@pytest.fixture
def poor_discrimination():
    """
    Model with near-random scores — AUC close to 0.50.
    Discrimination metrics should fail.
    """
    rng = np.random.default_rng(1)
    n = 2000
    true_pd = rng.beta(2, 18, n)
    y_true = rng.binomial(1, true_pd).astype(float)
    # Scores uncorrelated with outcomes
    y_score = rng.uniform(0, 0.20, n)
    return y_true, y_score


@pytest.fixture
def overestimating_model():
    """
    Model that systematically over-predicts default probability.
    Predicted DR >> observed DR → binomial test should reject.
    """
    rng = np.random.default_rng(7)
    n = 3000
    true_pd = rng.beta(2, 18, n)
    y_true = rng.binomial(1, true_pd).astype(float)
    # Inflate scores by 3x
    y_score = np.clip(true_pd * 3, 0, 1)
    return y_true, y_score


@pytest.fixture
def underestimating_model():
    """
    Model that systematically under-predicts default probability.
    Predicted DR << observed DR → binomial test should reject.
    """
    rng = np.random.default_rng(9)
    n = 3000
    true_pd = rng.beta(4, 8, n)  # higher true default rate
    y_true = rng.binomial(1, true_pd).astype(float)
    # Deflate scores significantly
    y_score = np.clip(true_pd * 0.3, 0, 1)
    return y_true, y_score


@pytest.fixture
def perfect_model():
    """
    Scores equal to true labels — AUC = 1.0, Gini = 1.0, KS = 1.0.
    Used to verify metric upper bounds.
    """
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.binomial(1, 0.10, n).astype(float)
    y_score = y_true.copy()
    return y_true, y_score


@pytest.fixture
def small_sample():
    """50 observations — minimum viable input without crashing."""
    rng = np.random.default_rng(55)
    n = 50
    true_pd = rng.beta(2, 18, n)
    y_true = rng.binomial(1, true_pd).astype(float)
    # Ensure at least one default and one non-default
    y_true[0] = 1.0
    y_true[1] = 0.0
    y_score = np.clip(true_pd + rng.normal(0, 0.02, n), 0, 1)
    return y_true, y_score
