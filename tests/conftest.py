import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def df_identical():
    """Two identical DataFrames — simulates a perfect dry run match."""
    data = {
        "obligor_id": [1001, 1002, 1003, 1004, 1005],
        "pd_score":   [0.023, 0.118, 0.045, 0.302, 0.011],
        "lgd":        [0.45,  0.60,  0.38,  0.72,  0.30],
        "ead":        [150_000, 80_000, 220_000, 45_000, 310_000],
    }
    df = pd.DataFrame(data)
    return df.copy(), df.copy()


@pytest.fixture
def df_with_diff():
    """
    Reference vs challenger with a controlled numeric difference.
    obligor 1003 has a PD delta of 0.05 — above default tolerance (1e-6).
    """
    ref = pd.DataFrame({
        "obligor_id": [1001, 1002, 1003, 1004, 1005],
        "pd_score":   [0.023, 0.118, 0.045, 0.302, 0.011],
        "lgd":        [0.45,  0.60,  0.38,  0.72,  0.30],
    })
    cha = ref.copy()
    cha.loc[cha["obligor_id"] == 1003, "pd_score"] = 0.095  # delta = 0.05
    return ref, cha


@pytest.fixture
def df_schema_mismatch():
    """Challenger is missing the 'lgd' column — common in dry-run hand-offs."""
    ref = pd.DataFrame({
        "obligor_id": [1001, 1002],
        "pd_score":   [0.023, 0.118],
        "lgd":        [0.45,  0.60],
    })
    cha = ref.drop(columns=["lgd"])
    return ref, cha


@pytest.fixture
def df_low_collinearity():
    """Four near-orthogonal features — expected VIF close to 1.0."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "ltv":        rng.uniform(0.3, 0.95, n),
        "dti":        rng.uniform(0.1, 0.6, n),
        "credit_age": rng.uniform(1, 30, n),
        "util_rate":  rng.uniform(0, 1, n),
    })


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
    return pd.DataFrame({
        "income":        income,
        "income_scaled": income / 1000 + rng.normal(0, 0.001, n),  # near-duplicate
        "dti":           rng.uniform(0.1, 0.6, n),
        "credit_age":    rng.uniform(1, 30, n),
    })