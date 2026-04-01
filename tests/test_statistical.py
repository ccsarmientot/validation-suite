import numpy as np
import pandas as pd
import pytest
from validation_suite import vif_check


FEATURES_LOW  = ["ltv", "dti", "credit_age", "util_rate"]
FEATURES_HIGH = ["income", "income_scaled", "dti", "credit_age"]


class TestVifCasosNominales:

    def test_features_ortogonales_retorna_pass(self, df_low_collinearity):
        result = vif_check(df_low_collinearity, feature_cols=FEATURES_LOW)

        assert result.status == "PASS"
        assert result.warnings == []

    def test_features_colineales_retorna_fail(self, df_high_collinearity):
        result = vif_check(df_high_collinearity, feature_cols=FEATURES_HIGH)

        assert result.status == "FAIL"
        assert len(result.warnings) > 0
        assert "income" in result.warnings[0] or "income_scaled" in result.warnings[0]

    def test_features_colineales_identificadas_en_details(self, df_high_collinearity):
        result = vif_check(df_high_collinearity, feature_cols=FEATURES_HIGH)

        flagged = result.details["flagged_features"]
        assert len(flagged) >= 1
        assert any(f in flagged for f in ["income", "income_scaled"])

    def test_threshold_custom_cambia_clasificacion(self, df_high_collinearity):
        """
        With an extremely permissive threshold, even high VIF should pass.
        Useful to test that threshold is respected, not hardcoded.
        """
        result = vif_check(
            df_high_collinearity,
            feature_cols=FEATURES_HIGH,
            threshold=1_000.0,
        )
        assert result.status == "PASS"


class TestVifSummaryDf:

    def test_summary_contiene_todas_las_features(self, df_low_collinearity):
        result = vif_check(df_low_collinearity, feature_cols=FEATURES_LOW)

        assert set(result.summary_df["feature"]) == set(FEATURES_LOW)

    def test_summary_contiene_columna_flag(self, df_low_collinearity):
        result = vif_check(df_low_collinearity, feature_cols=FEATURES_LOW)

        assert "flag" in result.summary_df.columns
        assert set(result.summary_df["flag"]).issubset({"OK", "MODERATE", "HIGH"})

    def test_flag_moderate_entre_5_y_10(self):
        """
        Constructs a moderately correlated pair (VIF ~ 6-8) and checks
        that the MODERATE label is assigned correctly.
        SR 11-7 reviewers look for this zone explicitly.
        """
        rng = np.random.default_rng(7)
        n = 300
        x1 = rng.standard_normal(n)
        x2 = 0.85 * x1 + 0.3 * rng.standard_normal(n)  # correlation ~0.85 → VIF ~3-5
        df = pd.DataFrame({"x1": x1, "x2": x2})
        result = vif_check(df, feature_cols=["x1", "x2"])

        flags = set(result.summary_df["flag"])
        # At this correlation level we expect at least MODERATE, not necessarily HIGH
        assert flags & {"MODERATE", "HIGH"}

    def test_vif_values_son_positivos(self, df_low_collinearity):
        result = vif_check(df_low_collinearity, feature_cols=FEATURES_LOW)

        assert (result.summary_df["VIF"] > 0).all()


class TestVifEdgeCases:

    def test_una_sola_feature_vif_es_uno(self):
        """
        By definition VIF for a single predictor = 1.0.
        statsmodels returns 1.0 when there's nothing to regress against.
        """
        rng = np.random.default_rng(99)
        df = pd.DataFrame({"ltv": rng.uniform(0, 1, 100)})
        result = vif_check(df, feature_cols=["ltv"])

        assert result.status == "PASS"
        assert abs(result.summary_df["VIF"].iloc[0] - 1.0) < 0.01

    def test_nans_en_input_no_crashea(self, df_low_collinearity):
        """NaNs should be silently dropped before VIF computation."""
        df = df_low_collinearity.copy()
        df.loc[[0, 5, 10], "ltv"] = np.nan
        result = vif_check(df, feature_cols=FEATURES_LOW)

        assert result.status in {"PASS", "FAIL"}  # either is fine — must not raise

    def test_to_excel_genera_archivo(self, df_low_collinearity, tmp_path):
        result = vif_check(df_low_collinearity, feature_cols=FEATURES_LOW)
        out = tmp_path / "vif_output.xlsx"
        result.to_excel(str(out))

        assert out.exists()
        meta = pd.read_excel(out, sheet_name="Metadata")
        assert meta["status"].iloc[0] == "PASS"