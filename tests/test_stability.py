# tests/test_stability.py

import numpy as np
import pandas as pd
import pytest
from validation_suite import psi_check, csi_check


# ── PSI: casos nominales ────────────────────────────────────────────────────


class TestPsiNominal:

    def test_identical_distribution_is_stable(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"])

        assert result.status == "PASS"
        row = result.summary_df.iloc[0]
        assert row["flag"] == "STABLE"
        assert row["psi_value"] < 0.10

    def test_moderate_shift_flagged_correctly(self, dist_moderate_shift):
        exp, act = dist_moderate_shift
        result = psi_check(exp, act, columns=["pd_score"])

        row = result.summary_df.iloc[0]
        assert row["flag"] in ("MODERATE", "UNSTABLE")
        assert result.status == "FAIL"

    def test_severe_shift_is_unstable(self, dist_unstable_shift):
        exp, act = dist_unstable_shift
        result = psi_check(exp, act, columns=["pd_score"])

        row = result.summary_df.iloc[0]
        assert row["flag"] == "UNSTABLE"
        assert row["psi_value"] > 0.25
        assert result.status == "FAIL"

    def test_warnings_populated_on_fail(self, dist_unstable_shift):
        exp, act = dist_unstable_shift
        result = psi_check(exp, act, columns=["pd_score"])

        assert len(result.warnings) > 0
        assert any("pd_score" in w for w in result.warnings)

    def test_no_warnings_on_stable(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"])

        assert result.warnings == []


# ── PSI: summary DataFrame ──────────────────────────────────────────────────


class TestPsiSummaryDf:

    def test_summary_has_expected_columns(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"])

        expected_cols = {
            "variable", "psi_value", "flag",
            "n_bins_shifted", "n_expected", "n_actual",
        }
        assert expected_cols.issubset(set(result.summary_df.columns))

    def test_one_row_per_column(self, dist_multivar):
        exp, act = dist_multivar
        result = psi_check(exp, act, columns=["pd_score", "ltv"])

        assert len(result.summary_df) == 2
        assert set(result.summary_df["variable"]) == {"pd_score", "ltv"}

    def test_multivar_mixed_status(self, dist_multivar):
        """One stable + one unstable variable → overall FAIL, flags differ."""
        exp, act = dist_multivar
        result = psi_check(exp, act, columns=["pd_score", "ltv"])

        flags = dict(zip(result.summary_df["variable"], result.summary_df["flag"]))
        assert flags["ltv"]      == "STABLE"
        assert flags["pd_score"] == "UNSTABLE"
        assert result.status     == "FAIL"

    def test_psi_values_are_non_negative(self, dist_stable):
        """PSI is a sum of (p-q)*log(p/q) terms — always >= 0."""
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"])

        assert (result.summary_df["psi_value"] >= 0).all()

    def test_n_expected_and_actual_match_input_size(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"])

        row = result.summary_df.iloc[0]
        assert row["n_expected"] == len(exp)
        assert row["n_actual"]   == len(act)


# ── PSI: details dict ───────────────────────────────────────────────────────


class TestPsiDetails:

    def test_bin_detail_present_for_each_column(self, dist_multivar):
        exp, act = dist_multivar
        result = psi_check(exp, act, columns=["pd_score", "ltv"])

        assert "pd_score" in result.details["bin_detail"]
        assert "ltv"      in result.details["bin_detail"]

    def test_bin_detail_has_correct_number_of_bins(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"], bins=10)

        detail = result.details["bin_detail"]["pd_score"]
        assert len(detail) == 10

    def test_bin_detail_psi_sum_equals_summary_value(self, dist_unstable_shift):
        """Sum of bin-level contributions must equal the reported PSI."""
        exp, act = dist_unstable_shift
        result = psi_check(exp, act, columns=["pd_score"])

        psi_from_summary = result.summary_df.iloc[0]["psi_value"]
        psi_from_bins    = result.details["bin_detail"]["pd_score"]["psi_contribution"].sum()
        assert abs(psi_from_summary - psi_from_bins) < 1e-10

    def test_thresholds_in_details(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"])

        thresholds = result.details["thresholds"]
        assert thresholds["stable"]   == 0.10
        assert thresholds["moderate"] == 0.25

    def test_custom_bins_reflected_in_details(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score"], bins=5)

        assert result.details["bins"] == 5
        assert len(result.details["bin_detail"]["pd_score"]) == 5


# ── PSI: inputs alternativos ─────────────────────────────────────────────────


class TestPsiAlternativeInputs:

    def test_accepts_series_input(self):
        rng = np.random.default_rng(10)
        exp = pd.Series(rng.beta(2, 18, 1000), name="pd_score")
        act = pd.Series(rng.beta(2, 18, 1000), name="pd_score")
        result = psi_check(exp, act)

        assert result.status == "PASS"
        assert "pd_score" in result.summary_df["variable"].values

    def test_nans_in_input_do_not_crash(self, dist_with_nans):
        exp, act = dist_with_nans
        result = psi_check(exp, act, columns=["pd_score"])

        assert result.status in ("PASS", "FAIL")  # must not raise
        assert result.summary_df["n_expected"].iloc[0] < 1000  # NaNs were dropped

    def test_columns_subset_respected(self, dist_multivar):
        """When columns= is passed, only those variables are evaluated."""
        exp, act = dist_multivar
        result = psi_check(exp, act, columns=["ltv"])

        assert list(result.summary_df["variable"]) == ["ltv"]

    def test_missing_column_adds_warning_and_skips(self, dist_stable):
        exp, act = dist_stable
        result = psi_check(exp, act, columns=["pd_score", "nonexistent_col"])

        assert any("nonexistent_col" in w for w in result.warnings)
        assert "nonexistent_col" not in result.summary_df["variable"].values

    def test_to_excel_generates_file(self, dist_unstable_shift, tmp_path):
        exp, act = dist_unstable_shift
        result = psi_check(exp, act, columns=["pd_score"])
        out = tmp_path / "psi_output.xlsx"
        result.to_excel(str(out))

        assert out.exists()
        meta = pd.read_excel(out, sheet_name="Metadata")
        assert meta["status"].iloc[0] == "FAIL"


# ── CSI: casos nominales ────────────────────────────────────────────────────


class TestCsiNominal:

    def test_stable_segment_passes(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(
            exp, act,
            score_col="pd_score",
            segment_col="risk_grade",
        )

        low_row = result.summary_df[result.summary_df["segment"] == "Low"].iloc[0]
        assert low_row["flag"] == "STABLE"

    def test_unstable_segment_flagged(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(
            exp, act,
            score_col="pd_score",
            segment_col="risk_grade",
        )

        high_row = result.summary_df[result.summary_df["segment"] == "High"].iloc[0]
        assert high_row["flag"] in ("MODERATE", "UNSTABLE")

    def test_overall_status_fail_when_any_segment_unstable(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(
            exp, act,
            score_col="pd_score",
            segment_col="risk_grade",
        )

        assert result.status == "FAIL"

    def test_warnings_identify_unstable_segments(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(
            exp, act,
            score_col="pd_score",
            segment_col="risk_grade",
        )

        assert any("High" in w for w in result.warnings)

    def test_all_segments_present_in_summary(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(
            exp, act,
            score_col="pd_score",
            segment_col="risk_grade",
        )

        assert set(result.summary_df["segment"]) == {"Low", "Medium", "High"}


# ── CSI: summary DataFrame ──────────────────────────────────────────────────


class TestCsiSummaryDf:

    def test_summary_has_expected_columns(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        expected_cols = {
            "segment", "csi_value", "flag",
            "n_expected", "n_actual",
            "segment_pct_exp", "segment_pct_act",
        }
        assert expected_cols.issubset(set(result.summary_df.columns))

    def test_csi_values_are_non_negative(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        assert (result.summary_df["csi_value"] >= 0).all()

    def test_segment_pct_exp_sums_to_one(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        assert abs(result.summary_df["segment_pct_exp"].sum() - 1.0) < 1e-9

    def test_segment_pct_act_sums_to_one(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        assert abs(result.summary_df["segment_pct_act"].sum() - 1.0) < 1e-9

    def test_bin_detail_sum_equals_csi_value(self, csi_frames):
        """Per-segment bin contributions must reconcile with reported CSI."""
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        for _, row in result.summary_df.iterrows():
            seg = str(row["segment"])
            csi_from_bins = (
                result.details["bin_detail"][seg]["psi_contribution"].sum()
            )
            assert abs(row["csi_value"] - csi_from_bins) < 1e-10, \
                f"Mismatch in segment '{seg}'"


# ── CSI: detalles y metadatos ────────────────────────────────────────────────


class TestCsiDetails:

    def test_details_contain_score_and_segment_cols(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        assert result.details["score_col"]   == "pd_score"
        assert result.details["segment_col"] == "risk_grade"

    def test_bin_detail_keys_match_segments(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        assert set(result.details["bin_detail"].keys()) == {"Low", "Medium", "High"}

    def test_test_name_is_csi_check(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")

        assert result.test_name == "csi_check"


# ── CSI: edge cases ──────────────────────────────────────────────────────────


class TestCsiEdgeCases:

    def test_segment_missing_in_actual_adds_warning(self, csi_frames):
        """
        A new segment in expected not present in actual (or vice versa)
        should warn and skip — not crash.
        Common scenario: a risk grade introduced after model development.
        """
        exp, act = csi_frames
        act_missing = act[act["risk_grade"] != "High"].copy()
        result = csi_check(
            exp, act_missing,
            score_col="pd_score",
            segment_col="risk_grade",
        )

        assert any("High" in w for w in result.warnings)
        assert "High" not in result.summary_df["segment"].values

    def test_to_excel_generates_file(self, csi_frames, tmp_path):
        exp, act = csi_frames
        result = csi_check(exp, act, score_col="pd_score", segment_col="risk_grade")
        out = tmp_path / "csi_output.xlsx"
        result.to_excel(str(out))

        assert out.exists()

    def test_custom_bins_reflected_in_bin_detail(self, csi_frames):
        exp, act = csi_frames
        result = csi_check(
            exp, act,
            score_col="pd_score",
            segment_col="risk_grade",
            bins=5,
        )

        for seg, detail_df in result.details["bin_detail"].items():
            assert len(detail_df) == 5, f"Segment '{seg}' has wrong bin count"

    def test_single_segment_does_not_crash(self):
        """Degenerate case: entire population in one segment."""
        rng = np.random.default_rng(55)
        n = 500
        exp = pd.DataFrame({"score": rng.beta(2, 18, n), "grade": "A"})
        act = pd.DataFrame({"score": rng.beta(3, 14, n), "grade": "A"})
        result = csi_check(exp, act, score_col="score", segment_col="grade")

        assert len(result.summary_df) == 1
        assert result.summary_df.iloc[0]["segment"] == "A"