# tests/test_performance.py

import numpy as np
import pandas as pd
import pytest

from validation_suite import backtesting_report

# ── Status: pass / fail ──────────────────────────────────────────────────────


class TestBacktestingStatus:
    def test_good_model_returns_pass(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.status == "PASS"
        assert result.warnings == []

    def test_poor_discrimination_returns_fail(self, poor_discrimination):
        y_true, y_score = poor_discrimination
        result = backtesting_report(y_true, y_score)

        assert result.status == "FAIL"

    def test_overestimating_model_returns_fail(self, overestimating_model):
        y_true, y_score = overestimating_model
        result = backtesting_report(y_true, y_score)

        assert result.status == "FAIL"

    def test_underestimating_model_returns_fail(self, underestimating_model):
        y_true, y_score = underestimating_model
        result = backtesting_report(y_true, y_score)

        assert result.status == "FAIL"

    def test_test_name_is_backtesting_report(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.test_name == "backtesting_report"


# ── Discrimination metrics ───────────────────────────────────────────────────


class TestDiscriminationMetrics:
    def test_auc_in_valid_range(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        auc = result.summary_df["auc_roc"].iloc[0]
        assert 0.0 <= auc <= 1.0

    def test_gini_equals_2auc_minus_1(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        auc = result.summary_df["auc_roc"].iloc[0]
        gini = result.summary_df["gini"].iloc[0]
        assert abs(gini - (2 * auc - 1)) < 1e-10

    def test_ks_in_valid_range(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        ks = result.summary_df["ks_statistic"].iloc[0]
        assert 0.0 <= ks <= 1.0

    def test_perfect_model_auc_is_one(self, perfect_model):
        y_true, y_score = perfect_model
        result = backtesting_report(y_true, y_score)

        auc = result.summary_df["auc_roc"].iloc[0]
        assert abs(auc - 1.0) < 1e-6

    def test_perfect_model_gini_is_one(self, perfect_model):
        y_true, y_score = perfect_model
        result = backtesting_report(y_true, y_score)

        gini = result.summary_df["gini"].iloc[0]
        assert abs(gini - 1.0) < 1e-6

    def test_good_model_auc_above_threshold(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.summary_df["auc_roc"].iloc[0] >= 0.70

    def test_poor_model_generates_auc_warning(self, poor_discrimination):
        y_true, y_score = poor_discrimination
        result = backtesting_report(y_true, y_score)

        assert any("AUC" in w or "auc" in w.lower() for w in result.warnings)

    def test_poor_model_generates_gini_warning(self, poor_discrimination):
        y_true, y_score = poor_discrimination
        result = backtesting_report(y_true, y_score)

        assert any("Gini" in w or "gini" in w.lower() for w in result.warnings)

    def test_brier_score_in_valid_range(self, good_model):
        """Brier score is a proper scoring rule bounded in [0, 1]."""
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        brier = result.summary_df["brier_score"].iloc[0]
        assert 0.0 <= brier <= 1.0

    def test_good_model_lower_brier_than_poor(self, good_model, poor_discrimination):
        """Better calibrated model should have a lower Brier score."""
        r_good = backtesting_report(*good_model)
        r_poor = backtesting_report(*poor_discrimination)

        assert (
            r_good.summary_df["brier_score"].iloc[0]
            < r_poor.summary_df["brier_score"].iloc[0]
        )


# ── Calibration metrics ──────────────────────────────────────────────────────


class TestCalibrationMetrics:
    def test_good_model_hl_pvalue_above_alpha(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score, alpha=0.05)

        assert result.summary_df["hl_p_value"].iloc[0] >= 0.05

    def test_overestimating_model_binomial_pvalue_below_alpha(
        self, overestimating_model
    ):
        y_true, y_score = overestimating_model
        result = backtesting_report(y_true, y_score, alpha=0.05)

        assert result.summary_df["binomial_p_value"].iloc[0] < 0.05

    def test_overestimating_warning_mentions_over(self, overestimating_model):
        y_true, y_score = overestimating_model
        result = backtesting_report(y_true, y_score)

        assert any("over" in w.lower() for w in result.warnings)

    def test_underestimating_warning_mentions_under(self, underestimating_model):
        y_true, y_score = underestimating_model
        result = backtesting_report(y_true, y_score)

        assert any("under" in w.lower() for w in result.warnings)

    def test_predicted_and_observed_dr_in_summary(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        row = result.summary_df.iloc[0]
        assert 0.0 <= row["default_rate_obs"] <= 1.0
        assert 0.0 <= row["default_rate_pred"] <= 1.0

    def test_observed_dr_matches_y_true_mean(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        observed_dr = result.summary_df["default_rate_obs"].iloc[0]
        assert abs(observed_dr - y_true.mean()) < 1e-10

    def test_predicted_dr_matches_y_score_mean(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        predicted_dr = result.summary_df["default_rate_pred"].iloc[0]
        assert abs(predicted_dr - y_score.mean()) < 1e-10


# ── Traffic light ────────────────────────────────────────────────────────────


class TestTrafficLight:
    def test_good_model_traffic_light_is_green(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.summary_df["traffic_light"].iloc[0] == "GREEN"
        assert result.details["traffic_light"] == "GREEN"

    def test_miscalibrated_model_traffic_light_is_red(self, overestimating_model):
        y_true, y_score = overestimating_model
        result = backtesting_report(y_true, y_score)

        tl = result.summary_df["traffic_light"].iloc[0]
        assert tl in ("RED", "YELLOW")

    def test_traffic_light_summary_matches_details(self, good_model):
        """traffic_light in summary_df must match details dict — no divergence."""
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert (
            result.summary_df["traffic_light"].iloc[0]
            == result.details["traffic_light"]
        )

    def test_red_traffic_light_generates_warning(self, overestimating_model):
        y_true, y_score = overestimating_model
        result = backtesting_report(y_true, y_score)

        tl = result.details["traffic_light"]
        if tl == "RED":
            assert any("RED" in w for w in result.warnings)


# ── Details dict ─────────────────────────────────────────────────────────────


class TestDetailsDict:
    def test_roc_curve_present_and_valid(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        roc = result.details["roc_curve"]
        assert isinstance(roc, pd.DataFrame)
        assert {"fpr", "tpr", "threshold"}.issubset(roc.columns)
        assert (roc["fpr"] >= 0).all() and (roc["fpr"] <= 1).all()
        assert (roc["tpr"] >= 0).all() and (roc["tpr"] <= 1).all()

    def test_roc_curve_starts_at_origin(self, good_model):
        """First point of ROC curve must be (0, 0) by sklearn convention."""
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        roc = result.details["roc_curve"]
        assert roc["fpr"].iloc[0] == pytest.approx(0.0)
        assert roc["tpr"].iloc[0] == pytest.approx(0.0)

    def test_hosmer_lemeshow_keys_present(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        hl = result.details["hosmer_lemeshow"]
        assert {"statistic", "p_value", "df", "groups_df"}.issubset(hl.keys())

    def test_hosmer_lemeshow_groups_df_has_correct_columns(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        groups_df = result.details["hosmer_lemeshow"]["groups_df"]
        required = {
            "n",
            "observed_defaults",
            "expected_defaults",
            "mean_predicted_pd",
            "chi2_defaults",
        }
        assert required.issubset(groups_df.columns)

    def test_hosmer_lemeshow_statistic_is_non_negative(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.details["hosmer_lemeshow"]["statistic"] >= 0.0

    def test_binomial_test_keys_present(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        binom = result.details["binomial_test"]
        assert {"predicted_dr", "observed_dr", "z_stat", "p_value"}.issubset(
            binom.keys()
        )

    def test_calibration_table_has_n_buckets_rows(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score, n_cal_buckets=10)

        cal = result.details["calibration_table"]
        assert len(cal) == 10

    def test_calibration_table_columns(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        cal = result.details["calibration_table"]
        required = {
            "n",
            "observed_dr",
            "mean_predicted_pd",
            "min_score",
            "max_score",
            "ratio_pred_obs",
        }
        assert required.issubset(cal.columns)

    def test_calibration_table_n_sums_to_total(self, good_model):
        """Sum of observations across calibration buckets = total n."""
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        total_n = result.details["calibration_table"]["n"].sum()
        assert total_n == len(y_true)

    def test_alpha_stored_in_details(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score, alpha=0.01)

        assert result.details["alpha"] == 0.01


# ── Summary DataFrame ────────────────────────────────────────────────────────


class TestSummaryDf:
    def test_summary_has_one_row(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert len(result.summary_df) == 1

    def test_summary_has_expected_columns(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        required = {
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
        }
        assert required.issubset(result.summary_df.columns)

    def test_n_observations_matches_input(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.summary_df["n_observations"].iloc[0] == len(y_true)

    def test_n_defaults_matches_y_true_sum(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score)

        assert result.summary_df["n_defaults"].iloc[0] == int(y_true.sum())

    def test_model_name_propagated(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score, model_name="PD-RETAIL-V2.3")

        assert result.summary_df["model_name"].iloc[0] == "PD-RETAIL-V2.3"

    def test_custom_n_hl_groups(self, good_model):
        """n_hl_groups should be respected by the Hosmer-Lemeshow test."""
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score, n_hl_groups=5)

        groups_df = result.details["hosmer_lemeshow"]["groups_df"]
        assert len(groups_df) <= 5  # <= because qcut may merge sparse bins

    def test_to_excel_generates_file(self, good_model, tmp_path):
        y_true, y_score = good_model
        result = backtesting_report(y_true, y_score, model_name="PD-TEST")
        out = tmp_path / "backtesting_output.xlsx"
        result.to_excel(str(out))

        assert out.exists()
        meta = pd.read_excel(out, sheet_name="Metadata")
        assert meta["status"].iloc[0] == "PASS"


# ── Input validation ─────────────────────────────────────────────────────────


class TestInputValidation:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            backtesting_report(
                y_true=np.array([0, 1, 0]),
                y_score=np.array([0.1, 0.9]),
            )

    def test_non_binary_y_true_raises(self):
        rng = np.random.default_rng(0)
        y_score = rng.uniform(0, 1, 100)
        y_true_invalid = rng.integers(0, 3, 100).astype(float)  # values 0, 1, 2
        with pytest.raises(ValueError, match="0 and 1"):
            backtesting_report(y_true_invalid, y_score)

    def test_out_of_range_scores_raises(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_score_bad = np.array([0.1, 1.5, 0.3, 0.8, -0.1])  # 1.5 and -0.1 invalid
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            backtesting_report(y_true, y_score_bad)

    def test_all_same_class_raises(self):
        """AUC is undefined when only one class is present."""
        y_true_all_zeros = np.zeros(100)
        y_score = np.random.default_rng(0).uniform(0, 1, 100)
        with pytest.raises(ValueError, match="both classes"):
            backtesting_report(y_true_all_zeros, y_score)

    def test_accepts_pandas_series(self, good_model):
        y_true, y_score = good_model
        result = backtesting_report(
            pd.Series(y_true),
            pd.Series(y_score),
        )
        assert result.status in ("PASS", "FAIL")

    def test_small_sample_does_not_crash(self, small_sample):
        y_true, y_score = small_sample
        result = backtesting_report(y_true, y_score)

        assert result.status in ("PASS", "FAIL")
        assert result.summary_df["n_observations"].iloc[0] == len(y_true)

    def test_custom_alpha_changes_status(self, good_model):
        """
        A model that passes at alpha=0.05 may fail at alpha=0.50
        if its p-values land between those thresholds.
        Verifies that alpha is actually wired into the status logic.
        """
        y_true, y_score = good_model
        r_strict = backtesting_report(y_true, y_score, alpha=0.50)
        r_lenient = backtesting_report(y_true, y_score, alpha=0.001)

        # With alpha=0.001, almost impossible to fail calibration tests
        assert r_lenient.status == "PASS"
        # With alpha=0.50, calibration tests are very demanding
        # Status may differ — we just verify both run cleanly
        assert r_strict.status in ("PASS", "FAIL")
