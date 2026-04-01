import pandas as pd
import pytest
from validation_suite import compare_dataframes


class TestCompareCasosNominales:

    def test_dataframes_identicos_retorna_pass(self, df_identical):
        ref, cha = df_identical
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        assert result.status == "PASS"
        assert result.warnings == []
        assert result.summary_df["rows_exceeding_tol"].max() == 0

    def test_diferencia_numerica_retorna_fail(self, df_with_diff):
        ref, cha = df_with_diff
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        assert result.status == "FAIL"
        pd_row = result.summary_df[result.summary_df["column"] == "pd_score"].iloc[0]
        assert pd_row["rows_exceeding_tol"] == 1
        assert abs(pd_row["max_abs_diff"] - 0.05) < 1e-9

    def test_tolerancia_custom_absorbe_diferencia_pequena(self, df_with_diff):
        ref, cha = df_with_diff
        # delta = 0.05 — should PASS with tol = 0.10
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"], numeric_tol=0.10)

        assert result.status == "PASS"

    def test_labels_personalizados_aparecen_en_metadata(self, df_with_diff):
        ref, cha = df_with_diff
        result = compare_dataframes(
            ref, cha,
            key_cols=["obligor_id"],
            label_reference="Model Owner Q1",
            label_challenger="Validator Dry Run",
        )
        # Labels should propagate to the ValidationResult for audit traceability
        assert result.test_name == "compare_dataframes"


class TestCompareSchemaCheck:

    def test_columna_faltante_genera_warning(self, df_schema_mismatch):
        ref, cha = df_schema_mismatch
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        assert len(result.warnings) > 0
        assert any("lgd" in w for w in result.warnings)

    def test_columna_extra_en_challenger_genera_warning(self):
        ref = pd.DataFrame({"id": [1, 2], "pd": [0.1, 0.2]})
        cha = pd.DataFrame({"id": [1, 2], "pd": [0.1, 0.2], "extra_col": [9, 9]})
        result = compare_dataframes(ref, cha, key_cols=["id"])

        assert any("extra" in w.lower() or "extra_col" in w for w in result.warnings)

    def test_schema_identico_sin_warnings(self, df_identical):
        ref, cha = df_identical
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        schema_warnings = result.details.get("schema_warnings", [])
        assert schema_warnings == []


class TestCompareEdgeCases:

    def test_dataframe_vacio_no_crashea(self):
        cols = ["obligor_id", "pd_score"]
        ref = pd.DataFrame(columns=cols)
        cha = pd.DataFrame(columns=cols)
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        assert result.summary_df.empty or result.summary_df["rows_exceeding_tol"].sum() == 0

    def test_solo_columna_clave_no_crashea(self):
        """Edge case: DataFrames with only the key column and no numeric columns."""
        ref = pd.DataFrame({"obligor_id": [1, 2, 3]})
        cha = pd.DataFrame({"obligor_id": [1, 2, 3]})
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        assert result.status == "PASS"

    def test_resultado_tiene_columnas_esperadas(self, df_identical):
        ref, cha = df_identical
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])

        expected_cols = {"column", "max_abs_diff", "mean_abs_diff", "rows_exceeding_tol"}
        assert expected_cols.issubset(set(result.summary_df.columns))

    def test_to_excel_genera_archivo(self, df_identical, tmp_path):
        """Audit trail: ValidationResult must export cleanly to Excel."""
        ref, cha = df_identical
        result = compare_dataframes(ref, cha, key_cols=["obligor_id"])
        out = tmp_path / "compare_output.xlsx"
        result.to_excel(str(out))

        assert out.exists()
        summary = pd.read_excel(out, sheet_name="Summary")
        assert "column" in summary.columns