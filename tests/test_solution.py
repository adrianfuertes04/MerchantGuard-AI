"""
Unit and Regression Tests for Merchant Data Pipeline & Analytics.

Validates:
  - CSV cleaning, date parsing, and type conversions
  - Monthly vector KPI calculations
  - Data quality audit detection logic
  - Merchant churn risk scoring heuristics
"""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.parte1_pandas import load_clean, monthly_kpis, quality_report, merchants_at_risk


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """Crea un CSV temporal con casos extremos para testear load_clean."""
    csv_content = """transaction_id,merchant_id,transaction_date,amount,status,channel,cancellation_reason,reference_date,fla_churn90,last_complaint_date,segment,mcc,dat_process
1,10,30/04/2024,"1.503,88",approved,pos,,2025-09-30,0,,SMB,4789,2024-04-30
2,10,2024-05-15,"50,00",denied,ecom,,2025-09-30,0,2025-05-14,SMB,7995,2024-05-15
3,11,2025-10-15,"200,50",approved,ecom,,2025-09-30,0,,SMB,5411,2025-10-15
4,11,2025-02-03,,INVALID,pos,bankruptcy,2025-09-30,1,,SMB,5411,2025-02-03
5,11,2025-02-03,,INVALID,pos,bankruptcy,2025-09-30,1,,SMB,5411,2025-02-03
"""
    file_path = tmp_path / "test_transactions.csv"
    file_path.write_text(csv_content, encoding="utf-8")
    return file_path


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Mini DataFrame limpio para tests rÃ¡pidos."""
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4],
        "merchant_id": [10, 10, 11, 11],
        "transaction_date": pd.to_datetime(["2025-01-15", "2025-01-20", "2025-02-01", "2025-10-03"]),
        "amount": [100.0, 50.0, 200.0, 75.0],
        "status": ["approved", "denied", "approved", "approved"],
        "channel": ["pos", "ecom", "ecom", "pos"],
        "cancellation_reason": [pd.NA, pd.NA, "fraud", pd.NA],
        "reference_date": pd.to_datetime(["2025-09-30"] * 4),
        "last_complaint_date": pd.to_datetime([pd.NaT, "2025-01-18", pd.NaT, "2025-11-01"])
    })


def test_smoke_pandas_imported(tiny_df: pd.DataFrame) -> None:
    """Smoke test mÃ­nimo para confirmar que pytest detecta el mÃ³dulo."""
    assert len(tiny_df) == 4


def test_load_clean_returns_dataframe(tiny_csv: Path) -> None:
    """Verifica que load_clean parsea correctamente fechas y montos europeos."""
    df = load_clean(tiny_csv)
    
    assert len(df) == 5
    assert df["amount"].dtype == "float64"
    assert df["amount"].iloc[0] == 1503.88
    assert pd.isna(df["amount"].iloc[3])
    
    assert pd.isna(df["status"].iloc[3])
    assert df["transaction_date"].iloc[0].month == 4


def test_monthly_kpis_columns_and_logic(tiny_df: pd.DataFrame) -> None:
    """Verifica el agrupamiento mensual y las lÃ³gicas vectorizadas."""
    out = monthly_kpis(tiny_df)
    
    expected_cols = {"merchant_id", "month", "tpv", "approval_rate", "pct_ecom", "n_tx"}
    assert expected_cols.issubset(out.columns)
    assert len(out) == 3
    
    m10 = out[out["merchant_id"] == 10].iloc[0]
    assert m10["n_tx"] == 2
    assert m10["tpv"] == 100.0
    assert m10["approval_rate"] == 0.5
    assert m10["pct_ecom"] == 0.0


def test_quality_report_finds_at_least_n_issues(tiny_df: pd.DataFrame) -> None:
    """Valida que el reporte detecta anomalias de calidad como leakage y missing data."""
    tiny_df = pd.concat([tiny_df, tiny_df.iloc[[0]]], ignore_index=True)
    report = quality_report(tiny_df)
    
    assert "issues" in report
    assert "summary" in report
    assert report["summary"]["n_issues"] > 0
    
    issue_columns = [i["column"] for i in report["issues"]]
    assert "transaction_date" in issue_columns
    assert "last_complaint_date" in issue_columns
    assert "cancellation_reason" in issue_columns
    assert "Multiple (Business Key)" in issue_columns


def test_merchants_at_risk_top_n_shape(tiny_df: pd.DataFrame) -> None:
    """Verifica el pipeline de la heurÃ­stica pre-churn."""
    out = merchants_at_risk(tiny_df, top_n=1)
    assert list(out.columns) == ["merchant_id", "risk_score", "top_signal"]
    assert len(out) == 1
    assert out["risk_score"].dtype == "float64"

