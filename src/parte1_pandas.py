"""
FinTech Merchant Transaction Data Pipeline & Exploratory Analysis
==================================================================
Vectorized data cleaning, transaction KPI aggregations, automated data quality
auditing, and early pre-churn heuristics for merchant acquiring platforms.

Core capabilities:
- Vectorized cleansing of non-standard timestamps, currencies, and status codes.
- High-performance monthly KPI rollups (TPV, approval rate, channel distribution).
- Automated diagnostic report on data anomalies and schema violations.
- Heuristic merchant-at-risk scoring engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1.1  load_clean
# -----------------------------------------------------------------------------
def load_clean(path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV y devuelve un DataFrame listo para anÃ¡lisis.

    Documenta en `DECISIONS.md`

    Args:
        path: ruta al CSV.

    Returns:
        DataFrame limpio.
    """
    df = pd.read_csv(
        path,
        dtype={
            "transaction_id": "Int64",
            "merchant_id": "Int64",
            "segment": "category",
            "mcc": "string",
            "fla_churn90": "Int64",
            "cancellation_reason": "string",
        },
    )

    # 1. Parse Dates
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", dayfirst=True)
    df["reference_date"] = pd.to_datetime(df["reference_date"], errors="coerce", format="%Y-%m-%d")
    df["last_complaint_date"] = pd.to_datetime(df["last_complaint_date"], errors="coerce", format="%Y-%m-%d")
    df["dat_process"] = pd.to_datetime(df["dat_process"], errors="coerce", format="%Y-%m-%d")

    # 2. Clean amount
    if df["amount"].dtype == "O":
        s_amount = df["amount"].astype(str).str.replace("$", "", regex=False).str.strip()
        s_amount = s_amount.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        s_amount = s_amount.replace(["nan", "<NA>", ""], np.nan)
        df["amount"] = pd.to_numeric(s_amount, errors="coerce")

    # 3. Clean status and channel strings
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df["channel"] = df["channel"].astype(str).str.lower().str.strip()

    valid_status = ["approved", "denied", "reversed"]
    valid_channel = ["pos", "ecom", "pix", "tef"]
    df.loc[~df["status"].isin(valid_status), "status"] = pd.NA
    df.loc[~df["channel"].isin(valid_channel), "channel"] = pd.NA

    df["status"] = df["status"].astype("category")
    df["channel"] = df["channel"].astype("category")

    return df


# -----------------------------------------------------------------------------
# 1.2  monthly_kpis
# -----------------------------------------------------------------------------
def monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs mensuales por merchant. Una fila por (merchant_id, month).

    Columnas de salida:
      - merchant_id
      - month               (primer dÃ­a del mes, dtype datetime64[ns])
      - tpv                 (suma de amount para status == 'approved')
      - approval_rate       (% transacciones aprobadas sobre total)
      - pct_ecom            (% del TPV que viene del canal 'ecom')
      - n_tx                (nÃºmero de transacciones del mes)
    """
    df_valid = df.dropna(subset=["transaction_date", "merchant_id"]).drop_duplicates(subset=["merchant_id", "transaction_date", "amount", "channel", "status"]).copy()
    
    # Create month period timestamp
    df_valid["month"] = df_valid["transaction_date"].dt.to_period("M").dt.to_timestamp()

    # Boolean flags for aggregations
    df_valid["is_approved"] = df_valid["status"] == "approved"
    is_ecom = df_valid["channel"] == "ecom"

    df_valid["amount_approved"] = df_valid["amount"].where(df_valid["is_approved"], 0.0)
    df_valid["amount_ecom_approved"] = df_valid["amount"].where(df_valid["is_approved"] & is_ecom, 0.0)

    # Aggregation
    grouped = df_valid.groupby(["merchant_id", "month"])
    kpis = grouped.agg(
        tpv=("amount_approved", "sum"),
        n_tx=("transaction_id", "count"),
        approved_tx=("is_approved", "sum"),
        tpv_ecom=("amount_ecom_approved", "sum")
    ).reset_index()

    # Derived metrics
    kpis["approval_rate"] = (kpis["approved_tx"] / kpis["n_tx"]).fillna(0.0)
    
    # Safe division for pct_ecom
    tpv_mask = kpis["tpv"] > 0
    kpis["pct_ecom"] = 0.0
    kpis.loc[tpv_mask, "pct_ecom"] = kpis.loc[tpv_mask, "tpv_ecom"] / kpis.loc[tpv_mask, "tpv"]

    return kpis[["merchant_id", "month", "tpv", "approval_rate", "pct_ecom", "n_tx"]]


# -----------------------------------------------------------------------------
# 1.3  quality_report
# -----------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Devuelve un reporte de calidad de datos con al menos 5 problemas detectados.
    """
    issues = []

    # 1. Missing amounts
    missing_amount = df["amount"].isna().sum()
    if missing_amount > 0:
        issues.append({
            "column": "amount",
            "rows_affected": int(missing_amount),
            "impact": "SubestimaciÃ³n del TPV real de los merchants si se descartan. Ruido en modelos de ML.",
            "fix": "Imputar con la mediana de las transacciones del merchant o pedir correcciÃ³n al origen.",
        })

    # 2. Future transaction_date vs reference_date
    future_tx = (df["transaction_date"] > df["reference_date"]).sum()
    if future_tx > 0:
        issues.append({
            "column": "transaction_date",
            "rows_affected": int(future_tx),
            "impact": "Data leakage in ML si se usan estas transacciones para predecir churn en la reference_date.",
            "fix": "Filtrar todas las transacciones donde transaction_date > reference_date antes de generar KPIs/features.",
        })

    # 3. Temporal leakage: last_complaint_date > reference_date
    future_complaint = (df["last_complaint_date"] > df["reference_date"]).sum()
    if future_complaint > 0:
        issues.append({
            "column": "last_complaint_date",
            "rows_affected": int(future_complaint),
            "impact": "Data leakage en ML, la variable last_complaint_date contiene informaciÃ³n del futuro respecto a reference_date.",
            "fix": "Setear last_complaint_date a NaT para las filas donde sea estrictamente mayor a reference_date.",
        })

    # 4. Inconsistency: Cancellation reason for approved transactions
    logic_inconsistent = ((df["status"] == "approved") & df["cancellation_reason"].notna()).sum()
    if logic_inconsistent > 0:
        issues.append({
            "column": "cancellation_reason",
            "rows_affected": int(logic_inconsistent),
            "impact": "Inconsistencia lÃ³gica (una transacciÃ³n no puede estar aprobada y cancelada al mismo tiempo). Afecta mÃ©tricas y features.",
            "fix": "Investigar si 'approved' es el estado final o si la transacciÃ³n fue cancelada despuÃ©s, asumiendo entonces estado 'reversed'.",
        })

    # 5. Semantic Duplicates
    dup_subset = ["merchant_id", "transaction_date", "amount", "channel", "status"]
    duplicates = df.duplicated(subset=dup_subset).sum()
    if duplicates > 0:
        issues.append({
            "column": "Multiple (Business Key)",
            "rows_affected": int(duplicates),
            "impact": "Doble conteo en TPV y volumen de transacciones, inflando el rendimiento real de los merchants.",
            "fix": "Aplicar drop_duplicates usando el subconjunto de columnas lÃ³gicas que definen una transacciÃ³n Ãºnica.",
        })

    # 6. Target leakage: cancellation_reason is post-event
    leakage_cancellation = df["cancellation_reason"].notna().sum()
    if leakage_cancellation > 0:
        issues.append({
            "column": "cancellation_reason",
            "rows_affected": int(leakage_cancellation),
            "impact": "Data leakage evidente. La variable contiene info post-evento (solo se rellena al hacer churn). ArruinarÃ­a el ML.",
            "fix": "Eliminar cancellation_reason antes del split y modelado.",
        })

    return {
        "issues": issues,
        "summary": {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_issues": len(issues),
        }
    }


# -----------------------------------------------------------------------------
# 1.4  merchants_at_risk
# -----------------------------------------------------------------------------
def merchants_at_risk(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """
    Devuelve los `top_n` merchants con mayor "seÃ±al dÃ©bil" de pre-churn.
    Args:
        df: DataFrame limpio.
        top_n: nÃºmero de merchants a devolver.

    Returns:
        DataFrame ordenado por risk_score descendente.
    """
    # 1. Remove leakages / future data based on reference_date
    ref_date = df["reference_date"].dropna().iloc[0] if not df["reference_date"].isna().all() else pd.Timestamp("2025-09-30")
    df_valid = df[df["transaction_date"] <= ref_date].drop_duplicates(subset=["merchant_id", "transaction_date", "amount", "channel", "status"]).copy()
    
    # 2. Generar KPIs mensuales para ver tendencias
    kpis = monthly_kpis(df_valid)
    
    # Mes actual (M0), Mes previo (M-1)
    ref_month = pd.Period(ref_date, "M").to_timestamp()
    prev_month = ref_month - pd.DateOffset(months=1)
    
    kpi_pivot = kpis.pivot(index="merchant_id", columns="month", values=["tpv", "n_tx", "approval_rate"]).fillna(0.0)
    
    if ref_month in kpi_pivot["tpv"].columns and prev_month in kpi_pivot["tpv"].columns:
        tpv_current = kpi_pivot["tpv"][ref_month]
        tpv_prev = kpi_pivot["tpv"][prev_month]
        
        ar_current = kpi_pivot["approval_rate"][ref_month]
        ar_prev = kpi_pivot["approval_rate"][prev_month]
    else:
        tpv_current = pd.Series(0.0, index=kpi_pivot.index)
        tpv_prev = pd.Series(0.0, index=kpi_pivot.index)
        ar_current = pd.Series(0.0, index=kpi_pivot.index)
        ar_prev = pd.Series(0.0, index=kpi_pivot.index)

    # 3. DÃ­as desde la Ãºltima queja vÃ¡lida
    df_complaints = df_valid.dropna(subset=["last_complaint_date"]).copy()
    df_complaints = df_complaints[df_complaints["last_complaint_date"] <= ref_date]
    last_complaints = df_complaints.groupby("merchant_id")["last_complaint_date"].max()
    
    days_since_complaint = (ref_date - last_complaints).dt.days
    
    # 4. Scores and Signals
    results = pd.DataFrame(index=kpi_pivot.index)
    results["tpv_drop_ratio"] = (tpv_prev - tpv_current) / (tpv_prev + 1.0)
    results["ar_drop"] = ar_prev - ar_current
    results["recent_complaint"] = results.index.map(days_since_complaint)
    
    results["risk_score"] = 0.0
    results["top_signal"] = "none"
    
    # Calculo vectorizado del score
    s1 = np.where(results["tpv_drop_ratio"] > 0.3, results["tpv_drop_ratio"] * 50, 0.0)
    s2 = np.where(results["ar_drop"] > 0.1, results["ar_drop"] * 100, 0.0)
    
    rc = results["recent_complaint"].fillna(999)
    s3 = np.where(rc <= 30, (30 - rc) * 2, 0.0)
    
    results["risk_score"] = s1 + s2 + s3
    
    signals_arr = np.array([s1, s2, s3])
    max_idx = np.argmax(signals_arr, axis=0)
    
    signal_names = ["tpv_drop", "approval_rate_drop", "recent_complaint"]
    results["top_signal"] = np.where(results["risk_score"] > 0, np.array(signal_names)[max_idx], "none")
        
    results = results.reset_index().rename(columns={"index": "merchant_id"})
    results = results.sort_values("risk_score", ascending=False).head(top_n)
    
    return results[["merchant_id", "risk_score", "top_signal"]]


# -----------------------------------------------------------------------------
# Entry point â€” genera artefactos en outputs/
# -----------------------------------------------------------------------------
def main(csv_path: str) -> None:
    """Carga, calcula KPIs y reporte, y persiste en outputs/."""
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    df = load_clean(csv_path)

    kpis = monthly_kpis(df)
    kpis.to_csv(outputs / "monthly_kpis.csv", index=False)

    report = quality_report(df)
    (outputs / "quality_report.json").write_text(json.dumps(report, indent=2, default=str))

    at_risk = merchants_at_risk(df, top_n=200)
    at_risk.to_csv(outputs / "merchants_at_risk.csv", index=False)

    print("> Outputs generados en outputs/")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/transactions_sample.csv"
    main(csv)

