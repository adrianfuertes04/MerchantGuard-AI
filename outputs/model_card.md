# Model Card — Churn Predictor v4.0 (XGBoost Final)

## Target
`fla_churn90`: 1 si el merchant abandona la plataforma en los próximos 90 días.

## Features
- Numéricas: txn_count, tpv_total, tpv_approved, approval_rate, denial_rate, pct_ecom, pct_pix, days_since_complaint
- Categóricas: segment, mcc

## Modelo Seleccionado
XGBoost evaluado sobre 1838 muestras.

## Métricas (test set, 20% holdout estratificado por merchant)
- ROC-AUC: 0.6276
- PR-AUC: 0.1188
- Brier Score: 0.216
- Recall@1%: 0.0062
- Recall@5%: 0.0625
- Recall@10%: 0.15
