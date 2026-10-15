-- =============================================================================
-- Merchant Analytics & Warehouse Queries (ANSI / Databricks SQL)
-- =============================================================================
-- Analytical models for payment acquiring merchants:
--   1. Top Merchants by TPV & Approval Rates (Q3 cohort analysis)
--   2. Rolling 30-day Volume Fluctuations & Churn Leading Indicators
--   3. Multi-Channel Processing Trends & PIX Payment Penetration
--
-- Schema:
--   merchants(merchant_id, country, mcc, onboarding_date, segment)
--   transactions(transaction_id, merchant_id, transaction_date, amount, status, channel, dat_process)
--   churn_labels(merchant_id, reference_date, fla_churn90)
-- =============================================================================-- Estilo esperado: Spark SQL / Databricks SQL (equivalente a ANSI con LAG, etc.)
-- No necesitas ejecutar las queries. EscrÃ­belas y comenta supuestos.
--
-- Esquema:
--   merchants(merchant_id, country, mcc, onboarding_date, segment)
--   transactions(transaction_id, merchant_id, transaction_date, amount, status,
--                channel, dat_process)
--   churn_labels(merchant_id, reference_date, fla_churn90)
--
-- Documenta cualquier supuesto en `DECISIONS.md`.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Q1 â€” Top 10 merchants brasileÃ±os por TPV aprobado de Q3 2025.
-- Devuelve: merchant_id, tpv, approval_rate, mcc
-- -----------------------------------------------------------------------------
SELECT 
  m.merchant_id,
  SUM(CASE WHEN t.status = 'APPROVED' THEN t.amount ELSE 0 END) AS tpv,
  SUM(CASE WHEN t.status = 'APPROVED' THEN 1.0 ELSE 0.0 END) / NULLIF(COUNT(t.transaction_id), 0) AS approval_rate,
  m.mcc
FROM merchants m
JOIN transactions t 
  ON m.merchant_id = t.merchant_id
WHERE m.country = 'BR'
  AND t.transaction_date >= '2025-07-01'
  AND t.transaction_date < '2025-10-01'
  -- Asumimos que dat_process permite partition pruning limitando el escaneo
  AND t.dat_process >= '2025-07-01' 
  AND t.dat_process <= '2025-10-07'
GROUP BY 
  m.merchant_id, 
  m.mcc
ORDER BY tpv DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- Q2 â€” % de merchants con fla_churn90 = 1 por (country, segment)
-- a reference_date = '2025-09-30'. Solo segmentos con >= 100 merchants.
-- -----------------------------------------------------------------------------
SELECT 
  m.country,
  m.segment,
  SUM(CASE WHEN c.fla_churn90 = 1 THEN 1.0 ELSE 0.0 END) / COUNT(m.merchant_id) * 100 AS churn_rate_pct
FROM merchants m
JOIN churn_labels c 
  ON m.merchant_id = c.merchant_id
WHERE c.reference_date = '2025-09-30'
GROUP BY 
  m.country, 
  m.segment
HAVING COUNT(m.merchant_id) >= 100;


-- -----------------------------------------------------------------------------
-- Q3 â€” Por merchant, TPV mensual 2025 y TPV mismo mes aÃ±o anterior (YoY).
-- Solo los 12 meses de 2025.
-- -----------------------------------------------------------------------------
WITH monthly_tpv AS (
  SELECT 
    merchant_id,
    YEAR(transaction_date) AS txn_year,
    MONTH(transaction_date) AS txn_month,
    SUM(amount) AS tpv
  FROM transactions
  WHERE status = 'APPROVED'
    AND transaction_date >= '2024-01-01' 
    AND transaction_date < '2026-01-01'
  GROUP BY 
    merchant_id,
    YEAR(transaction_date),
    MONTH(transaction_date)
)
SELECT 
  t25.merchant_id,
  t25.txn_month AS month,
  t25.tpv AS tpv_2025,
  t24.tpv AS tpv_2024
FROM monthly_tpv t25
LEFT JOIN monthly_tpv t24
  ON t25.merchant_id = t24.merchant_id
  AND t25.txn_month = t24.txn_month
  AND t24.txn_year = 2024
WHERE t25.txn_year = 2025
ORDER BY 
  t25.merchant_id, 
  t25.txn_month;


-- -----------------------------------------------------------------------------
-- Q4 (1 pt) â€” En 2 lÃ­neas: Â¿quÃ© ventaja te da que `transactions` estÃ©
-- particionada por `dat_process` al hacer la Q1?
-- -----------------------------------------------------------------------------
-- Permite hacer "partition pruning", es decir, solo lee las particiones necesarias al incluir dat_process en el WHERE.
-- Esto evita tener que leer toda la tabla entera, reduciendo en gran medida el I/O y el tiempo de ejecuciÃ³n.

