# Dataset Documentation (`data/`)

Synthetic transactional data for financial merchant analysis and churn prediction modeling.

## Files

### `transactions_sample.csv`
Synthetic transaction logs representing point-of-sale (POS), e-commerce, PIX, and TEF transactions.
- `transaction_id`: Unique transaction identifier
- `merchant_id`: Merchant identifier
- `transaction_date`: Timestamp of the transaction
- `amount`: Transaction amount
- `status`: Transaction status (`approved`, `denied`, `reversed`)
- `channel`: Transaction channel (`pos`, `ecom`, `pix`, `tef`)
- `cancellation_reason`: Specific reason if reversed or denied
- `reference_date`: Snapshot date for retrospective analysis
- `fla_churn90`: Churn target (1 if merchant churned within 90 days of snapshot)
- `segment`: Commercial segment (`SMB`, `MidMarket`, `Enterprise`)
- `mcc`: Merchant Category Code

### `merchants_context.json`
CRM metadata used by the retention agent tool to look up merchant profile, past 90-day TPV, and historical complaint volume.
