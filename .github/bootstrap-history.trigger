One-time trigger for the historical BSE SENSEX options backfill workflow.

Trigger re-armed after workflow activation.

Re-run one-time historical backfill.

# Trigger historical bootstrap after expiry validation and dashboard diagnostics improvements.

Re-attempt the legacy BSE SENSEX options archive before 2024-07-08 (from 2023-05-15) and validate actual coverage. Preserve all existing processed history. Record missing, unavailable and malformed dates; never synthesize option premiums or bid/ask quotes.

Run the targeted legacy-only backfill from 2023-05-15 through 2024-07-05. The workflow merges the older index range into the existing full index history before preprocessing and preserves all post-2024 option data.

Retry historical bootstrap after safe-push working-tree fix.
