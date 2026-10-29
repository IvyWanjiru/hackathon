# Dataset exploration

Input: `ml_workflow/work/data/reasons.csv`. Rows: **108**. Grain: one labeled request context.

## Label distribution

| Reason | Rows | Share |
|---|---:|---:|
| abandoned | 18 | 16.7% |
| lead_time | 18 | 16.7% |
| not_stocked | 18 | 16.7% |
| out_of_stock | 18 | 16.7% |
| price_too_high | 18 | 16.7% |
| quote_rejected | 18 | 16.7% |

## Checks and interpretation

- Synthetic rows: 108/108. Synthetic evaluation does not measure customer accuracy.
- Duplicate text rows: 0; conflicting text groups: 0; duplicate IDs: 0.
- Unknown reasons: 0. Training must stop on unknown or conflicting labels.
- Linked groups: 108. Keep conversations, customer/template groups and duplicates in one partition.
- Possible paraphrase pairs: 2 (nearest-neighbor screen, cosine ≥0.80). Review before splitting.
- Missing values by field: `{'id': 0, 'text': 0, 'reason': 0, 'source': 0, 'conversation_id': 0, 'group_id': 0, 'captured_at': 108, 'origin': 0, 'is_synthetic': 0, 'verified': 0}`.
- Character-length percentiles: `{'0': 24.0, '25': 43.75, '50': 49.0, '75': 57.25, '95': 64.0, '100': 72.0}`.
- Sources: `{'synthetic': 108}`. Compare future performance by channel and language.
- Missing dates prevent a later-period deployment evaluation. Missing customer/template groups can leave leakage undetected.
- Class balance alone does not prove coverage: include difficult, ambiguous, multilingual and ordinary non-loss conversations.

Inspect the paired JSON for the checks and candidate row IDs. Resolve conflicts; supply grouping and dates for real data.
