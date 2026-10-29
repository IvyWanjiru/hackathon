# Reason-model experiment

Synthetic starter results are a pipeline exercise, not real-customer accuracy.

- Selected settings: `{'C': 0.5, 'threshold': 1.0}` (validation only).
- Untouched test: 21 rows; macro F1 0.808; accuracy 0.810.
- Coverage at score threshold 1.0: 0.0% (0 accepted).
- Accuracy among accepted predictions: None (null means none accepted).
- The saved candidate contains only training rows. Validation and test rows were never refitted into it.
- Selective accuracy is a small validation-sample observation, not a performance guarantee.

## Per-class test results

| Reason | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| abandoned | 0.500 | 1.000 | 0.667 | 3 |
| lead_time | 1.000 | 0.750 | 0.857 | 4 |
| not_stocked | 1.000 | 0.750 | 0.857 | 4 |
| out_of_stock | 0.667 | 0.667 | 0.667 | 3 |
| price_too_high | 1.000 | 0.667 | 0.800 | 3 |
| quote_rejected | 1.000 | 1.000 | 1.000 | 4 |

See evaluation.json for the confusion matrix, threshold curve, provenance and channel/origin slices.
Review errors by joining test_predictions.csv IDs to the local test CSV. Do not publish private customer text.
