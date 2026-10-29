# Verified practice run

Run on 28 October 2026 using the local, synthetic GhostDemand starter data.
These files retain actual executed results; they are not production benchmarks.

- `eda.md` / `eda.json`: 108 synthetic rows, six classes, missing dates and
  near-duplicate review candidates.
- `reason-evaluation.md` / `.json`: 66 training, 21 validation, 21 test rows;
  macro F1 about .808 on the synthetic test. Validation selected C=.5; no tested
  threshold met the selective-accuracy policy, so the candidate defers all cases.
- `matching-evaluation.json`: lexical retrieval on eight fictional held-out
  queries over eight fictional SKUs. This is not a MiniLM training result.
- `test_predictions.csv`: local test row IDs, predictions and scores.

The executable artifacts remain under ignored `../work/`. No live business data
was exported. The app's active backend was not changed. Optional transformer,
tabular-model and Docker recipes were not executed. The complete local test
suite passed 44 tests at this stage; see tests/test_ml_workflow.py for the
learning-specific checks.
