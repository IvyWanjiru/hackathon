# Later models: annotations and outcomes first

These recipes complete the same EDA → training → evaluation → export → inference
sequence for the other three proposed operations. They are code for you to run;
no transformer, recovery or forecast model was trained in preparing this guide.
Blank schemas deliberately refuse training until real target data is supplied.

Use a separate Python 3.11/3.12 learning environment as described in
[README.md](README.md). Install only the extras for the lesson you are doing.
Every command runs from the project root.

## A. Extract product, quantity, budget, currency and location

**Model:** `distilbert/distilbert-base-uncased` with a new token-classification
head. The [official token-classification tutorial](https://huggingface.co/docs/transformers/en/tasks/token_classification)
explains subword alignment and BIO labeling. General NER corpora can teach that
mechanism; their PERSON/ORG labels cannot replace your retail annotations.

### A1. Build your labeled dataset

Start with permitted, minimized customer-request contexts. Annotate the same five
entity types consistently. Example only:

```text
Need   2            navy       shirts      under  2000      KES         in  Nairobi
O      B-QUANTITY   B-PRODUCT  I-PRODUCT   O      B-BUDGET  B-CURRENCY  O   B-LOCATION
```

`B-` begins an entity, `I-` continues that entity, and `O` is outside an entity.
Store one JSON object per line with `id`, `tokens`, `ner_tags`, `group_id`,
`split`, `origin`. Use [entities_schema_example.jsonl](data/entities_schema_example.jsonl)
as a format example, not as a model-training dataset. Keep tokenization stable
(whitespace-separated tokens in this recipe); include punctuation in your
annotation policy. Product “navy shirts” and currency “KES” are different spans.

Collect short and long requests, absent fields, multiple quantities, negation,
different money formats, sizes/variants and channel styles. An arbitrary number
of messages does not ensure coverage of each entity. Have two reviewers label a
subset independently, reconcile disagreements, then freeze a policy and split.
Group customers/conversations/templates and duplicates together. Choose about
60/20/20 grouped partitions, with a later-period real test when dates permit.

### A2. EDA and preparation

```sh
python -m pip install -r ml_workflow/requirements-transformers.txt
python -m ml_workflow.profile_advanced entities \
  --data ml_workflow/work/business/entities.jsonl \
  --out ml_workflow/work/entities-eda.json
```

Inspect record/group counts, entity-tag distribution, max tokens and split
coverage. `entity_rows()` in [advanced.py](advanced.py) rejects invalid BIO
sequences, duplicated IDs and crossing groups/text. Inspect annotation errors
and class absence before training. The tokenizer check rejects sequences over
256 subword tokens rather than silently truncating a budget or location.

### A3. Train and evaluate

```sh
python -m ml_workflow.advanced entities \
  --data ml_workflow/work/business/entities.jsonl \
  --out ml_workflow/work/entities
```

The fast tokenizer maps original words to subwords. The first subword receives
the word's label; continuation/special tokens receive `-100`, so they do not
duplicate the training loss. DistilBERT is fine-tuned with cross-entropy.
Validation span F1 selects the best epoch; the held-out test is evaluated once.
The script saves tokenizer/model weights and `evaluation.json` with per-entity
precision, recall and F1. `--revision` pins a reviewed base-checkpoint commit.

Span F1 measures an **exact entity boundary and type** match. It does not establish
that “2k” became the right numeric budget. On a separate reviewed extraction
test, compare exact parsed product, quantity, budget and currency; test decimal
separators, ranges, missing currency and ambiguous location. Keep the app's
numeric/currency validation and editable preview. Do not automatically treat a
size “42” as a quantity or convert “KES 2000 total” into price per unit.

### A4. Serve locally

```sh
GHOST_ENTITY_MODEL="$PWD/ml_workflow/work/entities/model" \
python -m flask --app 'ml_workflow.advanced_service:create_app()' \
  run --host 127.0.0.1 --port 5063
curl -X POST http://127.0.0.1:5063/entities \
  -H 'Content-Type: application/json' \
  -d '{"text":"Need 2 navy shirts under 2000 KES in Nairobi"}'
```

The response includes text spans, offsets and scores for review. A future app
adapter would map validated spans into `draft_signal()` before confirmation.
The service does not write demand records or overwrite the current extractor.

## B. Predict which requests convert within 14 days

**Model:** [CatBoostClassifier](https://catboost.ai/docs/en/quickstart) for mixed
categorical/numeric features. This predicts an observed outcome under historical
business practices; it does not estimate the causal effect of a phone call.

### B1. Build the outcome dataset

There is no supplied public dataset that establishes GhostDemand request-to-order
recovery. Use first-party requests joined to verified orders. The app's current
`closed` status and manually recorded recovery amount alone are insufficient.
Start with [recovery_template.csv](data/recovery_template.csv):

| Field | Meaning |
|---|---|
| `id`, `group_id`, `split` | Unique request, conversation/customer group, time partition |
| `captured_at` | Time at which this model would have made the prediction |
| `features_at` | Timestamp of the saved feature snapshot; must be ≤ capture time |
| `observed_through` | Verified order-system coverage through this timestamp |
| `converted_at` | Earliest verified linked order time; blank if no observed conversion |
| `source`, `reason`, `product_id`, `currency` | Capture-time categorical features |
| `qty`, `budget`, `urgency` | Reviewed finite nonnegative capture-time numeric features |

Use UTC or explicit-offset dates and a documented order-linking policy. For
budget/quantity unknowns, design an explicit missing-value policy/indicator in a
future feature revision; this first recipe refuses missing numeric values rather
than silently filling them with zero. Record currency as a feature; budgets in
different currencies are not directly comparable totals.

The target is 1 when a verified order occurs within 14 days after capture and 0
when a fully observed 14-day window contains no such order. The code excludes
all immature windows, including recent open requests. A sample observation
window is not a negative merely because the customer did not reply.

Use chronological train/validation/test periods, whole customer groups and at
least a 14-day purge gap so training labels would have been known before the
next period's predictions. If a customer spans periods, assign or purge the
whole group. Do not use final status, recovered revenue, future outreach or the
final reviewer reason as capture-time predictors. If the blocker feature comes
from ML in deployment, evaluate using equivalent historical/out-of-fold
predictions; a perfect future ground-truth reason would overstate performance.

### B2. Explore and validate

```sh
python -m pip install -r ml_workflow/requirements-tabular.txt
python -m ml_workflow.profile_advanced recovery \
  --data ml_workflow/work/business/recovery.csv \
  --as-of 2026-10-28T00:00:00Z \
  --out ml_workflow/work/recovery-eda.json
```

Use your actual analysis cutoff. Inspect raw/mature counts, positive prevalence,
time coverage and channels per partition. Verify order-link coverage and label
timing; both positive and negative outcomes must appear in each partition.
`recovery_frame()` checks maturity, future feature timestamps, impossible order
dates, group overlap and time-window leakage before fitting anything.

### B3. Train, evaluate and export

```sh
python -m ml_workflow.advanced recovery \
  --data ml_workflow/work/business/recovery.csv \
  --as-of 2026-10-28T00:00:00Z \
  --out ml_workflow/work/recovery
```

CatBoost learns nonlinear relationships and handles named categorical columns.
The initial settings are 300 trees, depth 5 and learning rate .05; early stopping
uses validation loss, not test labels. This is a starting recipe, not an
established optimal configuration. The saved native artifact is `model.cbm`.

Read average precision (a PR-curve summary), test prevalence, top-20% precision
and Brier score. Compare against a constant training-prior baseline; Brier score
measures squared probability error, not just ranking. A small AP improvement
does not prove useful follow-up prioritization. Inspect channel/product slices,
workload and later-period performance. For probability use, reserve separate
calibration data and evaluate reliability before calling the score calibrated.
The supplied endpoint explicitly marks scores `calibrated=false`.

### B4. Serve a scoring API

```sh
GHOST_RECOVERY_MODEL="$PWD/ml_workflow/work/recovery/model.cbm" \
python -m flask --app 'ml_workflow.advanced_service:create_app()' \
  run --host 127.0.0.1 --port 5063
curl -X POST http://127.0.0.1:5063/recovery \
  -H 'Content-Type: application/json' \
  -d '{"source":"whatsapp","reason":"out_of_stock","product_id":"P07","currency":"KES","qty":2,"budget":5000,"urgency":2}'
```

A later app change can store versioned scores alongside the transparent
follow-up priority. Do not replace observed recovered revenue or promise a sale
from this score. Evaluate effects of prioritization separately from prediction
quality; the model can perpetuate the old follow-up policy's selection bias.

## C. Forecast weekly completed sales

**Model:** [LightGBM LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
with lag/calendar features. This lesson forecasts completed sales one week ahead;
it does not recover the sales hidden by stockouts or price changes.

### C1. Get and prepare history

Follow [the UCI instructions](DATASETS.md#4-uci-online-retail-ii-forecasting-practice)
to prepare `work/uci/weekly_sales.csv`, or use your store's history with the
`sku,week,quantity` schema in [weekly_sales_template.csv](data/weekly_sales_template.csv).
`week` is the Monday date of a completed week. Aggregate each SKU/week once.

Choose the sales target deliberately: this recipe uses gross positive unit
sales, excluding returns/cancellations. Net units need a separate definition.
Resolve missing calendar weeks, SKU launch/retirement dates, duplicates and
availability before assigning zero. Drop partial weeks. Retain original units
and a stable SKU/variant identifier.

### C2. Explore and construct leakage-safe features

```sh
python -m ml_workflow.profile_advanced forecast \
  --data ml_workflow/work/uci/weekly_sales.csv \
  --out ml_workflow/work/forecast-eda.json
```

Inspect SKU/date coverage, zero share and quantity distribution. Investigate
large orders rather than deleting them automatically. `forecast_frame()`
rejects duplicate keys, non-Monday dates, gaps and invalid quantities. It creates
lag 1, 4 and 52 plus a four-week mean **shifted by one week**; today's target
never enters today's rolling mean. The annual lag needs at least 52 prior weeks,
plus sufficient later evaluation weeks (roughly 67+ continuous weeks per SKU).

For next Monday's forecast, `lag_1` is the last completed week's units,
`lag_4` is four completed weeks back, `lag_52` is 52 back, and `mean_4` is the
mean of the last four completed weeks. `week_of_year` describes the target week.
Features must be constructed in exactly this way at serving time.

### C3. Train, backtest and save

```sh
python -m ml_workflow.advanced forecast \
  --data ml_workflow/work/uci/weekly_sales.csv \
  --out ml_workflow/work/forecast
```

The code uses the earliest ~60% of eligible calendar weeks for training, the next
20% for validation/early stopping, and the final 20% for testing. All SKUs share
the same time boundaries. Known SKUs are categorical; new-SKU cold starts are
rejected for a separate policy rather than silently assigned an existing SKU.

Compare test MAE against previous-week and annual seasonal-naive (lag 52)
forecasts. Inspect `test_predictions.csv`, not just one overall number; a few
high-volume SKUs can dominate error. Repeat chronological backtests across
multiple earlier cutoffs before deployment, without reusing the test period to
tune. Add prediction intervals with an appropriate validated method before
using a forecast range; this first recipe outputs point forecasts only.

This is a **rolling one-week-ahead** backtest: when predicting week t+1, actual
sales for week t are available. It is not a prediction of the entire future test
block using no new actuals. Multiweek forecasting needs a different evaluation
and feature-generation policy. The model saves native `model.txt`, category
metadata in `serving.json` and the test metrics.

### C4. Deploy a next-week prediction

```sh
GHOST_FORECAST_DIR="$PWD/ml_workflow/work/forecast" \
python -m flask --app 'ml_workflow.advanced_service:create_app()' \
  run --host 127.0.0.1 --port 5063
```

Send the exact known SKU and features built from completed history. Example
payload shape (replace values and SKU with your actual history):

```json
{"sku":"YOUR_KNOWN_SKU","week_of_year":44,"lag_1":12,"lag_4":9,"lag_52":10,"mean_4":11.5}
```

POST this JSON to `/forecast`; a valid known SKU returns a nonnegative predicted
unit quantity for one week. A nightly/weekly batch can build these features and
store versioned forecasts for purchasing review. Do not mix forecast units,
customer budget totals and observed recovered revenue in one business metric.

## Deploy and monitor the later models together

[advanced_service.py](advanced_service.py) loads any configured subset once per
worker. `/health` reports which models are configured; unconfigured endpoints
return 503. For inference behind your chosen production proxy:

```sh
python -m pip install gunicorn
GHOST_ENTITY_MODEL="$PWD/ml_workflow/work/entities/model" \
GHOST_RECOVERY_MODEL="$PWD/ml_workflow/work/recovery/model.cbm" \
GHOST_FORECAST_DIR="$PWD/ml_workflow/work/forecast" \
gunicorn --workers 1 --bind 127.0.0.1:5063 \
  'ml_workflow.advanced_service:create_app()'
```

Install the appropriate extras and train the selected artifacts first. Use the
same token, HTTPS/proxy, latency, payload and artifact-version controls described
in the main guide. Check health, valid predictions, missing models, malformed
inputs, long texts, unseen SKUs and model-load failure on the actual deployment
machine. Freeze dependencies after those checks; no cloud account was configured.

Monitor entity exact-field errors and reviewer corrections, recovery maturity/
ranking/calibration, and forecast MAE/bias by SKU over completed weeks. Keep a
versioned model, feature definition and training manifest for every deployment.
Retraining is offline and evaluated before rollout; rollback restores the prior
artifact/configuration. Output integration into the app's preview, follow-up
queue and purchase planning is a separate product change once these models pass
real business evaluation.
