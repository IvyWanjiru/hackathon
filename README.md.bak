# GhostDemand AI

**Demand beyond the sale.** A retail and e-commerce assistant that turns
unsuccessful customer requests into evidence for what to stock, change and follow
up on next.

The refined description, source-backed competitor analysis, implemented gaps,
honest scope and three-minute presentation script are in [HACKATHON.md](HACKATHON.md).
Recommended model upgrades, verified dataset sources and a training/evaluation
plan are in [ML_ROADMAP.md](ML_ROADMAP.md).
For hands-on learning, follow the [step-by-step ML workflow](ml_workflow/README.md):
EDA, datasets, reproducible training, evaluation, optional transformer upgrades,
deployment examples and monitoring. Its candidates stay separate from the live app.
The real-business connection and reliability blueprint is in
[BUSINESS_INTEGRATION.md](BUSINESS_INTEGRATION.md).

## Run locally

Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app train-model
python app.py
```

Open [the landing page](http://127.0.0.1:5000), then choose **Open dashboard** for
[the live workspace](http://127.0.0.1:5000/dashboard), or open
[the retail demo](http://127.0.0.1:5000/demo) and click **Load retail demo**.
The demo contains 24 synthetic requests across nine channel labels, five product
mappings and a synthetic recovery. It uses a separate database and model. Loading
the samples again is idempotent and does not reset reviews or outcomes.

If port 5000 is occupied, use `PORT=5059 python app.py`, then open
[the landing page on port 5059](http://127.0.0.1:5059/).

No API key is needed for the local model or demo. Before a model is trained,
extraction rules still work. Model training is managed through the backend CLI
or API; the business dashboard has no training controls.

## The user workflow

1. **Capture request:** choose a channel, optionally add a customer alias, then
   paste a conversation or dictate a sales note in a supported browser.
2. **Check the signal:** review the extracted product/variant, quantity, total
   stated budget, currency, blocker, urgency and location before saving. Quantity
   defaults to one if it was not recognized; verify it. The optional verification
   checkbox marks the request details as reviewed; the backend can use verified
   blocker labels during training.
3. **Opportunity board:** see related requests, known customer references,
   channel coverage, blockers, urgency, currency-separated budgets and suggested
   actions. **View evidence** shows original messages and recent/prior seven-day
   counts.
4. **Signal inbox:** search/filter all records, review extractions, assign an
   owner, write a next step and choose open/following-up/recovered/closed. Record
   a recovery value only for an outcome confirmed by your team. No messages are
   sent and no payments are verified.
5. **Follow-up queue:** plan work independently of the opportunity-board filters.
   Assign an owner, set a follow-up date, record the next action and update the
   outcome. Filter by status, owner, urgency or overdue work. Planning a follow-up
   does not mark the extracted blocker as verified.
6. **Product aliases:** map customer wording to a canonical product/variant. Alias
   collisions are rejected; names differing by size/specification need review.
   Matching historical extracted names are regrouped. This is not live inventory.
7. **Channels & import:** preview a UTF-8 CSV, inspect row errors, then explicitly
   import valid rows. Export the currently filtered inbox as CSV.

Filters cover time, channel, status, blocker, review state and text. The inbox is
paginated. Anonymous requests are not counted as known customers. Repeated
customer references are counted once per product, even across channels.

### What the numbers mean

Stated budgets are expressions of interest, **not lost revenue**. Each budget has
its own currency. Totals are separated by currency; no exchange-rate assumptions
are made. Historical budgets without known currencies stay `UNKNOWN`. Optional
recovery amounts are team-recorded outcomes, counted only while the request has
`recovered` status. They are not a measured causal uplift.

Priority score = `2 × open requests + 3 × urgent open requests + known customer
references + number of channels`. High priority means score ≥16 or at least two
urgent open requests; medium means score ≥8. Suggestions follow the dominant
blocker on open requests, with the evidence count shown. They are explicit
business rules, not an ML sales forecast or a purchasing instruction. Trends are
observed counts, not forecasts; they honor the current dashboard filters.

## Machine learning

The local blocker classifier uses character TF-IDF (3–5 character sequences) and
multiclass logistic regression. It supports `out_of_stock`, `price_too_high`,
`not_stocked`, `quote_rejected`, `lead_time` and `abandoned`. Product and other field
extraction use rules or optional Claude extraction. Product grouping uses
human-approved aliases, not automatic semantic equivalence.

Training uses the 108 **synthetic** examples in `training_data.json` (including
retail examples) and explicitly reviewed signals. Unreviewed predictions never
become ground truth automatically. Case/whitespace-normalized duplicate texts
count once; the latest reviewed label wins over prior labels or matching starter
examples. Each reason needs at least four distinct examples. A review arriving
after the training snapshot remains pending for the next update.

Backend training records a stratified 25% development holdout accuracy, macro F1
and the synthetic/reviewed sample composition. Operators can inspect them through
`GET /api/model` and train with `flask --app app train-model` or
`POST /api/model/train`. Features are fit on the training
partition only, and the serving model is then refit on all examples. This small
development evaluation **does not establish real-customer accuracy**. Model
scores are uncalibrated; the default 0.55 threshold is a heuristic. Use a separate,
representative labeled retail dataset to evaluate a pilot.

Vocabulary, numeric weights and metadata are saved atomically as JSON. Missing,
corrupt, disabled or low-score models fall back to rules or configured Claude
extraction. No executable pickle/joblib artifacts or model uploads are accepted.
Implementation references: [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
and [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GHOST_DB` | `ghostdemand.db` beside `app.py` | Live SQLite database |
| `GHOST_DEMO_DB` | `ghostdemand-demo.db` beside `app.py` | Isolated synthetic demo database |
| `GHOST_ML_PATH` | `models/reason_classifier.json` | Live JSON model |
| `GHOST_ML_ENABLED` | `1` | Set to `0`, `false` or `no` to bypass live local ML |
| `GHOST_ML_THRESHOLD` | `0.55` | Minimum live ML score, between 0 and 1 |
| `GHOST_CURRENCY` | `KES` | Default capture currency; explicit currency in text takes precedence at preview |
| `GHOST_TIMEZONE` | `Africa/Nairobi` | Business timezone used for overdue and due-today follow-ups |
| `GHOST_SECRET_KEY` | random per process | Stable shared signing key for editable previews; set for multiple workers or restart persistence |
| `ANTHROPIC_API_KEY` | unset | Optional external extraction; live messages are then sent to Anthropic |
| `GHOST_MODEL` | existing Claude default | Choose a model ID available to your Anthropic account |
| `PORT` | `5000` | Local server port |
| `FLASK_DEBUG` | unset | Set to `1` for development debug mode |

Demo APIs use `X-Ghost-Demo: 1`. Demo training always uses its own model under
`models/demo_reason_classifier.json`; synthetic demo capture never calls Claude.
The demo selector is workspace routing, **not an authentication boundary**.

Voice dictation uses `SpeechRecognition` / `webkitSpeechRecognition`, when the
browser provides it. It starts only after clicking the voice control and requires
microphone permission. Some browsers use a remote recognition provider; it may
require connectivity. Transcripts are editable before saving. Typed input stays
available. See [MDN](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition).

The SQLite schema upgrades additively. Existing messages and predictions stay in
the live database; the prototype has no deletion action. The app binds to
localhost. Before exposing it publicly, add authentication, tenant isolation,
provider webhook verification and data retention controls. No production
connection, validated multilingual accuracy or live vendor integration is claimed.

## API

| Endpoint | Behavior |
| --- | --- |
| `POST /api/signals/preview` | `{text, source?, customer?, currency?, external_id?}` → editable fields, warnings, signed preview token; no write |
| `POST /api/signals` | Capture a request. Optionally send `fields`, `verified`, and `preview_token`. A preview is valid for one hour; identity fields must match. |
| `POST /api/signals/bulk` | `{items:[...]}`; 1–100 items, atomic writes |
| `GET /api/signals` | Filtered inbox; `page` parameter, 15 records per page |
| `GET /api/operations/follow-ups` | Operational queue and workspace totals; filters `status`, `assignment`, `owner`, `urgency`, `due`, `q`, `page`; 12 records per page |
| `PATCH /api/signals/<id>` | Review product, quantity, budget, currency, blocker, urgency; record owner/next action/status/recovery and optional `follow_up_on` (YYYY-MM-DD) |
| `PATCH /api/signals/<id>/review` | Backward-compatible `{reason}` review |
| `GET /api/insights` | KPIs, currency totals, opportunity summaries and recent signals |
| `GET /api/opportunities/evidence?key=...` | All requests supporting the selected opportunity in the current view |
| `GET /api/signals/export` | Filtered CSV, with formula-like text escaped |
| `GET, POST /api/catalog` | Canonical product names, SKU and aliases |
| `POST /api/import/preview` | Multipart `file`; CSV extraction preview with per-row errors, no write |
| `POST /api/events/website-search` | `{query, results_count, event_id?, customer?}`; capture only zero-result searches |
| `POST /api/events/stockout` | `{product, qty?, event_id?, customer?}`; preserve the explicit stockout reason |
| `POST /webhook/whatsapp` | Twilio-style form fields `Body`, `From`, `MessageSid` |
| `GET /api/model` | Model readiness, metadata and pending reviews |
| `POST /api/model/train` | Update the active workspace's model |
| `POST /api/demo/seed` | Requires demo header; idempotently add synthetic samples |

Query filters: `days=7|30|90|all`, `source`, `status`, `reason`, `review=needed`,
`q`. API default is all time; the dashboard starts with 30 days.

The follow-up queue uses its own filters: `status=active|open|follow_up|recovered|closed|all`,
`assignment=any|assigned|unassigned`, exact `owner` (case-insensitive),
`urgency=all|urgent`, and `due=all|overdue|today|unscheduled`. Overdue counts include
only active requests. Follow-up dates can be cleared by sending null. Recovery
values remain separated by currency. Model readiness and training metrics are
kept out of business workflows.

Sources: `whatsapp`, `website_search`, `salesperson`, `salesperson_voice`,
`chatbot`, `rejected_quote`, `unavailable_product`, `out_of_stock`,
`abandoned_enquiry`. Supported currencies: KES, USD, EUR, GBP, UGX, TZS, UNKNOWN.

Text limit: 10,000 characters; request size: 1 MB. Stable `external_id` values are
unique per channel and make retries idempotent. CSV reimports derive stable IDs
from file content and row number when none are provided. Changing a file or
omitting event IDs can create new requests; this is not probabilistic customer
identity resolution. Automated imports stay unreviewed.

## Verify

```sh
source .venv/bin/activate
python -W error::ResourceWarning -m unittest discover -s tests -v
node --check static/app.js
```

The tests use temporary models and databases. They cover model persistence and
fallback, reviewed-label learning, legacy migration, concurrent startup,
editable previews, currencies, aliases/variants, retries, recovery, CSV
preview/export, pagination/filters, atomic imports and demo/live isolation.
