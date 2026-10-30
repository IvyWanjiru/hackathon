# GhostDemand — speaker notes

The application now presents a business workspace. Training, model scores and notebook experiments stay in backend documentation; `/present` redirects to `/demo#activity`.

## Three-minute operational story

- **0:00–0:30:** A completed sale cannot reveal every request the shop could not fulfil. GhostDemand retains that evidence and gives the team a next step.
- **0:30–1:15:** In Channel activity, click **Send sample channel events**. State that these are synthetic events, not connected accounts. Four are saved and analysed automatically; one needs context. Repeat delivery is counted once.
- **1:15–2:00:** Inspect a captured request and its original message. Open Opportunities to show the related requests and recommended stock, price or delivery action.
- **2:00–2:35:** Assign an owner, follow-up date and next action. Distinguish team-entered sample recoveries from verified orders and real revenue impact.
- **2:35–3:00:** Explain the pilot: connect a retailer's channel relays, measure extraction quality and response time, then verify recoveries against actual orders.

Use the following material for technical questions, outside the business interface.

## Understand the model before the panel

### What exactly is learned?

This is supervised multiclass text classification. Each training row contains text and one human-reviewed blocker. There are six learned labels:

| Label | Meaning |
| --- | --- |
| `out_of_stock` | The retailer normally offers the product but it was unavailable. |
| `price_too_high` | The price prevented this purchase. |
| `not_stocked` | The retailer does not offer the requested product. |
| `quote_rejected` | A quotation was declined without a more specific recorded blocker. |
| `lead_time` | Delivery or availability timing prevented the purchase. |
| `abandoned` | The enquiry ended without completion, according to observed business context. |

An ordinary availability question does not establish a failed purchase. Production intake needs conversation context and an eligibility check for confirmed unmet requests before assigning a loss reason. The app also has rule labels such as unknown outcome and unsupported service; the six-class model does not learn those labels.

### How does the prediction work?

1. **Input:** text available when the request is captured. Voice capture supplies a transcript; this classifier does not recognise audio.
2. **Features:** character TF-IDF with 3–5 character fragments inside word boundaries, sublinear term frequency and up to 6,000 features. Frequent fragments in one message get weight; fragments common throughout the training corpus get less distinguishing weight. Character features can share some evidence across spelling variations, but robustness still needs testing.
3. **Training:** logistic regression learns a weight per feature per class. It minimises multiclass classification loss with regularisation to control overly large weights. Despite its name, logistic regression predicts categories here.
4. **Inference:** the model scores the six blockers. The current backend uses a configured score threshold before overriding the rule-derived blocker. These scores have not been calibrated as trustworthy probabilities.
5. **Review:** the app displays an editable draft. Explicit corrections become reviewed training labels. They improve a model only after a separate retraining, validation and release step; saving a correction does not instantly update model weights.

Example: “The requested 2 wireless earbuds are sold out” becomes numeric fragment features. Learned weights can favour `out_of_stock`. Quantity, currency and product extraction are separate parsing tasks. The action suggestion then uses the reviewed blocker and request evidence.

The feature extractor and classifier are both saved in the model artifact. Serving must reuse the training vocabulary and IDF weights, not fit a new vectorizer on incoming requests. The app's JSON artifact stores vocabulary and numeric weights, avoiding an executable pickle.

### What is ML, and what is a business rule?

| Operation | Current implementation |
| --- | --- |
| Blocker suggestion | TF-IDF + logistic regression, with rule fallback and human review |
| Product, quantity, budget, currency | Parsing rules and product-name handling; editable by a reviewer |
| Grouping product wording | Catalog aliases and normalization rules; preserve variants for review |
| Opportunity rank and next action | Explicit business rules, not a learned demand forecast |
| Ownership and follow-up | Operational records and dates |
| Recovery amount | Team-entered outcome, not a model prediction or verified order |

### What do the notebook results mean?

The accompanying `notebooks/GhostDemand_Training_Improved.ipynb` used **108 synthetic practice examples**, partitioned into **66 training, 21 validation and 21 test** examples. The splitter keeps provided message groups together. The starter set does not identify every paraphrase or template family; manually curate those groups before a stronger evaluation. Feature fitting uses training only; validation chooses settings; test measures the resulting candidate once.

- Test **macro F1: 0.860**. F1 balances precision and recall; macro averaging gives each blocker equal weight.
- Test **accuracy: 0.857** (18 of 21 classified correctly before abstention).
- Selected automatic-acceptance **threshold: 1.0**; test **coverage: 0%**. The small validation set did not support a useful automatic-acceptance policy under the notebook's minimum-evidence rule. No test prediction qualified for automatic acceptance.

The F1 is the candidate's raw classification result, not the accuracy of automatically accepted production decisions. A tiny synthetic test set has limited diversity and high uncertainty. Do not say “86% accurate on real customers,” “production ready,” or “guaranteed confidence.” The app's running saved baseline has its own version, training count and configured threshold, shown separately on screen. The walkthrough does not deploy the notebook candidate.

## Likely panel questions

**Why this model instead of a large language model?** It is a practical starting baseline for a small labeled dataset: fast local inference, low infrastructure cost, reproducible training and inspectable weights. Compare a richer model only on a shared independent test set. More complexity does not establish better results.

**What is your next model?** SetFit using a MiniLM sentence encoder is an optional few-shot experiment in the learning workflow. It is not serving this demo. Compare per-class F1, coverage, review workload and latency against TF-IDF using the same real held-out requests. Entity extraction, catalog retrieval and forecasting require different targets and evaluations.

**Where does the training data come from?** The included 108 examples are explicitly synthetic. The core production dataset must come from permitted, minimized business conversations with context and reviewed labels. Bitext provides auxiliary retail intent language; Amazon ESCI provides query/product relevance judgments; UCI Online Retail II provides historical transactions. None directly supplies verified six-class unmet-demand labels for this business. See [the dataset guide](ml_workflow/DATASETS.md) for adapters and limitations.

**What is new compared with a CRM or chatbot?** NLP classification and lost-deal tracking already exist. Our focus is an accessible operational loop around small missed retail requests: retain evidence, group product demand, explain a next step, assign ownership, record an outcome and collect corrections. This complements existing tools. Do not claim no competitor has similar capabilities.

**Does it understand Swahili or mixed language?** That has not been established. The practice data is mainly English. Collect separately reviewed English, Swahili and mixed-language evaluation sets before making a language claim. Translate only with provenance and separate validation.

**Are WhatsApp and email connected?** The prototype has receivers and source labels, including a Twilio-style WhatsApp webhook. Real accounts need configuration and message forwarding; email needs an adapter. An available receiver is not a connected business account.

**How do you prevent duplicate demand?** Existing capture uses source plus stable external event IDs. The panel sample derives a stable ID from its source and exact text, so repeating it is idempotent. Real integrations should use provider event IDs; two genuine identical customer requests can still be separate events.

**How would you validate business impact?** Use a measured retailer pilot: extraction agreement, blocker F1 by channel/language, review coverage, time to assigned action, and recoveries linked to verified orders. Compare outcomes against an agreed baseline. Customer budgets and synthetic demo recoveries do not demonstrate incremental revenue.

**What is required before deployment?** Representative first-party evaluation, a versioned release and rollback process, access controls, tenant isolation, provider webhook verification, retention controls and order-linked outcome measurement. Monitor label distributions and sampled reviewer disagreement over time. There is no unattended production deployment in this walkthrough.

## Rehearsal

Run `PORT=5059 .venv/bin/python app.py`, then open `/demo#activity`. Use the text-based sample event button so the stage demo does not depend on microphone permissions. The backend handles normalization, analysis, storage and duplicate handling. Confirming an unclear outcome remains a team action; it does not automatically create a verified training label.

Read [BACKEND_AUTOMATION.md](BACKEND_AUTOMATION.md) for real channel setup. No actual WhatsApp, email or voice provider account is connected by the sample traffic.
