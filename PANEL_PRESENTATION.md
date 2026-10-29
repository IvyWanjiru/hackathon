# GhostDemand AI — panel presentation guide

**Pitch:** GhostDemand turns requests a retailer could not fulfil into evidence for what to stock, change and follow up on next.

Open **http://127.0.0.1:5059/present**. This is a guided view of the working local app. It uses an isolated demo database and synthetic requests. The regular retail dashboard remains at `/demo`; the business workspace remains at `/dashboard`.

## Three-minute script and clicks

### 0:00–0:25 · The problem

“A sale tells a retailer what sold. It does not explain the customer who asked for a product that was unavailable, too expensive or too late. GhostDemand keeps that missed request and turns it into evidence the team can act on.”

Click **Start the three-minute demo**. Say clearly that the requests and outcomes are synthetic. The stage illustration is an authored example; the following steps use the actual app APIs.

### 0:25–1:00 · Capture

“Here is a request for two earbuds that are sold out. The backend extracts a draft: product, quantity, budget and blocker. I can correct it before saving. The budget is an expression of interest; it is not revenue.”

Click **Extract the demand signal**. Check product, quantity, budget, currency and blocker. Tick the review box, then **Save reviewed sample**. If repeating a rehearsal, the app reports an existing request instead of adding a duplicate. This does not overwrite its earlier review.

### 1:00–1:25 · Evidence

“The workspace groups related requests and preserves the original wording. We can see how many requests, known references and channel labels support this opportunity. Transparent business rules suggest a next step based on the recorded blockers.”

Advance to **Evidence**. Point to the current counts, original messages and explanation. Known references are labels supplied with requests, not verified unique buyers. A channel label does not prove that a live provider account is connected. There are 24 starter requests across five products; subsequent counts reflect what has actually been saved in the demo.

### 1:25–1:50 · Action

“An insight is useful when someone owns the response. I assign this request to a team member with a date and an action. The workspace also records outcomes. Real recovered revenue needs an order link; this demo shows synthetic, team-entered amounts.”

Advance to **Take action**. Save the follow-up and show the confirmation. This updates the demo record and queue. It does not send a customer message or place an order. The recovery figure comes from existing sample records; assigning this follow-up does not create revenue.

### 1:50–2:35 · Machine learning

“We convert the request text into TF-IDF features: weighted character fragments. A logistic regression classifier learns weights for six reasons a purchase failed. Its scores guide the blocker suggestion; a person reviews it. Product parsing and prioritisation use separate rules. The accompanying notebook achieved macro F1 of 0.860 on 21 synthetic test requests, but accepted none automatically at its selected threshold. This is a workflow experiment, not evidence of real-customer accuracy.”

Advance to **The ML**. Point to the four-step diagram. The left card describes the baseline currently serving this demo; the right card describes a **separate notebook experiment**. Do not describe them as the same trained artifact. If the notebook changes and the evaluation snapshot no longer matches its hash, the score is hidden until it is reviewed again.

### 2:35–3:00 · Pilot

“Our next step is one retailer: connect the actual channels, label real unmet requests, and evaluate by channel, language and time. Then measure useful actions and order-verified recoveries. The contribution is the complete loop from missed request to evidence, ownership and learning.”

Finish on **What’s next**. Open the retail workspace for questions, or download the notebook and these notes. No customer trial, commercial uplift or production connection is claimed.

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

## Rehearsal checklist

1. Start the app from the project root: `PORT=5059 .venv/bin/python app.py`.
2. Visit `/present`, start the sample setup, and run capture → review → save → evidence → follow-up once before the session. Keep the local model dependencies installed. The local demo does not require a hosted language-model call.
3. Use text capture for the stage demonstration. Browser voice support and microphone/network permissions can vary.
4. Use **← / →** for chapters, **N** for notes and **F** for fullscreen when the browser supports it. Input fields keep their normal keyboard behaviour.
5. The three-minute timer is optional: start/pause with its play button and reset with ↺. Opening a scene never starts it automatically.
6. Repeat rehearsals safely: sample setup is idempotent; exact repeated samples reuse a request. The demo database persists, so counts can exceed the original 24. Never describe a repeated sample as a fresh customer.
7. Download the notebook and notes beforehand. Keep the regular workspace available for deeper questions. If an API fails, the app displays the error; it does not invent successful results.

## Method references

- [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [SetFit documentation](https://huggingface.co/docs/setfit/en/index)
- [Bitext retail/e-commerce dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset)
- [Amazon Science ESCI dataset](https://github.com/amazon-science/esci-data)
- [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The notebook is the experiment record; the app's live model-status endpoint is the serving record. Keep those two sources distinct when quoting results.
