# Datasets: what to use, what to label, what not to infer

Sources checked 28 October 2026. The scripts do not upload business data or
silently download large datasets. Keep original downloads alongside their
license/readme; adapters save SHA-256 hashes and sampling rules.

| Model | Start with | Required target | Important gap |
|---|---|---|---|
| Blocker classifier: TF-IDF → SetFit | Included `../training_data.json`: 108 explicitly synthetic request/reason examples | One reviewed blocker for a confirmed unmet request | No public dataset here supplies the exact six labels with verified retail loss outcomes |
| Product matching: lexical → MiniLM | Included `data/catalog.csv`, `queries.csv`, `pairs.csv`; then Amazon ESCI | Query ↔ exact product judgments; explicit negative pairs | A substitute, accessory or similar size is not necessarily the requested SKU |
| Entity extractor: DistilBERT | Your own annotated, minimized requests; `data/entities_schema_example.jsonl` shows the format only | PRODUCT, QUANTITY, BUDGET, CURRENCY, LOCATION spans/BIO tags | General entity datasets do not annotate these retail fields |
| Recovery: CatBoost | Your own requests joined to verified orders; `data/recovery_template.csv` is a blank schema | Conversion within 14 days, with a fully observed window | An unresolved/closed request does not establish non-conversion |
| Weekly completed sales: LightGBM | UCI Online Retail II for practice; your store's transactions/listing/stock history for a pilot | SKU/week gross positive sales quantity | Sales history does not directly observe demand hidden by stockouts |

## 1. Included practice data

Run `python -m ml_workflow.prepare_data` from the project root to convert the
existing JSON starter into a CSV with provenance and grouping fields. It has
18 examples for each of six reasons. They are synthetic, mostly short English
sentences, with no real capture timestamps or known template families.

The matching exercise includes **8 fictional SKUs, 24 fictional queries and 16
explicitly judged training pairs**. Three query partitions are supplied. The
queries are deliberately simple and related; the exercise verifies the code
path, not a claim of semantic understanding or commercial search quality.

## 2. Bitext retail/e-commerce: language and annotation practice

[Publisher dataset and files](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset)

The publisher describes a hybrid synthetic English dataset of 44,884 examples,
46 intents and 13 categories, with a declared CDLA-Sharing-1.0 license. Use
`instruction`, `intent`, `category` for exploring retail language. Do not use
`response` as a predictor: it is generated after the question and can leak
answer/outcome information.

Download `bitext-retail-ecommerce-llm-chatbot-training-dataset.csv` using the
publisher's **Files** tab. Place it in `ml_workflow/work/downloads/`, then run:

```sh
python -m ml_workflow.prepare_public bitext \
  --csv ml_workflow/work/downloads/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv \
  --out ml_workflow/work/bitext --limit 460
```

This creates an **annotation queue**, not a reason-training dataset. Inspect
`original_intent`; label eligibility as `confirmed_unmet`, `pending`,
`fulfilled` or `not_a_request` only when the context establishes it. Keep
unknowns. Most availability questions cannot establish a loss reason. Do not
force `"Do you have shoes?"` into `not_stocked`. Add actual business context,
have a reviewer assign a blocker, and group paraphrases/templates together.
The adapter intentionally leaves target labels blank.

## 3. Amazon ESCI: product-query relevance

[Amazon Science repository, dataset and usage instructions](https://github.com/amazon-science/esci-data)

ESCI distinguishes **Exact (E), Substitute (S), Complement (C), Irrelevant (I)**.
The repository declares Apache-2.0. Follow its dataset link to obtain:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`

These can be substantially larger than the starter fixtures. Install the tabular
extras in a separate learning environment, then prepare a bounded English sample:

```sh
python -m pip install -r ml_workflow/requirements-tabular.txt
python -m ml_workflow.prepare_public esci \
  --examples ml_workflow/work/downloads/shopping_queries_dataset_examples.parquet \
  --products ml_workflow/work/downloads/shopping_queries_dataset_products.parquet \
  --out ml_workflow/work/esci --limit 500
```

The adapter checks unique product keys and missing join matches, keeps US English
and the small-version judgments, preserves the official test partition, and
derives validation from official train by normalized-query hash. Duplicate query
text cannot cross train/test. It keeps E/I pairs for training an exact-match
candidate; S and C remain different business meanings. It includes only queries
with known exact matches.

`matching.py` evaluates against the sampled catalog. This is a **local experiment,
not the official ESCI benchmark**: unjudged products are not known negatives,
and requests with no exact match need a separate rejection test set. Before
deployment, add your own catalog/query judgments, hard wrong-size/brand/model
negatives, multilingual queries, and genuinely absent products. Keep customer
and paraphrase families together when creating your own query partitions.

## 4. UCI Online Retail II: forecasting practice

[UCI publisher page, workbook and attribution](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The publisher provides 1,067,371 UK retail transaction records from 2009–2011,
including missing values, under CC BY 4.0. Download and extract
`online_retail_II.xlsx` from that page. Its age and geography make it a teaching
dataset, not evidence about the current Kenyan retail market.

```sh
python -m ml_workflow.prepare_sales \
  --xlsx ml_workflow/work/downloads/online_retail_II.xlsx \
  --out ml_workflow/work/uci --max-skus 20 \
  --assume-active-between-first-last
```

Read the generated manifest. The adapter excludes cancellation invoices,
non-positive quantities/prices and partial boundary weeks; the target is
**gross positive unit sales**, not net sales/revenue. It retains repeated lines
because invoice-line identifiers are unavailable. It selects SKUs from the first
12 weeks, avoiding selection by future test-period sales.

The explicit flag acknowledges a practice assumption: zero-fill weeks between
first and last sale as if the SKU were continuously offered. UCI does not prove
active listing or stock availability. In your store, use listing/availability
events; do not silently turn missing records into zero demand.

## 5. Your business dataset: the route to a usable model

Use permitted, minimized text from WhatsApp, email, chatbot, sales notes and
voice **transcripts**, with enough conversation context to establish what
happened. Text datasets do not train speech recognition. Existing browser voice
capture is separate. Evaluate English, Swahili and mixed-language messages
separately; the proposed English checkpoints do not establish Swahili quality.

For each request retain:

- An anonymous request ID and a stable conversation ID.
- A `group_id` covering the same customer/conversation, duplicate case or
  synthetic template family. Use a consistent grouping policy; never reuse one
  template across different evaluation partitions.
- Text available at prediction time, channel, timezone-aware capture timestamp,
  origin, synthetic flag and explicit reviewer verification.
- One reason only for a confirmed unmet request; preserve pending/ambiguous
  cases in a separate annotation queue. Split multi-product cases before labeling.
- For entities: token/span annotations. For matching: candidate SKU + judgment.
- For recovery: capture-time features, earliest verified linked order timestamp,
  observation coverage and a fixed 14-day conversion definition.
- For forecasts: complete dated transactions, active listing periods and stock
  availability. Customer budgets are not orders or recovered revenue.

Start planning around 50–100 reviewed real examples per blocker, then collect
more where errors occur. This is a collection target, not an algorithm minimum
or accuracy guarantee. Keep an independent real test set and a later-period
check. Add first-party text to training only after consent/access, minimization
and label review appropriate to the business.

Optional read-only export of the app's **already reviewed** reason labels:

```sh
python -m ml_workflow.prepare_data \
  --reviewed-db ghostdemand.db --out ml_workflow/work/business
```

The current DB does not store true conversation/template groups. The export uses
`REVIEW-ME-*` placeholders; the splitter refuses them until you replace them.
It does basic email/phone removal, which does not anonymize all names, addresses
or order references. Review the local export before sharing it. No live export
is performed by the starter workflow.
