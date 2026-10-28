# PesaShield AI — Synthetic mobile money SMS dataset

## Contents
- 1,200 **fully synthetic** English-language SMS examples for a **demo / baseline** model.
- 600 ordinary-looking notification examples (`label=0`), 600 suspicious/scam-pattern examples (`label=1`).
- Eight categories, 150 messages each.
- Columns: `message_id`, `message_text`, `label`, `category`, `template_group`, `data_origin`, `payment_verified`.

## Interpretation
- `label=0` means an SMS **does not contain the suspicious patterns in this synthetic dataset**. It does NOT certify the payment as real, the sender as genuine, or the message as safe.
- `label=1` means the message **contains a synthetic fraud / scam warning pattern**.
- `payment_verified=not_checked` for ALL rows. The dataset contains no real account or provider verification.
- `*.example.invalid` URLs are deliberately unusable placeholders. Reference IDs prefixed `DEMO-` are fictional.

## Important limitations
This dataset is generated from a relatively small number of hand-written templates and randomly substituted values. **It is NOT real M-Pesa customer data or a validated real-world benchmark.** Models trained on it may memorize templates, overfit to words such as 'demo' or 'example', and perform poorly on genuine M-Pesa scams. Ordinary-looking messages have explicit demo language and artificial references. Do not report resulting model accuracy as real-world fraud accuracy or deploy it to make financial decisions.

For meaningful evaluation, collect opt-in, anonymized and independently labeled real messages; distinguish verified legitimate payment records from SMS text appearance; verify transaction status using merchant/provider-authorized systems; and test on unseen campaigns, dates, and users. Beware that even group-held-out templates here are artificial and can leak category-specific wording.

## First run
```python
import pandas as pd
sms = pd.read_csv('pesashield_synthetic_mpesa_sms.csv')
print(sms.shape)
print(sms['category'].value_counts())
print(sms[['message_text', 'label', 'category']].head())
```

## Suggested workflow
1. EDA: class counts and message lengths.
2. TF-IDF + LogisticRegression baseline using `message_text` -> `label`.
3. Group the split by `template_group` with `GroupShuffleSplit` (still only a synthetic demonstration).
4. Report classification report, confusion matrix, precision/recall; manually inspect false positives.
5. Supplement with genuine opt-in/consented labeled examples before any real use.

## Repository hygiene
Do not commit personally identifiable SMS or real customer transaction details. Keep raw datasets out of Git unless permission/license/privacy requirements are satisfied.
