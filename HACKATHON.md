# GhostDemand AI — demand beyond the sale

**One-line pitch:** GhostDemand turns the requests retailers couldn't fulfil into evidence for what to stock, change, and follow up on next.

## Refined project description

GhostDemand AI is a demand intelligence assistant for retailers and e-commerce
businesses. It captures unsuccessful customer requests from conversations, failed
website searches, sales notes, unaccepted quotes, unavailable products, stockouts
and abandoned enquiries. NLP extracts the requested product, quantity, stated
budget, urgency and blocker, while a local machine learning model learns blocker
classification from verified corrections. Product aliases bring customer wording
together without silently merging different variants. An opportunity board shows
the underlying requests, explains suggested actions, and helps teams assign
follow-ups and record confirmed recoveries. The goal is to make demand visible
before it becomes a sale, while keeping evidence, model uncertainty and actual
outcomes clear.

## The competitive landscape

Research checked on 27 October 2026. These are documented product capabilities,
not an exhaustive feature audit. The opportunities below are our interpretation
of how to package a focused retail workflow; they are not claims that no vendor
could offer the same capabilities.

| Existing capability | Evidence | GhostDemand's focused opportunity |
| --- | --- | --- |
| Support platforms classify intent, sentiment, language and entities and support routing/reporting. | [Zendesk intelligent triage](https://support.zendesk.com/hc/en-us/articles/4964463770650-About-intelligent-triage) | Put the requested product, commercial blocker and next retail action at the center of the workflow. NLP classification alone is not our novelty. |
| Search analytics expose searches that return no results, including the filters involved. | [Algolia searches without results](https://support.algolia.com/hc/en-us/articles/13079831222033-Where-can-I-see-searches-without-results) | Combine zero-result evidence with conversations and sales notes; distinguish a missing product from a naming problem. |
| Inventory tools already account for stockouts when forecasting and report stockout-related lost sales. | [Inventory Planner stockouts](https://help.inventory-planner.com/en/articles/2242706-stockouts) | Capture stated requests for products not yet offered and blockers beyond stock availability. Do not claim that inventory tools ignore stockouts. |
| CRMs track deal stages and closed-lost reasons. | [HubSpot default deal properties](https://knowledge.hubspot.com/properties/hubspots-default-deal-properties) | Make small, pre-deal enquiries worth capturing without first requiring a formal deal. Connect the evidence to a product-level decision. |

**Positioning:** an accessible demand-to-action workflow for a small retail team,
complementing its store, CRM and messaging tools. Cross-channel integration is a
product direction, not proof of a unique moat. Our hackathon advantage is a clear,
reviewable demonstration of the complete loop.

## Gaps addressed in this prototype

| Gap in the original prototype | Implemented improvement | Why it matters |
| --- | --- | --- |
| A ranked list did not show what to do next. | Evidence-backed opportunity cards, explicit rule-based suggestions, urgency and recent/prior seven-day counts. | A retailer can inspect the supporting messages before acting. |
| Model output went directly into storage. | Capture → editable preview → save, full signal review and explicit verified training labels. | People can correct product names, quantities, currency and blockers before relying on them. |
| Different product wording fragmented demand. | Human-managed canonical names and aliases, retaining sizes/specifications and warning on multiple matches. | "Bluetooth earbuds" can map to a store's product name without guessing that different variants are identical. |
| Provider retries could inflate demand. | Per-channel event IDs, Twilio MessageSid handling and stable IDs for CSV reimports. | A delivery retry is counted once. Different genuine requests can still be counted. |
| Anonymous requests were presented as buyers. | Known customer references counted separately, with unidentified request counts shown in evidence. | A search is not automatically assumed to be a unique buyer. |
| Mixed-currency budgets were added as a value of lost sales. | Currency-separated stated budgets and separately recorded recovery outcomes. | Interest is not revenue, and KES and USD are not interchangeable. |
| The interface assumed technical users and had no presentation path. | Retail dashboard, plain-language actions, guided capture, filters, paginated inbox, CSV preview/export and an isolated demo workspace. | A judge or shop owner can understand the workflow without an API tutorial. |
| No follow-through after an insight. | Owner, next action, open/follow-up/recovered/closed status and optional recorded recovered amount. | The team can show whether a decision led to a recorded recovery. |

## Three-minute demo

For the panel, open **`/present`** for the guided six-chapter demo with live
capture, evidence, follow-up, the ML explanation, keyboard navigation and an
optional rehearsal timer. [Presenter script and Q&A](PANEL_PRESENTATION.md)
explains the serving model and the separate notebook evaluation. The workspace
walkthrough below remains useful for deeper questions.

1. **0:00–0:30 — The problem.** "Sales reports show what people bought. Our next
   product decision also needs the requests we couldn't fulfil." Open `/demo` and
   click **Load retail demo**. State that all 24 requests and sample outcomes are
   synthetic.
2. **0:30–1:10 — Evidence to action.** Open the Wireless Earbuds opportunity. Show
   requests across channels, known customer references, blockers, the explicit
   action suggestion and the underlying messages. These are observations, not a
   sales forecast.
3. **1:10–1:50 — Capture a new signal.** Choose **Capture request**, use the
   sold-out-earbuds example, preview it, check the details and tick the verified
   label box. Save it. Voice is optional and browser-dependent; keep text capture
   ready for the stage demo.
4. **1:50–2:25 — Close the loop.** Open a request, assign an owner and next action.
   If demonstrating recovery, explicitly describe the amount as a simulated,
   team-recorded outcome. No customer messages are sent.
5. **2:25–3:00 — Work queue & honest scope.** Open **Follow-up queue**. Show who
   owns the next action, when it is due, and how overdue work or recovered
   requests can be filtered. Model training and evaluation are backend operations;
   the retail team uses request review and follow-up tools. Close with the pilot question:
   "Can this help a retailer discover an actionable product gap and recover a
   request it would otherwise have forgotten?"

## Working now versus next

**Working locally:** text capture across labeled channels, browser dictation where
supported, editable NLP previews, local blocker classification and retraining,
product aliases, CSV import/export, zero-result and stockout APIs, a Twilio-style
form webhook, opportunity evidence, filters, review and recovery tracking. The
demo has its own SQLite database and model; loading or training demo data never
writes to the live workspace.

**Requires configuration:** actual website or WhatsApp traffic, provider-specific
account credentials and event forwarding. The UI labels receivers as available,
not connected. Existing API access does not imply vendor account access.

**Before a public pilot:** authentication and tenant isolation, webhook signature
validation, consent and data retention controls, representative labeled retail
evaluation, language coverage testing, and outcome verification against real
orders. Browser voice can use a remote recognition service and is not available
in every browser; see [MDN SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition).

**Deliberately outside this prototype:** automatic outreach, supplier ordering,
profitability recommendations, a validated sales forecasting model, and claims
that every enquiry represents incremental or recoverable revenue.

## Pilot questions and measures

Measure extraction agreement on a separately labeled retail test set; duplicate
handling on known event retries; the fraction of opportunities with evidence and
an assigned next step; and the number/value of team-confirmed recoveries. A later
pilot should compare follow-up outcomes against an agreed baseline. Do not claim
causal revenue lift from the hackathon's synthetic scenario. No pilot results or
customer validation are claimed here.
