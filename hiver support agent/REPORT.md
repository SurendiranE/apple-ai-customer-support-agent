# Apple AI Customer Support Agent — Evaluation Report

## Problem Framing

Build a support agent specialized to `@AppleSupport` that:

1. Classifies incoming customer messages.
2. Drafts replies grounded in historical AppleSupport behaviour.
3. Decides whether to auto-handle or escalate.
4. Provides an explicit reason for the handling decision.

The prototype intentionally does not access live Apple account, billing, order, or device systems.

The system is fully runnable locally and does not require Gemini or any external API key.

---

## Evaluation Design

A fixed **250-example hand-labelled golden set** is used.

The golden set contains six intents:

- `device_software_issue`
- `account_access`
- `payment_billing`
- `refund_request`
- `purchase_order`
- `general_information`

Two examples per intent are reserved as fixed examples, giving:

```text
250 total examples
- 12 reserved examples
= 238 evaluation examples
```

The golden set is highly imbalanced:

| Intent | Examples |
|---|---:|
| `device_software_issue` | 166 |
| `general_information` | 57 |
| `account_access` | 11 |
| `purchase_order` | 9 |
| `refund_request` | 4 |
| `payment_billing` | 3 |

Because of this imbalance, accuracy is reported alongside:

- Macro F1
- Weighted F1

Accuracy alone is not considered sufficient evidence of good intent classification.

---

## Classification System

The final runnable classifier is:

```text
TF-IDF + Multinomial Naive Bayes
```

The classifier uses:

- maximum 5,000 TF-IDF features
- unigram and bigram features
- English stop-word removal

The classifier returns:

- predicted intent
- confidence score
- classifier reasoning

No external API is required.

---

## Baselines

The project compares the local classifier against two reproducible baselines.

### 1. Trivial Majority Baseline

The majority baseline always predicts:

```text
device_software_issue
```

On the current 238-example evaluation split:

| Metric | Result |
|---|---:|
| Accuracy | **68.91%** |
| Macro F1 | **13.60%** |
| Weighted F1 | **56.22%** |

This baseline demonstrates how a highly imbalanced dataset can produce a deceptively high accuracy.

### 2. Keyword Diagnostic Baseline

A transparent rule-based classifier checks lexical indicators associated with:

- refunds
- payments
- purchases and orders
- account access
- device and software issues

On the current 238-example evaluation split:

| Metric | Result |
|---|---:|
| Accuracy | **66.81%** |
| Macro F1 | **18.58%** |
| Weighted F1 | **56.53%** |

### 3. TF-IDF + Multinomial Naive Bayes

The final local classifier uses TF-IDF features with Multinomial Naive Bayes.

Current observed evaluation:

| Metric | Result |
|---|---:|
| Accuracy | **68.91%** |
| Macro F1 | **13.60%** |
| Weighted F1 | **56.22%** |

The minority classes have very few examples, so their individual performance estimates are noisy.

---

## Historical Reply Grounding

The reply component retrieves historically similar AppleSupport conversations.

For each incoming message:

1. The customer message is converted into a TF-IDF vector.
2. It is compared against historical AppleSupport customer messages.
3. Cosine similarity is calculated.
4. The top three similar conversations are retrieved.
5. Their historical AppleSupport replies are returned as grounding evidence.
6. The highest-ranked historical reply is used as the draft response.

This approach keeps the draft grounded in observed historical support behaviour.

The prototype does not claim that the retrieved historical response is always appropriate for current Apple policy.

---

## Auto-Handle vs Escalate

The handling layer is deterministic and separate from intent classification.

### Auto-handle

A message can be automatically handled when:

- classifier confidence is at least `0.60`
- the predicted intent is not considered sensitive

### Escalate

The current conservative policy escalates:

- `account_access`
- `payment_billing`
- `refund_request`
- any prediction below the `0.60` confidence threshold

The agent returns both:

```text
decision
decision_reason
```

This makes the operational policy inspectable.

Sensitive cases are escalated because the prototype does not have access to live account, payment, billing, refund, or order systems.

---

## Headline Classification Result

The current headline classification result on the 238-example evaluation split is:

```text
Accuracy:    68.91%
Macro F1:    13.60%
Weighted F1: 56.22%
```

Run the evaluation with:

```bash
python src/evaluate.py
```

The results are written to:

```text
data/processed/evaluation_results.csv
data/processed/evaluation_summary.json
```

---

## Reply Quality Evaluation

Intent classification and reply quality are evaluated separately.

### Sample Generation

Run:

```bash
python src/generate_reply_sample.py
```

This samples up to 20 examples from the golden set.

Messages that are escalated and therefore do not receive a draft reply are excluded from the reply-quality sample.

The current completed evaluation contains:

```text
17 valid reply examples
```

### Human Evaluation

Run:

```bash
python src/label_replies.py
```

Each reply is rated from 1–5 on:

- Groundedness
- Helpfulness
- Tone
- Conciseness
- Overall quality

The ratings are stored in:

```text
data/processed/reply_judge_sample.csv
```

### Local LLM Judge

The project uses:

```text
Model:    qwen2.5:3b
Runtime:  Ollama
Endpoint: localhost:11434
```

No Gemini API or other external API is required.

Make sure the model is available:

```bash
ollama pull qwen2.5:3b
```

Then run:

```bash
python src/evaluate_judge.py
```

The evaluation produces:

```text
data/processed/judge_results.csv
data/processed/judge_agreement.json
```

---

## Human vs LLM-Judge Agreement

The completed 17-example evaluation produced the following results:

| Criterion | Human Mean | Judge Mean | MAE | Exact Agreement | Within 1 Point |
|---|---:|---:|---:|---:|---:|
| Groundedness | 4.00 | 4.41 | 0.65 | 35.29% | 100.00% |
| Helpfulness | 2.94 | 3.06 | 0.47 | 58.82% | 94.12% |
| Tone | 4.00 | 4.35 | 0.35 | 64.71% | 100.00% |
| Conciseness | 5.00 | 3.29 | 1.71 | 0.00% | 29.41% |
| Overall | 3.00 | 3.35 | 0.35 | 64.71% | 100.00% |

Pearson correlation is reported only when both human and judge scores have non-zero variance.

Some criteria have constant human ratings in this small sample, so Pearson correlation is not defined for those criteria.

The agreement results show that the local LLM judge is useful as an evaluation aid but should not be treated as a substitute for human evaluation.

---

## Reply Diagnostics

Additional deterministic reply diagnostics can be generated with:

```bash
python src/reply_metrics.py
```

The diagnostics include:

- reply length
- message/reply word overlap
- presence of actionable language

Output:

```text
data/processed/reply_metrics.csv
```

---

## Failure Analysis

The current prototype has five important failure patterns.

### 1. Device/software vs General Information

Short or vague messages can be difficult to classify.

**Hypothesis:** TF-IDF relies strongly on lexical overlap and may not capture implied intent.

### 2. Purchase/Order vs General Information

Customers may discuss an order without explicitly using terms such as "order", "delivery", or "purchase".

**Hypothesis:** lexical classifiers can miss implicit purchase context.

### 3. Payment vs General Information

Some billing issues are described indirectly.

**Hypothesis:** indirect payment language is difficult for keyword and TF-IDF methods.

### 4. Account Access vs General Information

A customer may describe a login symptom without explicitly mentioning an Apple ID or account.

**Hypothesis:** the classifier lacks broader account-state context.

### 5. Context-Dependent or Multilingual Messages

A single tweet may not contain enough information to understand the complete customer problem.

Spelling variation, multilingual messages and missing thread context can also reduce lexical similarity.

**Hypothesis:** the system needs conversation-level context and stronger semantic representations.

For a future iteration, failure examples should be selected directly from:

```text
data/processed/evaluation_results.csv
```

rather than manually inventing examples.

---

## What Is Misleading About My Headline Number?

The headline accuracy is:

```text
68.91%
```

However, this number is **not a production estimate of support quality**.

The golden set is highly imbalanced. The largest class contains:

```text
device_software_issue = 166 / 250
```

while:

```text
payment_billing = 3 / 250
```

The trivial majority baseline already achieves **68.91% accuracy** on the same 238-example evaluation split.

Therefore, accuracy alone makes the classifier appear stronger than it actually is.

The macro F1 of:

```text
13.60%
```

shows that performance across all six intents is much weaker.

This is especially important for the minority classes.

Classification is also only one part of the support agent.

A useful support system must additionally consider:

- reply grounding
- helpfulness
- tone
- conciseness
- escalation safety
- operational context

Therefore, the headline accuracy should always be interpreted together with macro F1, weighted F1, baseline comparisons, handling behaviour and reply-quality evaluation.

---

## Limitations

1. The golden set is small and highly imbalanced.
2. Several minority intents have very few examples.
3. TWCS conversations are noisy and can lose context when evaluated tweet-by-tweet.
4. TF-IDF retrieval measures lexical similarity rather than semantic similarity.
5. The prototype does not access live Apple account, order, billing or device systems.
6. Historical replies may not reflect current Apple policies.
7. Classifier confidence is not calibrated as a production probability.
8. Reply-quality evaluation currently uses only 17 human-labelled examples.
9. LLM-judge results can vary with local model behaviour.
10. Automatic handling is intentionally conservative for sensitive intents.
11. The prototype currently uses the highest-ranked historical reply rather than generating a completely new response.
12. Multilingual and context-dependent requests require stronger semantic and conversational modelling.

---

## Reproducibility

The fixed golden set and processed AppleSupport conversations are included in the project.

The main classification evaluation can be reproduced with:

```bash
python src/evaluate.py
```

The classification evaluation runs locally and does not require an API key.

For reply-quality evaluation, install Ollama and pull the local model:

```bash
ollama pull qwen2.5:3b
```

Then run:

```bash
python src/evaluate_judge.py
```

The main commands are:

```bash
python src/evaluate.py
python src/generate_reply_sample.py
python src/label_replies.py
python src/evaluate_judge.py
python src/reply_metrics.py
python -m pytest -q
```

---

## One-Week Improvement Plan

With one more week, I would:

1. Expand minority-class annotation coverage.
2. Add a second independent annotator.
3. Measure inter-annotator agreement.
4. Review ambiguous examples and refine the taxonomy if necessary.
5. Replace TF-IDF retrieval with embedding-based semantic retrieval.
6. Evaluate a stronger classifier on the same fixed test set.
7. Calibrate classifier confidence.
8. Tune the auto-handle/escalation threshold on a validation split.
9. Preserve conversation/thread context where available.
10. Improve multilingual handling.
11. Increase the human-labelled reply-quality sample.
12. Re-evaluate the local LLM judge after prompt improvements.

---

## Conclusion

This project implements an end-to-end Apple customer-support prototype:

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Handling Decision
       |
       +----------------------+
       |                      |
       v                      v
Auto-handle              Escalate
       |
       v
Historical AppleSupport Retrieval
       |
       v
Grounded Reply Draft
       |
       v
Local Qwen2.5 LLM Judge
       |
       v
Human Agreement Analysis
```

The project deliberately separates:

- **classification performance**
- **operational handling**
- **historical grounding**
- **reply quality**
- **LLM-judge reliability**

The headline accuracy is reported together with macro F1 and baseline comparisons because the golden set is strongly imbalanced.

The prototype is fully local, conservative for sensitive requests, and designed to be reproducible without Gemini or other external APIs.
