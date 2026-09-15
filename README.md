# Apple AI Customer Support Agent

An AI-assisted customer-support prototype built for the **Hiver SDE Intern take-home assignment** using the Customer Support on Twitter dataset.

The system:

1. Classifies incoming customer messages into a small Apple-specific intent taxonomy.
2. Retrieves historically similar **@AppleSupport** conversations.
3. Drafts a concise reply grounded in historical Apple support responses.
4. Decides whether to auto-handle or escalate to a human, with a stated reason.
5. Uses **Qwen2.5 3B through Ollama** as a local LLM-as-judge for reply-quality evaluation.

> **Evaluation principle:** classification is evaluated on a fixed 250-example hand-labelled golden set. Reply quality is evaluated separately using a local Qwen2.5 judge and compared against human ratings.

---

## 1. Dataset

**Dataset:** Customer Support on Twitter (TWCS)

**Brand selected:** `@AppleSupport`

The raw TWCS dataset is intentionally not committed because it is large.

If reproducing the data-preparation step from scratch, download the dataset and place it at:

```text
data/raw/twcs.csv
```

The committed processed artifacts include:

- `data/processed/apple_conversations.csv` — 74,649 Apple customer/reply pairs
- `data/processed/golden_set.csv` — 250 hand-labelled evaluation examples
- `data/processed/support_data.csv` — 15,000 Apple-only development rows

The dataset is noisy, multi-turn and imperfect, so the project uses a fixed evaluation set rather than claiming production-level performance from the full corpus.

---

## 2. Intent Taxonomy

The project uses six intents defined by inspecting AppleSupport traffic before finalizing the taxonomy.

| Intent | Meaning |
|---|---|
| `device_software_issue` | iOS, apps, updates, crashes, battery, bugs, or device-performance problems |
| `account_access` | Apple ID, login, password, verification, locked-account issues |
| `payment_billing` | Charges, subscriptions, billing, invoices, or payment problems |
| `refund_request` | Explicit requests for a refund or money back |
| `purchase_order` | Purchases, orders, delivery, shipping, trade-ins, or order status |
| `general_information` | General questions, praise, unclear or other requests |

### Golden-set distribution

The fixed golden set contains 250 examples:

| Intent | Examples |
|---|---:|
| `device_software_issue` | 166 |
| `general_information` | 57 |
| `account_access` | 11 |
| `purchase_order` | 9 |
| `refund_request` | 4 |
| `payment_billing` | 3 |

The strong class imbalance is important when interpreting accuracy. Macro F1 and weighted F1 are therefore reported alongside accuracy.

---

## 3. System Architecture

```text
Customer Message
       |
       v
+------------------------+
| Local Intent Classifier|
| TF-IDF + Naive Bayes   |
+------------------------+
       |
       v
+------------------------+
| Handling Decision      |
| Confidence + Risk     |
+------------------------+
       |
       +-----------------------------+
       |                             |
       v                             v
+------------------+       +------------------------+
| Auto-handle /    |       | Historical Retrieval  |
| Escalate         |       | TF-IDF Similarity     |
+------------------+       +------------------------+
                                      |
                                      v
                           Similar AppleSupport
                           Conversations + Replies
                                      |
                                      v
                              Grounded Reply Draft
```

### 3.1 Customer Message

The incoming customer tweet is the primary input.

### 3.2 Intent Classifier

The message is classified into exactly one of the six Apple support intents.

The runnable agent uses a local **TF-IDF + Multinomial Naive Bayes** classifier trained on the fixed golden-set labels.

No API key or external classification service is required.

### 3.3 Handling Decision

The predicted intent and classifier confidence are passed to a deterministic escalation layer.

Sensitive intents and low-confidence predictions are routed to a human.

### 3.4 Historical Retrieval

The incoming message is compared with historical AppleSupport customer messages using TF-IDF cosine similarity.

The top three conversations are retrieved together with their historical replies.

### 3.5 Draft Reply

For auto-handled cases, the highest-ranked historical AppleSupport reply is used as the draft response.

This keeps the response grounded in observed historical support behaviour instead of generating an unconstrained response.

### 3.6 LLM-as-Judge

Reply quality is evaluated separately using **Qwen2.5 3B running locally through Ollama**.

The judge evaluates:

- Groundedness
- Helpfulness
- Tone
- Conciseness
- Overall quality

The LLM judge does not require Gemini or any external API.

---

## 4. Classification Approach

### Final local classifier

The main runnable classifier uses:

```text
TF-IDF
   +
Multinomial Naive Bayes
```

The model uses:

- maximum 5,000 TF-IDF features
- unigram and bigram features
- English stop-word removal

The classifier returns:

- predicted intent
- confidence score
- short classifier reasoning

### Evaluation split

The golden set contains 250 examples.

Two examples per intent are reserved as fixed examples, giving:

```text
250 total examples
- 12 reserved examples
= 238 evaluation examples
```

The classification evaluation therefore uses **238 held-out examples**.

### Why local ML?

A local classifier makes the main evaluation:

- reproducible
- deterministic
- offline
- inexpensive
- easy to inspect

The project separately demonstrates LLM evaluation through the local Qwen2.5 judge.

---

## 5. Baselines

The project compares the main local classifier against two simple baselines.

### 5.1 Trivial majority baseline

Always predicts:

```text
device_software_issue
```

On the current 238-example evaluation split:

| Metric | Result |
|---|---:|
| Accuracy | **68.91%** |
| Macro F1 | **13.60%** |
| Weighted F1 | **56.22%** |

This demonstrates why accuracy alone is misleading for this highly imbalanced dataset.

### 5.2 Keyword diagnostic baseline

A transparent rule-based classifier checks terms associated with:

- refunds
- payments
- purchases/orders
- account access
- device/software issues

On the current 238-example evaluation split:

| Metric | Result |
|---|---:|
| Accuracy | **66.81%** |
| Macro F1 | **18.58%** |
| Weighted F1 | **56.53%** |

### 5.3 TF-IDF + Multinomial Naive Bayes

The local ML classifier is evaluated using stratified cross-validation.

Current observed evaluation:

| Metric | Result |
|---|---:|
| Accuracy | **68.91%** |
| Macro F1 | **13.60%** |
| Weighted F1 | **56.22%** |

The rarest class contains only three examples, so minority-class estimates are noisy.

---

## 6. Historical Reply Retrieval

For each incoming message:

1. The message is transformed using a TF-IDF vectorizer.
2. The vector is compared against historical AppleSupport customer messages.
3. Cosine similarity is calculated.
4. The top three conversations are retrieved.
5. Their historical AppleSupport replies are returned as grounding evidence.
6. The highest-ranked historical reply is used as the draft response.

This approach uses real historical brand responses instead of treating an LLM as an unconstrained answer generator.

---

## 7. Auto vs Escalate

The handling layer returns both:

```text
decision
decision_reason
```

### Auto-handle

A request can be auto-handled when:

- classifier confidence is at least `0.60`
- the intent is not considered sensitive

### Escalate

The current conservative policy escalates:

- `account_access`
- `payment_billing`
- `refund_request`
- any prediction below the confidence threshold

The reason is included in the agent output.

The goal is to avoid automatically giving potentially consequential account or money-related guidance.

---

## 8. Golden Evaluation Set

A fixed **250-example** golden set is stored at:

```text
data/processed/golden_set.csv
```

Each row contains:

- `tweet_id`
- `text`
- `custom_intent`

The examples were sampled from AppleSupport traffic and manually reviewed using the six-intent taxonomy.

Twelve fixed examples, two per intent, are reserved from the classification evaluation split.

The golden set is deliberately not presented as statistically representative of production traffic. Its purpose is to provide a fixed, auditable evaluation target.

---

## 9. Final Classification Evaluation

Run:

```bash
python src/evaluate.py
```

The script writes:

```text
data/processed/evaluation_results.csv
data/processed/evaluation_summary.json
```

The evaluation compares:

1. Majority baseline
2. Keyword diagnostic baseline
3. TF-IDF + Multinomial Naive Bayes

The main metrics are:

- Accuracy
- Macro F1
- Weighted F1

### Current headline result

On the 238-example held-out split:

```text
Accuracy:    68.91%
Macro F1:    13.60%
Weighted F1: 56.22%
```

The headline accuracy should not be interpreted as production support quality.

---

## 10. Reply Quality Evaluation

Reply quality is evaluated separately from intent classification.

### Step 1 — Generate reply sample

Run:

```bash
python src/generate_reply_sample.py
```

This creates a maximum 20-example sample from the golden set.

Only examples that receive a drafted reply are retained for judging.

The current completed human-labelled run contains **17 valid reply examples**.

### Step 2 — Human ratings

Run:

```bash
python src/label_replies.py
```

Each reply is rated from 1–5 on:

- Groundedness
- Helpfulness
- Tone
- Conciseness
- Overall

The human ratings are stored in:

```text
data/processed/reply_judge_sample.csv
```

### Step 3 — Qwen2.5 LLM judge

The project uses:

```text
Model:    qwen2.5:3b
Runtime:  Ollama
Endpoint: localhost:11434
```

Make sure Ollama is installed and the model is available:

```bash
ollama pull qwen2.5:3b
```

Then run:

```bash
python src/evaluate_judge.py
```

Outputs:

```text
data/processed/judge_results.csv
data/processed/judge_agreement.json
```

### Judge agreement results

The completed 17-example evaluation produced:

| Criterion | Human Mean | Judge Mean | MAE | Exact Agreement | Within 1 Point |
|---|---:|---:|---:|---:|---:|
| Groundedness | 4.00 | 4.41 | 0.65 | 35.29% | 100.00% |
| Helpfulness | 2.94 | 3.06 | 0.47 | 58.82% | 94.12% |
| Tone | 4.00 | 4.35 | 0.35 | 64.71% | 100.00% |
| Conciseness | 5.00 | 3.29 | 1.71 | 0.00% | 29.41% |
| Overall | 3.00 | 3.35 | 0.35 | 64.71% | 100.00% |

Pearson correlation is reported only when both human and judge scores have non-zero variance. Some criteria have constant human ratings in this small sample, so correlation is not defined.

These results show why the LLM judge should not be treated as automatically reliable without human validation.

---

## 11. Reply Diagnostics

Additional deterministic diagnostics are available through:

```bash
python src/reply_metrics.py
```

The script measures:

- reply length
- message/reply word overlap
- presence of actionable language

Output:

```text
data/processed/reply_metrics.csv
```

---

## 12. Example End-to-End Run

Run:

```bash
python src/agent.py "My iPhone won't update to the latest iOS, it just keeps failing"
```

Typical output fields are:

```text
message
intent
confidence
classifier_reasoning
decision
decision_reason
draft_reply
grounding_examples
```

The agent runs completely locally.

No API key is required.

---

## 13. Top Five Failure Patterns

The main failure patterns identified for this prototype are:

### 1. Device/software vs general information

Short or vague messages can be lexically ambiguous.

**Hypothesis:** TF-IDF relies heavily on surface-level terms and can miss the implied intent.

### 2. Order/purchase vs general information

Customers may ask about an order without explicitly using words such as "order", "delivery", or "purchase".

**Hypothesis:** keyword and TF-IDF methods can miss implicit purchase context.

### 3. Payment vs general information

Payment problems may be described indirectly.

**Hypothesis:** indirect billing language can be difficult for a lexical classifier.

### 4. Account access vs general information

Customers may mention a symptom without explicitly saying that they are locked out of an account.

**Hypothesis:** the classifier lacks broader account-state context.

### 5. Context-dependent or multilingual messages

A single tweet may not contain enough information to determine the complete issue.

**Hypothesis:** thread context, spelling variation and multilingual text reduce lexical similarity.

For a future analysis, the most useful failure examples should be selected directly from `evaluation_results.csv` rather than inventing examples.

---

## 14. What Is Misleading About My Headline Number?

The headline accuracy is **68.91%**, but it is not a production estimate of support quality.

First, the golden set contains only 250 examples and is highly imbalanced. The dominant `device_software_issue` class contains 166 examples, while `payment_billing` contains only three.

Second, the current majority baseline itself reaches **68.91% accuracy** on the held-out evaluation split.

Therefore, accuracy alone makes the system appear stronger than it is.

The **13.60% macro F1** shows that performance across the six intents is much weaker, especially for minority classes.

Third, intent classification is only one component of the support agent. A useful system also needs:

- safe handling decisions
- historically grounded replies
- appropriate escalation
- helpfulness
- tone
- concise responses

Fourth, historical Twitter support data is noisy and context-dependent. A single tweet does not always contain enough information to infer the customer's complete issue.

Therefore, the headline result should always be read together with macro F1, baseline comparisons, handling behaviour and reply-quality results.

---

## 15. Limitations

- The golden set is small and highly imbalanced.
- Several minority intents have very few examples.
- TWCS conversations can lose context when evaluated tweet-by-tweet.
- TF-IDF retrieval measures lexical similarity rather than semantic similarity.
- The prototype does not access live Apple account, order or billing systems.
- Historical replies represent past behaviour and are not guaranteed to reflect current policy.
- The local classifier confidence is not calibrated as a production probability.
- Reply-quality evaluation currently uses only 17 human-labelled examples.
- Qwen judge results can vary with model/runtime behaviour.
- Automatic handling remains conservative for sensitive cases.

---

## 16. Next-Week Improvement Plan

With one more week I would:

1. Expand the golden set with targeted minority-class examples.
2. Add a second independent human annotator and measure annotation agreement.
3. Review ambiguous examples and refine the taxonomy where necessary.
4. Replace TF-IDF retrieval with embedding-based semantic retrieval.
5. Evaluate a stronger classifier against the same fixed test set.
6. Calibrate confidence and tune the auto/escalate threshold on a validation split.
7. Preserve conversation/thread context where available.
8. Improve multilingual handling.
9. Increase the reply-judge sample size.
10. Repeat the human-vs-judge agreement analysis after judge prompt changes.

---

## 17. Decision Log

### Decision 1 — Selected AppleSupport

AppleSupport was selected as the single brand so the classifier and reply behaviour could be specialized to one support domain.

### Decision 2 — Six-intent taxonomy

A compact taxonomy was chosen to avoid creating many poorly represented classes.

### Decision 3 — Fixed 250-example golden set

250 examples are within the assignment's required 150–250 range and provide a fixed evaluation target.

### Decision 4 — Hand review of labels

Labels are manually reviewed rather than treating automatically generated keyword labels as ground truth.

### Decision 5 — Separate intent and handling

Intent describes the customer's problem; handling describes operational risk. Keeping them separate makes escalation policy inspectable.

### Decision 6 — Twelve held-out examples

Two examples per intent are reserved from the classification evaluation to keep a fixed evaluation split.

### Decision 7 — TF-IDF retrieval

TF-IDF was selected because it is deterministic, inexpensive and easy to inspect without an external vector database.

### Decision 8 — Top-three retrieval

Three examples provide useful historical context without overloading the pipeline.

### Decision 9 — Conservative sensitive-intent escalation

Account, payment and refund requests can involve information or actions unavailable to the prototype, so they are routed to humans.

### Decision 10 — Confidence threshold of 0.60

Low-confidence predictions are escalated rather than automatically answered.

### Decision 11 — Trivial majority baseline

The majority baseline exposes the effect of the highly imbalanced golden set.

### Decision 12 — TF-IDF + Naive Bayes baseline

A conventional lightweight ML model provides a stronger reference than the trivial baseline while remaining fully reproducible offline.

### Decision 13 — Separate reply-quality evaluation

Intent accuracy cannot tell whether a response is grounded, helpful, concise and appropriate.

### Decision 14 — Human validation of the LLM judge

The Qwen judge is compared with human ratings so its scores are not treated as automatically reliable.

### Decision 15 — Local Qwen2.5 judge

Qwen2.5 3B through Ollama was selected so the LLM-as-judge evaluation could run locally without an external API key.

---

## 18. Installation

### Requirements

- Python 3.7.6 or compatible Python 3.7 environment
- Ollama for the LLM-as-judge evaluation
- pandas
- NumPy
- scikit-learn
- SciPy
- pytest

Install the Python dependencies:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

### Ollama

Install Ollama separately.

Pull the Qwen model:

```bash
ollama pull qwen2.5:3b
```

Verify it:

```bash
ollama run qwen2.5:3b
```

The main agent and classification evaluation do not require Ollama.

Ollama is required only for the LLM-as-judge evaluation.

**No Gemini API key or other external API key is required.**

---

## 19. Running the Agent

```bash
python src/agent.py "My iPhone won't update to the latest iOS"
```

The agent prints:

- intent
- confidence
- classifier reasoning
- handling decision
- decision reason
- draft reply
- historical grounding examples

---

## 20. Running Evaluation

### Main classification evaluation

```bash
python src/evaluate.py
```

### Simple baseline

```bash
python src/simple_baseline.py
```

### Reply sample

```bash
python src/generate_reply_sample.py
```

### Human reply labels

```bash
python src/label_replies.py
```

### Qwen LLM judge + agreement

Make sure Ollama is running first:

```bash
python src/evaluate_judge.py
```

### Reply diagnostics

```bash
python src/reply_metrics.py
```

### Tests

```bash
python -m pytest -q
```

---

## 21. Project Structure

```text
hiver support agent/
│
├── data/
│   ├── raw/
│   │   └── twcs.csv
│   │
│   └── processed/
│       ├── apple_conversations.csv
│       ├── golden_set.csv
│       ├── support_data.csv
│       ├── evaluation_results.csv
│       ├── evaluation_summary.json
│       ├── reply_judge_sample.csv
│       ├── judge_results.csv
│       ├── judge_agreement.json
│       └── reply_metrics.csv
│
├── src/
│   ├── prepare_data.py
│   ├── agent.py
│   ├── simple_baseline.py
│   ├── evaluate.py
│   ├── judge.py
│   ├── generate_reply_sample.py
│   ├── label_replies.py
│   ├── evaluate_judge.py
│   └── reply_metrics.py
│
├── tests/
│   ├── test_agent.py
│   └── test_judge.py
│
├── README.md
├── REPORT.md
├── DECISIONS.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 22. Reproducibility

The headline classification evaluation is based on a fixed 250-example golden set.

The main evaluation command is:

```bash
python src/evaluate.py
```

Results are saved to:

```text
data/processed/evaluation_results.csv
data/processed/evaluation_summary.json
```

The raw TWCS file is not required to rerun the main evaluation because the processed Apple conversations and golden set are included.

The raw dataset is only needed if the data-preparation step needs to be reproduced from scratch.

The classification evaluation runs locally without an API.

The reply-quality judge requires Ollama and the local Qwen2.5 3B model.

---

## 23. Conclusion

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

The main headline accuracy is reported together with macro F1 and baseline comparisons because the fixed golden set is strongly imbalanced.

The current prototype is intentionally conservative and fully runnable locally. The main improvement areas are semantic retrieval, minority-intent coverage, confidence calibration, thread context, multilingual handling and stronger human-validated reply evaluation.
