# Apple AI Customer Support Agent

An AI-assisted customer-support prototype built for the **Hiver SDE Intern take-home assignment** using the Customer Support on Twitter dataset.

The system:
1. Classifies incoming customer messages into a small Apple-specific intent taxonomy.
2. Retrieves historically similar **@AppleSupport** conversations.
3. Drafts a concise reply grounded in historical Apple support responses.
4. Decides whether to auto-handle or escalate to a human, with a stated reason.

> **Evaluation principle:** the system is judged on a fixed hand-labelled golden set, against a trivial baseline and a conventional ML baseline. Reply quality is evaluated separately with an LLM judge and a human-agreement check.

---

## 1. Dataset

**Dataset:** Customer Support on Twitter (Kaggle, thoughtvector/customer-support-on-twitter)

**Brand selected:** `@AppleSupport`

The raw TWCS dataset is intentionally not committed because it is large. Download it from Kaggle and place it at:

```text
data/raw/twcs.csv
```

The committed processed artifacts are small enough to support reproducible evaluation:

- `data/processed/apple_conversations.csv` — 74,649 Apple customer/reply pairs
- `data/processed/golden_set.csv` — 250 hand-labelled evaluation examples
- `data/processed/support_data.csv` — 15,000 Apple-only development rows

The dataset is noisy, multi-turn and imperfect, so the project uses a fixed evaluation set rather than claiming production-level performance from the full corpus.

---

## 2. Intent Taxonomy

The project uses six intents defined by inspecting AppleSupport traffic before finalizing the taxonomy:

| Intent | Meaning |
|---|---|
| `device_software_issue` | iOS, apps, updates, crashes, battery, bugs, or device-performance problems |
| `account_access` | Apple ID, login, password, verification, locked-account issues |
| `payment_billing` | Charges, subscriptions, billing, invoices, or payment problems |
| `refund_request` | Explicit requests for a refund or money back |
| `purchase_order` | Purchases, orders, delivery, shipping, trade-ins, or order status |
| `general_information` | General questions, praise, unclear or other requests |

### Golden-set distribution

The 250 examples are intentionally kept fixed for evaluation. Their observed distribution is:

| Intent | Examples |
|---|---:|
| `device_software_issue` | 166 |
| `general_information` | 57 |
| `account_access` | 11 |
| `purchase_order` | 9 |
| `refund_request` | 4 |
| `payment_billing` | 3 |

This imbalance is important when interpreting accuracy; macro F1 and per-intent results are reported as complementary metrics.

---

## 3. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
       |
       +-----------------------+
       |                       |
       v                       v
Handling Decision       Historical Retrieval
       |                       |
       |                       v
       |                 Similar AppleSupport
       |                 Conversations
       |                       |
       +-----------+-----------+
                   |
                   v
              Draft Reply
```

### 3.1 Customer Message

The incoming customer tweet is the primary input.

### 3.2 Intent Classifier

The message is classified into exactly one of the six Apple support intents. With a Gemini API key, the final classifier uses few-shot prompting. Without an API key, the runnable agent falls back to a local TF-IDF + Multinomial Naive Bayes classifier so the repository remains executable offline.

### 3.3 Handling Decision

The classifier's intent and confidence are passed to a deterministic escalation layer. Sensitive intents and low-confidence predictions are routed to a human.

### 3.4 Historical Retrieval

The message is compared with historical AppleSupport customer messages using TF-IDF cosine similarity. The top three conversations are retrieved together with their historical replies.

### 3.5 Draft Reply

For auto-handled cases, Gemini receives the customer message, predicted intent and historical examples and produces a concise draft. The prompt explicitly asks it to stay grounded in the retrieved examples and under 280 characters.

---

## 4. Classification Approach

### Final classifier

The API-enabled agent uses a Gemini few-shot classifier. The prompt contains:

- the six intent definitions;
- two fixed examples per intent from the golden set;
- an instruction to return one valid intent;
- a confidence score and short reasoning.

Those 12 few-shot examples are excluded from the classification evaluation, leaving **238 held-out golden examples** when the full set is evaluated.

### Simple local baseline

A conventional TF-IDF + Multinomial Naive Bayes classifier is also implemented. It is evaluated with stratified 3-fold cross-validation because the rarest golden-set class contains only three examples.

### Trivial baseline

The trivial baseline always predicts the majority intent, `device_software_issue`. On this golden set that gives:

- Accuracy: **66.4%**
- Macro F1: **11.1%**

This is intentionally weak and demonstrates why accuracy alone can be misleading on this dataset.

---

## 5. Historical Reply Retrieval

For each incoming message:

1. Clean the text.
2. Transform it using a TF-IDF vectorizer fitted on Apple historical customer messages.
3. Compute cosine similarity against the historical message matrix.
4. Select the top three conversations.
5. Retrieve their corresponding AppleSupport replies.
6. Pass those examples to the reply drafter as grounding context.

This approach uses real historical brand responses instead of treating the LLM as an unconstrained answer generator.

---

## 6. Auto vs Escalate

The handling layer returns both a decision and a reason.

### Auto-handle

A request can be auto-handled when:

- classifier confidence is at least `0.60`; and
- the intent is not considered sensitive.

### Escalate

The current conservative policy escalates:

- `account_access`
- `payment_billing`
- `refund_request`
- any prediction below the confidence threshold

The reason is included in the agent output. The goal is to avoid automatically giving potentially consequential account or money-related guidance.

---

## 7. Golden Evaluation Set

A fixed **250-example** golden set is stored in:

```text
data/processed/golden_set.csv
```

Each row contains:

- `tweet_id`
- `text`
- `custom_intent`

The examples were sampled from AppleSupport traffic, duplicate message text was removed, and the final labels were manually reviewed using the six-intent taxonomy after inspecting the underlying data.

Twelve fixed examples (two per intent) are reserved as few-shot exemplars for the API classifier and excluded from its final scoring split.

The golden set is deliberately not presented as statistically representative of production traffic; its purpose is to provide a fixed, auditable test set.

---

## 8. Baselines

### 8.1 Trivial baseline

Always predict the majority class.

| Metric | Result |
|---|---:|
| Accuracy | **66.4%** |
| Macro F1 | **11.1%** |

### 8.2 Simple baseline — TF-IDF + Multinomial Naive Bayes

Three-fold stratified cross-validation on the golden set currently gives:

| Metric | Result |
|---|---:|
| Accuracy | **66.4%** |
| Macro F1 | **13.3%** |
| Weighted F1 | **53.0%** |

The simple baseline is useful as a conventional ML reference, but the very small minority classes make its per-class estimates noisy.

---

## 9. Final Evaluation

Run:

```bash
python src/evaluate.py
```

The script writes:

```text
data/processed/evaluation_results.csv
data/processed/evaluation_summary.json
```

When `GEMINI_API_KEY` is configured, the summary also contains the LLM classifier's held-out intent metrics. The repository intentionally does **not** hard-code an unverified LLM score because API/model availability and results can vary.

The evaluation compares:

1. trivial majority baseline;
2. keyword diagnostic baseline;
3. TF-IDF + Naive Bayes baseline;
4. Gemini few-shot classifier, when an API key is configured.

The main metrics are accuracy, macro F1 and weighted F1.

---

## 10. Cross-Validation Experiment

The simple baseline uses stratified 3-fold cross-validation because the smallest intent contains only three examples. The current observed results are:

```text
Accuracy: 0.664
Macro F1: 0.133
Weighted F1: 0.530
```

The macro F1 is much lower than accuracy because the classifier is strong on the dominant `device_software_issue` class but fails to recover several minority intents.

---

## 11. Reply Quality Evaluation

Reply quality is evaluated separately from intent classification.

Run:

```bash
python src/generate_reply_sample.py
```

This creates a fixed 20-example reply sample for judge/human comparison. Only examples that receive a draft reply are retained.

Then manually score the sample:

```bash
python src/label_replies.py
```

The human rubric contains four criteria plus an overall score, each from 1–5:

- Groundedness
- Helpfulness
- Tone
- Conciseness
- Overall

Then run the LLM judge:

```bash
python src/evaluate_judge.py
```

The judge uses the same four criteria and produces:

```text
data/processed/judge_results.csv
data/processed/judge_agreement.json
```

Agreement statistics include:

- human mean;
- judge mean;
- mean absolute error;
- exact agreement rate;
- within-one-point agreement rate;
- Pearson correlation where defined.

This prevents the judge score from being presented as trustworthy without checking it against human ratings.

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

With no API key, the local classifier and retrieval-only fallback are used. With a Gemini API key, the API-enabled classifier and reply drafter are used.

---

## 13. Top Five Failure Patterns

The final report should be populated from `evaluation_results.csv` after an evaluation run rather than from invented examples. The analysis script should focus on the following failure categories:

1. **Device/software vs general information** — short or vague messages can be lexically ambiguous.
2. **Order/purchase vs general information** — customers may omit explicit order or delivery terms.
3. **Payment vs general information** — payment problems are sometimes described indirectly.
4. **Account access vs general information** — account context may be implied rather than stated.
5. **Context-dependent or multilingual messages** — a single tweet can be insufficient without earlier thread context.

For each selected real example, record the expected intent, predicted intent, observed reply/retrieval behaviour and a hypothesis for the failure.

---

## 14. What Is Misleading About My Headline Number?

The headline accuracy is not a production estimate of support quality.

First, the golden set is only 250 examples and is strongly imbalanced: 166 examples belong to `device_software_issue`, while `payment_billing` has only three. A model can therefore achieve high accuracy by favouring the dominant class while performing poorly on minority intents.

Second, intent classification is only one part of the agent. A useful support system also needs a safe handling decision and a grounded, helpful reply.

Third, the historical dataset contains noisy, short and context-dependent customer messages. A single tweet does not always contain enough information to infer the customer's complete issue.

Therefore, the headline result should always be read together with macro F1, per-intent performance, handling behaviour and reply-quality/judge-agreement results.

---

## 15. Limitations

- The golden set is small and imbalanced.
- Several minority intents have very few examples.
- TWCS conversations can lose context when evaluated tweet-by-tweet.
- TF-IDF retrieval measures lexical similarity rather than semantic similarity.
- The prototype does not access live Apple account, order or billing systems.
- Historical replies are examples of past behaviour, not guaranteed current policy.
- The confidence value returned by an LLM is not a calibrated probability.
- Reply-quality evaluation uses a small sample.
- Automatic handling should remain conservative for sensitive cases.

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

AppleSupport was selected as the single brand for the prototype so the classifier and reply generator could be specialized to one support voice.

### Decision 2 — Six-intent taxonomy

A compact taxonomy was chosen to avoid creating many poorly represented classes.

### Decision 3 — Fixed 250-example golden set

250 examples are within the assignment's required 150–250 range and provide a fixed test target.

### Decision 4 — Hand review of labels

Labels are manually reviewed rather than treating automatically generated keyword labels as ground truth.

### Decision 5 — Separate intent and handling

Intent describes the customer's problem; handling describes operational risk. Keeping them separate makes escalation policy inspectable.

### Decision 6 — Twelve fixed few-shot examples

Two examples per intent provide the LLM with compact demonstrations while keeping the evaluation examples held out.

### Decision 7 — TF-IDF retrieval

TF-IDF was selected because it is deterministic, inexpensive and easy to inspect without an external vector database.

### Decision 8 — Top-three retrieval

Three examples provide enough historical context without overloading the generation prompt.

### Decision 9 — Conservative sensitive-intent escalation

Account, payment and refund requests can involve information or actions unavailable to the prototype, so they are routed to humans.

### Decision 10 — Confidence threshold of 0.60

Low-confidence predictions are escalated rather than automatically answered.

### Decision 11 — Trivial majority baseline

The majority baseline exposes the effect of the highly imbalanced golden set.

### Decision 12 — TF-IDF + Naive Bayes baseline

A conventional lightweight ML model provides a stronger reference than the trivial baseline while remaining fully reproducible offline.

### Decision 13 — Separate reply-quality evaluation

Intent accuracy cannot tell whether a generated response is grounded, helpful, concise and appropriate.

### Decision 14 — Human validation of the LLM judge

The judge is compared with human ratings so its scores are not treated as automatically reliable.

### Decision 15 — Do not hard-code unverified LLM metrics

LLM evaluation depends on API access and model behaviour. The repository generates those metrics when run instead of claiming numbers that were not actually measured.

---

## 18. Installation

### Requirements

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- SciPy
- google-genai
- python-dotenv
- pytest

Install:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

For Gemini-powered classification/replies/judging, create `.env` locally:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

**Never commit `.env` or an API key.**

---

## 19. Running the Agent

```bash
python src/agent.py "My iPhone won't update to the latest iOS"
```

The agent prints the intent, confidence, handling decision, reason, draft reply and retrieved grounding examples.

---

## 20. Running Evaluation

### Main classification evaluation

```bash
python src/evaluate.py
```

Optional limited API run:

```bash
python src/evaluate.py --limit 20
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

### LLM judge + agreement

```bash
python src/evaluate_judge.py
```

### Reply diagnostics

```bash
python src/reply_metrics.py
```

### Tests

```bash
pytest tests/
```

---

## 21. Project Structure

```text
hiver-support-agent/
├── data/
│   ├── raw/
│   │   └── twcs.csv                 # downloaded separately; gitignored
│   └── processed/
│       ├── apple_conversations.csv
│       ├── golden_set.csv
│       └── support_data.csv
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

The headline evaluation is based on a fixed 250-example golden set and committed processed Apple artifacts.

The main command is:

```bash
python src/evaluate.py
```

Results are saved to:

```text
data/processed/evaluation_results.csv
data/processed/evaluation_summary.json
```

The raw TWCS file is not required to rerun the main evaluation because the processed Apple conversations and golden set are included. The raw dataset is only needed if the data-preparation step needs to be reproduced from scratch.

The evaluation is designed to finish quickly on the committed processed artifacts; API-enabled runs take longer because each LLM classification is an external model call.

---

## 23. Conclusion

This project implements an end-to-end Apple customer-support prototype:

```text
Customer Message
       ↓
Intent Classification
       ↓
Handling Decision
       ↓
Historical AppleSupport Retrieval
       ↓
Grounded Reply Draft
       ↓
Auto-handle / Escalate
```

The project deliberately separates **classification performance**, **operational handling**, and **reply quality**. It also reports the weaknesses of the fixed golden set rather than relying on a single headline accuracy number.

The main remaining improvement areas are semantic retrieval, minority-intent coverage, confidence calibration, thread context and stronger human-validated reply evaluation.
