# Apple AI Customer Support Agent — Evaluation Report

## Problem framing

Build a support agent specialized to @AppleSupport that classifies incoming messages, drafts replies grounded in historical AppleSupport behaviour, and decides whether to auto-handle or escalate. The prototype intentionally does not access live Apple account, billing, order, or device systems.

## Evaluation design

A fixed 250-example golden set is used. Twelve examples are reserved as few-shot demonstrations for the API classifier, leaving 238 held-out examples for that classifier's headline evaluation. The golden set is highly imbalanced, so accuracy is reported alongside macro F1 and weighted F1.

## Baselines

- Trivial majority baseline: always predicts `device_software_issue`.
- Simple baseline: TF-IDF + Multinomial Naive Bayes with stratified 3-fold cross-validation.
- Final system: Gemini few-shot classifier when an API key is configured.

## Current verified baseline results

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Trivial majority | 66.4% | 11.1% | 44.1% |
| TF-IDF + Naive Bayes | 66.4% | 13.3% | 53.0% |
| Gemini few-shot | Run `python src/evaluate.py` | Run command | Run command |

The final LLM result is intentionally generated during evaluation rather than fabricated in this report.

## Reply quality

Run `generate_reply_sample.py`, manually label the resulting sample with `label_replies.py`, then run `evaluate_judge.py`. The resulting human-vs-judge statistics are saved to `data/processed/judge_agreement.json`.

## Failure analysis

After the final evaluation run, inspect the confusion patterns in `evaluation_results.csv` and document five real examples. The expected categories are device/software vs general information, purchase/order vs general information, payment vs general information, account vs general information, and context-dependent or multilingual messages.

## What is misleading about the headline number?

The majority class contains 166 of 250 examples. Consequently, a classifier that predicts only that class already reaches 66.4% accuracy while having macro F1 of only 11.1%. Accuracy therefore does not establish that the agent understands all six support intents. Reply usefulness and safe handling are also separate dimensions.

## One more week

Increase minority-class annotation coverage, add a second annotator, use semantic retrieval, calibrate escalation thresholds, preserve thread context, improve multilingual handling, and enlarge the human-validated reply-quality evaluation.
