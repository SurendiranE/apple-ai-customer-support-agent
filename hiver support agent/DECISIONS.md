# Decision Log

1. **Selected AppleSupport** — keeps the agent focused on one brand and support domain.

2. **Six intents** — a compact taxonomy avoids creating many sparse classes.

3. **250-example golden set** — within the assignment's required 150–250 range and provides a fixed evaluation target.

4. **Manual label review** — prevents automatically generated keyword labels from being treated as ground truth.

5. **Separate intent and handling** — problem type and operational risk are different decisions, so they are evaluated independently.

6. **Two fixed examples per intent** — reserves 12 examples from the classification evaluation split and provides a small fixed demonstration set.

7. **TF-IDF retrieval** — deterministic, inexpensive, and easy to reproduce without a vector database or external service.

8. **Top-three retrieval** — provides enough historical context while keeping the retrieval output compact and inspectable.

9. **Sensitive-intent escalation** — account, payment, and refund cases may require human or account-specific handling that the prototype cannot safely perform.

10. **0.60 confidence threshold** — low-confidence classifications are escalated rather than automatically answered.

11. **Majority baseline** — exposes the effect of the strong class imbalance in the golden set.

12. **Keyword diagnostic baseline** — provides a transparent rule-based reference that is easy to inspect.

13. **TF-IDF + Naive Bayes** — provides a conventional lightweight offline ML reference without requiring an external API.

14. **Separate reply evaluation** — classification metrics do not measure whether a drafted response is grounded, helpful, concise, or appropriately toned.

15. **Human validation of LLM judge** — Qwen2.5 judge scores are compared against human ratings rather than being treated as ground truth.

16. **Local Qwen2.5 3B through Ollama** — provides an LLM-as-judge without Gemini or an external API key.

17. **No fabricated LLM metrics** — judge agreement results are produced by the evaluation pipeline from the labelled reply sample.

18. **Historical reply grounding** — the prototype uses the highest-ranked historical AppleSupport reply as the draft instead of generating an unconstrained response.

19. **Conservative prototype scope** — the system does not access live Apple account, billing, order, refund, or device systems, so sensitive cases are routed to humans.

20. **Accuracy reported with Macro F1** — the headline accuracy is explicitly paired with macro F1 and baseline results because the golden set is highly imbalanced.
