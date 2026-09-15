"""
generate_reply_sample.py

Builds a sample of (message, draft_reply) pairs for the
LLM-judge vs human agreement evaluation.

The agent runs completely locally using:
- TF-IDF + Naive Bayes classification
- Historical Apple Support reply retrieval

No API key or external service is required.

Run after this:
    python src/label_replies.py

Then:
    python src/evaluate_judge.py
"""

import os
import sys

import pandas as pd


sys.path.insert(
    0,
    os.path.dirname(__file__),
)

from agent import (  # noqa: E402
    AppleSupportAgent,
    GOLDEN_SET_PATH,
)


SAMPLE_SIZE = 20

OUTPUT_FILE = (
    "data/processed/reply_judge_sample.csv"
)


def main():
    df = pd.read_csv(
        GOLDEN_SET_PATH
    )

    sample = df.sample(
        n=min(
            SAMPLE_SIZE,
            len(df),
        ),
        random_state=7,
    )

    agent = AppleSupportAgent()

    rows = []

    for _, row in sample.iterrows():

        result = agent.handle(
            row["text"]
        )

        rows.append(
            {
                "tweet_id": row["tweet_id"],
                "message": row["text"],
                "intent": result["intent"],
                "decision": result["decision"],
                "draft_reply": result["draft_reply"],
                "grounding_context": "\n\n".join(
                    (
                        "Past customer: {}\n"
                        "Apple reply: {}"
                    ).format(
                        example["past_message"],
                        example["past_reply"],
                    )
                    for example in result.get(
                        "grounding_examples",
                        [],
                    )
                ),
                "human_groundedness": "",
                "human_helpfulness": "",
                "human_tone": "",
                "human_conciseness": "",
                "human_overall": "",
            }
        )

    out = pd.DataFrame(
        rows
    )

    # Escalated messages do not receive a draft reply.
    # They are excluded because there is no reply to evaluate.
    before = len(out)

    out = out[
        out["draft_reply"].notna()
        & (
            out["draft_reply"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Sampled {} golden-set messages; "
        "{} received a drafted reply. "
        "Escalated messages were excluded.".format(
            before,
            len(out),
        )
    )

    print(
        "Saved to {}".format(
            OUTPUT_FILE
        )
    )

    print(
        "Next: run src/label_replies.py "
        "to add human ratings."
    )


if __name__ == "__main__":
    main()