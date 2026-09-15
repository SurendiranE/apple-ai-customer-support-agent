"""End-to-end evaluation harness for the Apple Support Agent.

Evaluates three reproducible local classification approaches on a fixed
holdout of the 250-example golden set:

1. Majority baseline
2. Keyword baseline
3. TF-IDF + Naive Bayes

The LLM-as-judge evaluation using Qwen2.5 through Ollama is handled
separately by evaluate_judge.py.

Results are written to:
- data/processed/evaluation_results.csv
- data/processed/evaluation_summary.json
"""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))

from agent import (  # noqa: E402
    load_few_shot_examples,
    GOLDEN_SET_PATH,
)
from simple_baseline import build_pipeline  # noqa: E402


OUTPUT = "data/processed/evaluation_results.csv"
SUMMARY = "data/processed/evaluation_summary.json"


def trivial_baseline(text):
    """Always predict the majority intent."""
    return "device_software_issue"


def keyword_baseline(text):
    """Transparent lexical baseline used as an additional diagnostic."""
    text = str(text).lower()

    if any(
        w in text
        for w in ["refund", "money back", "reimburse"]
    ):
        return "refund_request"

    if any(
        w in text
        for w in [
            "charged",
            "billing",
            "payment",
            "invoice",
            "subscription",
            "price",
        ]
    ):
        return "payment_billing"

    if any(
        w in text
        for w in [
            "order",
            "purchase",
            "bought",
            "buy",
            "delivery",
            "shipping",
            "arrived",
            "package",
            "trade in",
            "pre-order",
            "preorder",
        ]
    ):
        return "purchase_order"

    if any(
        w in text
        for w in [
            "password",
            "login",
            "log in",
            "sign in",
            "locked",
            "account",
            "verification",
            "two-factor",
            "2fa",
            "apple id",
        ]
    ):
        return "account_access"

    if any(
        w in text
        for w in [
            "app",
            "update",
            "ios",
            "software",
            "bug",
            "crash",
            "not working",
            "error",
            "battery",
            "freeze",
            "frozen",
            "restart",
        ]
    ):
        return "device_software_issue"

    return "general_information"


def build_eval_split():
    """Remove the few-shot examples from the evaluation set."""
    df = pd.read_csv(GOLDEN_SET_PATH)

    _, few_shot_ids = load_few_shot_examples(
        GOLDEN_SET_PATH
    )

    eval_df = df[
        ~df["tweet_id"].isin(few_shot_ids)
    ].copy()

    return eval_df, few_shot_ids


def metrics(y_true, y_pred):
    """Calculate standard classification metrics."""
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for evaluation rows.",
    )

    args = parser.parse_args()

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    df, held_out_ids = build_eval_split()

    if args.limit:
        df = df.head(args.limit).copy()

    rows = []

    y_true = df[
        "custom_intent"
    ].tolist()

    # ---------------------------------------------------------
    # Baseline 1: Majority classifier
    # ---------------------------------------------------------

    trivial = [
        trivial_baseline(text)
        for text in df["text"]
    ]

    # ---------------------------------------------------------
    # Baseline 2: Keyword diagnostic
    # ---------------------------------------------------------

    keyword = [
        keyword_baseline(text)
        for text in df["text"]
    ]

    # ---------------------------------------------------------
    # Baseline 3: TF-IDF + Naive Bayes
    # ---------------------------------------------------------

    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_predict,
    )

    simple_df = pd.read_csv(
        GOLDEN_SET_PATH
    )

    min_class = int(
        simple_df[
            "custom_intent"
        ].value_counts().min()
    )

    if min_class >= 2:
        cv = StratifiedKFold(
            n_splits=min(5, min_class),
            shuffle=True,
            random_state=42,
        )

        simple_pred_all = cross_val_predict(
            build_pipeline(),
            simple_df["text"],
            simple_df["custom_intent"],
            cv=cv,
        )

        simple_map = dict(
            zip(
                simple_df["tweet_id"],
                simple_pred_all,
            )
        )

        simple = [
            simple_map[tweet_id]
            for tweet_id in df["tweet_id"]
        ]

    else:
        simple = [
            "general_information"
            for _ in range(len(df))
        ]

    # ---------------------------------------------------------
    # Evaluation summary
    # ---------------------------------------------------------

    full_golden = pd.read_csv(
        GOLDEN_SET_PATH
    )

    summary = {
        "golden_set_size": int(
            full_golden.shape[0]
        ),
        "evaluation_size": len(df),
        "held_out_few_shot_examples": len(
            held_out_ids
        ),
        "trivial_majority": metrics(
            y_true,
            trivial,
        ),
        "keyword_diagnostic": metrics(
            y_true,
            keyword,
        ),
        "tfidf_naive_bayes_cv": metrics(
            y_true,
            simple,
        ),
        "llm_classifier": None,
        "llm_classifier_note": (
            "The classification evaluation uses "
            "reproducible local baselines. "
            "Qwen2.5 through Ollama is used separately "
            "as the LLM-as-judge in evaluate_judge.py."
        ),
    }

    # ---------------------------------------------------------
    # Store baseline predictions
    # ---------------------------------------------------------

    for i, (_, row) in enumerate(
        df.iterrows()
    ):
        rows.append(
            {
                "tweet_id": row["tweet_id"],
                "text": row["text"],
                "expected_intent": row[
                    "custom_intent"
                ],
                "trivial_pred": trivial[i],
                "keyword_pred": keyword[i],
                "simple_baseline_pred": simple[i],
            }
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    out = pd.DataFrame(rows)

    out.to_csv(
        OUTPUT,
        index=False,
    )

    with open(
        SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nSaved: {}\n"
        "Saved: {}".format(
            OUTPUT,
            SUMMARY,
        )
    )


if __name__ == "__main__":
    main()