"""Run the local Qwen LLM-as-judge and compare it with human ratings.

Prerequisite:
  python src/generate_reply_sample.py
  python src/label_replies.py

The judge uses:
  Qwen2.5 3B
  Ollama
  localhost:11434

No Gemini or external API is used.

Outputs:
  data/processed/judge_results.csv
  data/processed/judge_agreement.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))

from judge import ReplyJudge, CRITERIA


INPUT = "data/processed/reply_judge_sample.csv"
OUTPUT = "data/processed/judge_results.csv"
SUMMARY = "data/processed/judge_agreement.json"


def main():
    if not os.path.exists(INPUT):
        raise SystemExit(
            "{} not found. Run generate_reply_sample.py first.".format(
                INPUT
            )
        )

    df = pd.read_csv(INPUT)

    required = [
        "human_{}".format(c) for c in CRITERIA
    ] + ["human_overall"]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise SystemExit(
            "Missing human rating columns: {}".format(missing)
        )

    if df[required].isna().any().any():
        raise SystemExit(
            "Human ratings are incomplete. "
            "Finish label_replies.py first."
        )

    print("Starting Qwen2.5 3B LLM-as-judge...")
    print("Model: qwen2.5:3b")
    print("Runtime: Ollama")
    print("Samples: {}".format(len(df)))
    print()

    judge = ReplyJudge()

    judge_rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(
            "Judging {}/{}...".format(i, len(df)),
            flush=True
        )

        grounding = str(
            row.get("grounding_context", "")
        )

        result = judge.judge(
            row["message"],
            row["draft_reply"],
            grounding,
        )

        judge_rows.append(result)

    judge_df = pd.DataFrame(judge_rows)

    out = pd.concat(
        [
            df.reset_index(drop=True),
            judge_df.reset_index(drop=True).add_prefix("judge_"),
        ],
        axis=1,
    )

    out.to_csv(OUTPUT, index=False)

    summary = {
        "evaluation_type": "local_llm_as_judge",
        "llm_judge_available": True,
        "model": "qwen2.5:3b",
        "runtime": "Ollama",
        "endpoint": "localhost:11434",
        "n": int(len(out)),
        "criteria": {},
    }

    for criterion in CRITERIA + ["overall"]:
        human = pd.to_numeric(
            out["human_{}".format(criterion)],
            errors="coerce",
        )

        judge_scores = pd.to_numeric(
            out["judge_{}".format(criterion)],
            errors="coerce",
        )

        mask = human.notna() & judge_scores.notna()

        if not mask.any():
            continue

        x = human[mask].to_numpy()
        y = judge_scores[mask].to_numpy()

        if (
            len(x) > 1
            and np.std(x) > 0
            and np.std(y) > 0
        ):
            correlation = float(
                np.corrcoef(x, y)[0, 1]
            )
        else:
            correlation = None

        summary["criteria"][criterion] = {
            "human_mean": float(np.mean(x)),
            "judge_mean": float(np.mean(y)),
            "mae": float(
                mean_absolute_error(x, y)
            ),
            "exact_agreement_rate": float(
                np.mean(x == y)
            ),
            "within_1_point_rate": float(
                np.mean(np.abs(x - y) <= 1)
            ),
            "pearson_correlation": correlation,
        }

    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("QWEN LLM-AS-JUDGE RESULTS")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    print()
    print("Saved:")
    print("  {}".format(OUTPUT))
    print("  {}".format(SUMMARY))


if __name__ == "__main__":
    main()