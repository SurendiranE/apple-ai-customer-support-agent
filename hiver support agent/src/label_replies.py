"""
label_replies.py — CLI for a human (you) to rate drafted replies on the
same rubric the LLM judge uses. This produces the ground truth that
evaluate_judge.py checks the judge's agreement against — the assignment's
"evidence of how well your judge agrees with a human" requirement.

Run after src/generate_reply_sample.py.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from judge import RUBRIC, CRITERIA  # noqa: E402

FILE = "data/processed/reply_judge_sample.csv"


def prompt_score(label, description):
    while True:
        raw = input(f"  {label} (1-5) — {description}\n  > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter a number from 1 to 5.")


def main():
    if not os.path.exists(FILE):
        print(f"{FILE} not found — run src/generate_reply_sample.py first.")
        return

    df = pd.read_csv(FILE)
    df["human_overall"] = df["human_overall"].fillna("")

    remaining = df["human_overall"].astype(str).str.strip().eq("").sum()
    if remaining == 0:
        print("All replies already rated. Delete a human_* value in the "
              "CSV if you want to re-rate a specific row.")
        return

    print(f"\n{remaining} replies to rate. Enter 'q' at any prompt to save and quit.\n")

    for i in range(len(df)):
        if str(df.loc[i, "human_overall"]).strip() != "":
            continue

        print("=" * 60)
        print(f"Example {i + 1} of {len(df)}  (intent: {df.loc[i, 'intent']})")
        print(f"\nCustomer message:\n  {df.loc[i, 'message']}")
        print(f"\nDraft reply:\n  {df.loc[i, 'draft_reply']}\n")

        scores = {}
        quit_now = False
        for criterion in CRITERIA:
            while True:
                raw = input(f"  {criterion} (1-5) — {RUBRIC[criterion]}\n  > ").strip()
                if raw.lower() == "q":
                    quit_now = True
                    break
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    scores[criterion] = int(raw)
                    break
                print("  Enter a number from 1 to 5, or 'q' to quit.")
            if quit_now:
                break

        if quit_now:
            df.to_csv(FILE, index=False)
            print("\nProgress saved.")
            return

        while True:
            raw = input("  overall (1-5) — your holistic judgment\n  > ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 5:
                overall = int(raw)
                break
            print("  Enter a number from 1 to 5.")

        for criterion, score in scores.items():
            df.loc[i, f"human_{criterion}"] = score
        df.loc[i, "human_overall"] = overall

        df.to_csv(FILE, index=False)
        print("Saved.\n")

    print("All replies rated. Run src/evaluate_judge.py next.")


if __name__ == "__main__":
    main()