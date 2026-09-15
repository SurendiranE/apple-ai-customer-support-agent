"""Small, deterministic diagnostics for generated replies."""
import argparse
import os
import re
import pandas as pd

INPUT = "data/processed/reply_judge_sample.csv"
OUTPUT = "data/processed/reply_metrics.csv"


def overlap(a, b):
    aw = set(re.findall(r"\b[a-z0-9']+\b", str(a).lower()))
    bw = set(re.findall(r"\b[a-z0-9']+\b", str(b).lower()))
    return len(aw & bw) / max(1, len(aw))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    df["reply_length"] = df["draft_reply"].fillna("").astype(str).str.len()
    df["message_reply_word_overlap"] = [overlap(a, b) for a, b in zip(df["message"], df["draft_reply"].fillna(""))]
    df["has_actionable_language"] = df["draft_reply"].fillna("").str.lower().str.contains(r"try|check|visit|contact|follow|go to|open", regex=True)
    df[["tweet_id", "reply_length", "message_reply_word_overlap", "has_actionable_language"]].to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")

if __name__ == "__main__":
    main()
