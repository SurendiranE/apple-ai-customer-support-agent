"""Apple Support Agent.

Three pieces:
1. IntentClassifier   - classifies a message into one of 6 intents
2. ReplyDrafter       - drafts a reply using similar historical replies
3. EscalationDecider  - decides auto-handle vs escalate

The implementation is fully local and does not use Gemini or any
external API.

Run:
    python src/agent.py "My iPhone won't update to the latest iOS"
"""

import json
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


INTENTS = [
    "device_software_issue",
    "account_access",
    "payment_billing",
    "refund_request",
    "purchase_order",
    "general_information",
]


INTENT_DESCRIPTIONS = {
    "device_software_issue": (
        "Problems with iOS, apps, updates, bugs, crashes, "
        "battery, or device performance."
    ),
    "account_access": (
        "Trouble signing in, Apple ID, passwords, "
        "two-factor authentication, or locked accounts."
    ),
    "payment_billing": (
        "Questions or complaints about charges, subscriptions, "
        "invoices, or billing."
    ),
    "refund_request": (
        "Explicit requests for a refund or money back."
    ),
    "purchase_order": (
        "Order status, purchases, trade-ins, delivery, or shipping."
    ),
    "general_information": (
        "Everything else, including general questions, praise, "
        "unclear messages, or off-topic messages."
    ),
}


GOLDEN_SET_PATH = "data/processed/golden_set.csv"
CONVERSATIONS_PATH = "data/processed/apple_conversations.csv"


SENSITIVE_INTENTS = {
    "payment_billing",
    "refund_request",
    "account_access",
}


CONFIDENCE_THRESHOLD = 0.6


def load_few_shot_examples(
    golden_set_path=GOLDEN_SET_PATH,
    per_intent=2,
):
    """Load a small fixed number of examples from each intent."""

    df = pd.read_csv(golden_set_path)

    examples = []
    exclude_ids = set()

    for intent, group in df.groupby("custom_intent"):
        picked = group.sort_values("tweet_id").head(per_intent)

        exclude_ids.update(
            picked["tweet_id"].tolist()
        )

        for _, row in picked.iterrows():
            examples.append(
                {
                    "text": row["text"],
                    "intent": row["custom_intent"],
                }
            )

    return examples, exclude_ids


class IntentClassifier:
    """Local TF-IDF + Multinomial Naive Bayes intent classifier."""

    def __init__(
        self,
        golden_set_path=GOLDEN_SET_PATH,
    ):
        df = pd.read_csv(golden_set_path)

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    MultinomialNB(),
                ),
            ]
        )

        self.pipeline.fit(
            df["text"],
            df["custom_intent"],
        )

    def classify(self, text):
        """Classify a customer message locally."""

        prediction = self.pipeline.predict([text])[0]

        probabilities = self.pipeline.predict_proba([text])[0]

        confidence = float(
            max(probabilities)
        )

        return {
            "intent": prediction,
            "confidence": confidence,
            "reasoning": (
                "Local TF-IDF + Naive Bayes classifier; "
                "no external API used."
            ),
        }


# Backward-compatible alias used by older code.
LocalIntentClassifier = IntentClassifier


class ReplyRetriever:
    """Retrieve similar historical Apple Support conversations."""

    def __init__(
        self,
        conversations_path=CONVERSATIONS_PATH,
    ):
        self.df = pd.read_csv(
            conversations_path
        ).dropna(
            subset=[
                "text",
                "reply_text",
            ]
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["text"]
        )

    def retrieve(
        self,
        query,
        k=3,
    ):
        """Return the k most similar historical conversations."""

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        top_indices = similarities.argsort()[::-1][:k]

        results = []

        for index in top_indices:
            row = self.df.iloc[index]

            results.append(
                {
                    "past_message": row["text"],
                    "past_reply": row["reply_text"],
                    "similarity": float(
                        similarities[index]
                    ),
                }
            )

        return results


class ReplyDrafter:
    """Draft a reply from the most similar historical Apple replies."""

    def __init__(
        self,
        retriever=None,
    ):
        self.retriever = (
            retriever
            if retriever is not None
            else ReplyRetriever()
        )

    def draft(
        self,
        text,
        intent,
        k=3,
    ):
        """Return a historically grounded reply and its evidence."""

        examples = self.retriever.retrieve(
            text,
            k=k,
        )

        if not examples:
            return (
                "Thanks for reaching out. "
                "Please contact Apple Support for further assistance.",
                [],
            )

        # Use the highest-similarity historical reply as the draft.
        best_reply = examples[0]["past_reply"]

        return best_reply, examples


class EscalationDecider:
    """Deterministic escalation rules."""

    def __init__(
        self,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sensitive_intents=SENSITIVE_INTENTS,
    ):
        self.confidence_threshold = confidence_threshold
        self.sensitive_intents = sensitive_intents

    def decide(
        self,
        intent,
        confidence,
    ):
        if confidence < self.confidence_threshold:
            return (
                "escalate",
                (
                    "classifier confidence {:.2f} is below "
                    "threshold {:.2f}".format(
                        confidence,
                        self.confidence_threshold,
                    )
                ),
            )

        if intent in self.sensitive_intents:
            return (
                "escalate",
                (
                    "intent '{}' involves money or account "
                    "access — routed to a human".format(
                        intent
                    )
                ),
            )

        return (
            "auto_handle",
            (
                "high-confidence ({:.2f}) non-sensitive intent".format(
                    confidence
                )
            ),
        )


class AppleSupportAgent:
    """Main Apple Support Agent."""

    def __init__(self):
        self.few_shot_ids = set()

        self.classifier = IntentClassifier()

        self.retriever = ReplyRetriever()

        self.drafter = ReplyDrafter(
            retriever=self.retriever
        )

        self.decider = EscalationDecider()

    def handle(self, text):
        """Process one customer message."""

        classification = self.classifier.classify(
            text
        )

        intent = classification["intent"]

        confidence = float(
            classification.get(
                "confidence",
                0.0,
            )
        )

        decision, reason = self.decider.decide(
            intent,
            confidence,
        )

        reply = None
        grounding = []

        if decision == "auto_handle":
            reply, grounding = self.drafter.draft(
                text,
                intent,
            )

        return {
            "message": text,
            "intent": intent,
            "confidence": confidence,
            "classifier_reasoning": classification.get(
                "reasoning"
            ),
            "decision": decision,
            "decision_reason": reason,
            "draft_reply": reply,
            "grounding_examples": grounding,
        }


if __name__ == "__main__":
    message = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "My iPhone won't update to the latest iOS, "
            "it just keeps failing"
        )
    )

    agent = AppleSupportAgent()

    result = agent.handle(
        message
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )