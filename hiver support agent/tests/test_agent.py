"""
Unit tests for agent.py.

These tests use the local TF-IDF + Naive Bayes classifier and
historical-reply retriever. No API key or network access is required.
"""

import os
import sys

import pandas as pd


sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
    ),
)

from agent import (  # noqa: E402
    EscalationDecider,
    IntentClassifier,
    ReplyRetriever,
)


class TestEscalationDecider:

    def setup_method(self):
        self.decider = EscalationDecider(
            confidence_threshold=0.6,
            sensitive_intents={
                "payment_billing",
                "refund_request",
            },
        )

    def test_low_confidence_escalates(self):
        decision, reason = self.decider.decide(
            "device_software_issue",
            0.3,
        )

        assert decision == "escalate"
        assert "confidence" in reason

    def test_sensitive_intent_escalates_even_with_high_confidence(self):
        decision, reason = self.decider.decide(
            "refund_request",
            0.95,
        )

        assert decision == "escalate"
        assert "refund_request" in reason

    def test_high_confidence_non_sensitive_auto_handles(self):
        decision, reason = self.decider.decide(
            "device_software_issue",
            0.9,
        )

        assert decision == "auto_handle"


class TestIntentClassifier:

    def test_classifier_returns_valid_intent(self):
        classifier = IntentClassifier()

        result = classifier.classify(
            "my iPhone won't update"
        )

        assert result["intent"] in {
            "device_software_issue",
            "account_access",
            "payment_billing",
            "refund_request",
            "purchase_order",
            "general_information",
        }

    def test_classifier_returns_confidence(self):
        classifier = IntentClassifier()

        result = classifier.classify(
            "my iPhone won't update"
        )

        assert isinstance(
            result["confidence"],
            float,
        )

        assert 0.0 <= result["confidence"] <= 1.0

    def test_classifier_returns_reasoning(self):
        classifier = IntentClassifier()

        result = classifier.classify(
            "my iPhone won't update"
        )

        assert "reasoning" in result
        assert len(result["reasoning"]) > 0


class TestReplyRetriever:

    def test_retrieve_returns_k_results(self, tmp_path):
        csv_path = tmp_path / "conversations.csv"

        pd.DataFrame(
            {
                "tweet_id": [1, 2, 3],
                "text": [
                    "my iphone battery drains fast",
                    "app keeps crashing on ios 17",
                    "how do I reset my password",
                ],
                "reply_tweet_id": [2, 4, 6],
                "reply_text": [
                    "DM us your device details",
                    "Try reinstalling the app",
                    "Reset it from Settings",
                ],
            }
        ).to_csv(
            csv_path,
            index=False,
        )

        retriever = ReplyRetriever(
            conversations_path=str(csv_path)
        )

        results = retriever.retrieve(
            "my battery is draining quickly",
            k=2,
        )

        assert len(results) == 2
        assert "past_message" in results[0]
        assert "past_reply" in results[0]
        assert "similarity" in results[0]

    def test_retrieve_returns_similarity_score(self, tmp_path):
        csv_path = tmp_path / "conversations.csv"

        pd.DataFrame(
            {
                "tweet_id": [1, 2],
                "text": [
                    "iphone battery problem",
                    "reset apple id password",
                ],
                "reply_tweet_id": [3, 4],
                "reply_text": [
                    "Please send us your device details",
                    "Please follow the Apple ID recovery steps",
                ],
            }
        ).to_csv(
            csv_path,
            index=False,
        )

        retriever = ReplyRetriever(
            conversations_path=str(csv_path)
        )

        results = retriever.retrieve(
            "iphone battery problem",
            k=1,
        )

        assert len(results) == 1
        assert isinstance(
            results[0]["similarity"],
            float,
        )