import json
from unittest.mock import MagicMock, patch

import pytest

from src.judge import ReplyJudge


class TestReplyJudge:

    def test_judge_parses_valid_scores(self):
        mock_response = json.dumps({
            "response": json.dumps({
                "groundedness": 4,
                "helpfulness": 5,
                "tone": 4,
                "conciseness": 5,
                "overall": 4,
            })
        })

        with patch(
            "src.judge.urllib.request.urlopen"
        ) as mock_urlopen:

            response = MagicMock()
            response.read.return_value = (
                mock_response.encode("utf-8")
            )

            mock_urlopen.return_value.__enter__.return_value = (
                response
            )

            judge = ReplyJudge()

            result = judge.judge(
                message="My iPhone won't update.",
                draft_reply=(
                    "Please check your internet connection "
                    "and try again."
                ),
                grounding_context=(
                    "Past customer: iPhone update failing\n"
                    "Apple reply: Try checking your connection."
                ),
            )

        assert result["groundedness"] == 4
        assert result["helpfulness"] == 5
        assert result["tone"] == 4
        assert result["conciseness"] == 5
        assert result["overall"] == 4

    def test_judge_requires_ollama(self):
        with patch(
            "src.judge.urllib.request.urlopen",
            side_effect=Exception(
                "Ollama unavailable"
            ),
        ):
            with pytest.raises(RuntimeError):
                ReplyJudge()

    def test_judge_handles_malformed_response(self):
        mock_response = json.dumps({
            "response": "This is not valid JSON"
        })

        with patch(
            "src.judge.urllib.request.urlopen"
        ) as mock_urlopen:

            response = MagicMock()
            response.read.return_value = (
                mock_response.encode("utf-8")
            )

            mock_urlopen.return_value.__enter__.return_value = (
                response
            )

            judge = ReplyJudge()

            with pytest.raises(ValueError):
                judge.judge(
                    message="My iPhone won't update.",
                    draft_reply="Please try again.",
                    grounding_context="",
                )