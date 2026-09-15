"""Local LLM-as-judge using Qwen2.5 through Ollama.

No Gemini or external API is used.
The model runs locally through Ollama at localhost:11434.
"""

import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

CRITERIA = [
    "groundedness",
    "helpfulness",
    "tone",
    "conciseness",
]


RUBRIC = {
    "groundedness": (
        "Does the reply stay consistent with the supplied historical "
        "support context and avoid unsupported claims?"
    ),
    "helpfulness": (
        "Does the reply meaningfully address the customer's issue "
        "and provide a useful next step?"
    ),
    "tone": (
        "Is the reply professional, empathetic, respectful, and "
        "appropriate for customer support?"
    ),
    "conciseness": (
        "Is the reply concise and appropriate for a Twitter-style "
        "customer-support response?"
    ),
}


def call_ollama(prompt, timeout=180):
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data.get("response", "").strip()


def extract_json(text):
    """Extract a JSON object from an LLM response."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Qwen did not return a valid JSON object:\n" + text
        )

    return json.loads(text[start:end + 1])


class ReplyJudge:
    """Evaluate support replies using Qwen2.5 3B."""

    def __init__(self):
        # Test the local Ollama connection immediately.
        try:
            call_ollama(
                "Return only the word OK.",
                timeout=30
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen2.5/Ollama is unavailable. "
                "Make sure Ollama is running and qwen2.5:3b is installed. "
                "Original error: {}".format(exc)
            )

    def judge(self, message, draft_reply, grounding_context):
        rubric_text = "\n".join(
            "- {}: {}".format(k, v)
            for k, v in RUBRIC.items()
        )

        prompt = """You are an expert customer-support quality evaluator.

Evaluate the drafted support reply against the customer's message
and the historical support context.

CUSTOMER MESSAGE:
{}

DRAFT REPLY:
{}

HISTORICAL GROUNDING CONTEXT:
{}

Score each criterion from 1 to 5:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Criteria:
{}

Overall is your overall quality score from 1 to 5.

Important:
- Groundedness must be based on the supplied historical context.
- Do not reward unsupported claims.
- Helpfulness should consider whether the customer receives a useful
  answer or next step.
- Tone should reflect professional customer support.
- Conciseness should consider Twitter-style support.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not include explanations outside the JSON.

Return exactly this structure:

{{
  "groundedness": <integer 1-5>,
  "helpfulness": <integer 1-5>,
  "tone": <integer 1-5>,
  "conciseness": <integer 1-5>,
  "overall": <integer 1-5>
}}
""".format(
            message,
            draft_reply,
            grounding_context,
            rubric_text,
        )

        raw = call_ollama(prompt)
        result = extract_json(raw)

        output = {}

        for criterion in CRITERIA + ["overall"]:
            value = result.get(criterion)

            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Invalid {} score returned by Qwen: {}".format(
                        criterion,
                        value
                    )
                )

            if value < 1 or value > 5:
                raise ValueError(
                    "{} score must be between 1 and 5, got {}".format(
                        criterion,
                        value
                    )
                )

            output[criterion] = value

        return output