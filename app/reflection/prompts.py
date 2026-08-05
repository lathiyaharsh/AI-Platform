"""
Prompts for response reflection and auto-improvement.

Kept as plain builders so PromptManager / versioning can adopt them later
without changing ReflectionService call sites.
"""

REFLECTION_SYSTEM_PROMPT = (
    "You are a strict response quality evaluator for a production AI platform. "
    "Judge only the given answer against the question. "
    "Respond with ONLY valid JSON. No markdown. No commentary."
)

IMPROVE_SYSTEM_PROMPT = (
    "You are a careful answer editor for a production AI platform. "
    "Improve clarity and completeness while preserving factual content. "
    "Remove hallucinations and unsupported claims. "
    "Return ONLY the improved answer. No preamble. No markdown fences."
)


def build_reflection_prompt(*, question: str, answer: str) -> str:
    return (
        "Evaluate the assistant answer on:\n"
        "- correctness\n"
        "- completeness\n"
        "- clarity\n"
        "- hallucination risk\n"
        "- missing information\n"
        "\n"
        "Original Question:\n"
        f"{question.strip()}\n"
        "\n"
        "Original Answer:\n"
        f"{answer.strip()}\n"
        "\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  "improve": true,\n'
        '  "reason": "short explanation",\n'
        '  "score": 0.0\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- score is from 0.0 (poor) to 1.0 (excellent)\n"
        "- set improve=true when the answer needs a rewrite\n"
        "- do not include any text outside the JSON object\n"
    )


def build_improve_prompt(*, question: str, answer: str) -> str:
    return (
        "Improve the following answer.\n"
        "\n"
        "Keep factual information.\n"
        "Remove hallucinations.\n"
        "Increase clarity.\n"
        "Fill obvious missing information only when grounded in the question.\n"
        "\n"
        "Original Question:\n"
        f"{question.strip()}\n"
        "\n"
        "Original Answer:\n"
        f"{answer.strip()}\n"
        "\n"
        "Return ONLY the improved answer.\n"
    )
