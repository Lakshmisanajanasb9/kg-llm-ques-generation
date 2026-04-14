from __future__ import annotations

import json
from typing import Any, Dict, List


def path_to_text(path: Dict[str, Any]) -> str:
    return (
        f"{path.get('event', '')} "
        f"{path.get('relation1', '')} "
        f"{path.get('middle', '')} "
        f"{path.get('relation2', '')} "
        f"{path.get('related', '')}"
    ).strip()


def build_judge_prompt(topic: str, path: Dict[str, Any], question: str) -> str:
    path_text = path_to_text(path)

    return f"""
You are evaluating a generated analytical question.

TOPIC:
{topic}

SOURCE KNOWLEDGE GRAPH PATH:
{path_text}

GENERATED QUESTION:
{question}

Evaluate the question on two criteria using a 1-5 scale:

1. coherence
- Is the question clear, grammatical, and meaningful?

2. validity
- Is the question logically supported by the source path?
- Penalize hallucination, over-generalisation, or unsupported causal claims.

Return JSON only in exactly this format:
{{
  "coherence": 0,
  "validity": 0,
  "reason": ""
}}
""".strip()


def parse_judge_response(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return {
            "coherence": int(data.get("coherence", 0)),
            "validity": int(data.get("validity", 0)),
            "reason": str(data.get("reason", "")),
        }
    except Exception:
        return {
            "coherence": 0,
            "validity": 0,
            "reason": f"Failed to parse judge response: {text[:200]}",
        }


def average_judge_scores(judgments: List[Dict[str, Any]]) -> Dict[str, float]:
    if not judgments:
        return {
            "avg_coherence": 0.0,
            "avg_validity": 0.0,
        }

    coherence_vals = [j["coherence"] for j in judgments]
    validity_vals = [j["validity"] for j in judgments]

    return {
        "avg_coherence": sum(coherence_vals) / len(coherence_vals),
        "avg_validity": sum(validity_vals) / len(validity_vals),
    }