from typing import Callable, Dict, Any, List

from src.eval.llm_judge import (
    build_judge_prompt,
    parse_judge_response,
    average_judge_scores,
)


def judge_questions_with_llm(
    topic: str,
    paths: List[Dict[str, Any]],
    questions: List[str],
    generate_text_fn: Callable[[str], str],
) -> Dict[str, Any]:
    judgments: List[Dict[str, Any]] = []

    n = min(len(paths), len(questions))
    for i in range(n):
        judge_prompt = build_judge_prompt(topic, paths[i], questions[i])
        raw_response = generate_text_fn(judge_prompt)
        parsed = parse_judge_response(raw_response)

        judgments.append({
            "index": i,
            "question": questions[i],
            "path": paths[i],
            **parsed,
        })

    summary = average_judge_scores(judgments)

    return {
        "judgments": judgments,
        "summary": summary,
    }