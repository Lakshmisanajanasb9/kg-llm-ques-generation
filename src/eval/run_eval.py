from __future__ import annotations

from typing import Any, Dict, List

from src.eval.metrics import summarize_embedding_metrics
from src.eval.results_io import save_json, save_summary_csv
from src.eval.sbert_eval import (
    compute_grounding_scores,
    compute_pairwise_question_similarities,
    find_redundant_pairs,
)
from src.kg.retrieve_context import get_context
from src.kg.wikidata_sparql import get_wikidata_context_sparql as get_wikidata_context
from src.rag.generate_questions import generate_questions, clean_questions
from src.rag.prompt_template import build_prompt
from src.rag.prompt.zero_shot import build_zero_prompt
from src.rag.prompt.frewShot import build_few_prompt
from src.rag.prompt.ChainOfThought import build_advanced_cot_prompt
from src.rag.prompt.dspPrompt import build_dsp_prompt

from src.eval.llm_runner import judge_questions_with_llm


def evaluate_question_set(
    method_name: str,
    topic: str,
    paths: List[Dict[str, Any]],
    questions: List[str],
    judge_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    grounding_scores = compute_grounding_scores(paths, questions)
    pairwise_similarities = compute_pairwise_question_similarities(questions)
    redundant_pairs = find_redundant_pairs(questions, threshold=0.85)

    summary = summarize_embedding_metrics(
        grounding_scores=grounding_scores,
        pairwise_similarities=pairwise_similarities,
        redundant_pairs=redundant_pairs,
    )
    if judge_result is not None:
        summary.update(judge_result["summary"])

    return {
        "method": method_name,
        "topic": topic,
        "num_paths": len(paths),
        "num_questions": len(questions),
        "questions": questions,
        "paths": paths,
        "grounding_scores": grounding_scores,
        "pairwise_similarities": pairwise_similarities,
        "redundant_pairs": redundant_pairs,
        "judge_result": judge_result,
        "summary": summary,
    }


def run_all_experiments() -> None:
    topic = "JPMorgan Chase"
    n = 5

    results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # ---------- LOCAL KG ----------
    raw_local = get_context(topic, limit=100, split="train")
    local_paths = raw_local[:10]

    methods = [
        ("zero_shot_findkg", build_prompt),
        ("few_shot_findkg", build_few_prompt),
        ("cot_findkg", build_advanced_cot_prompt),
        ("dsp_findkg", build_dsp_prompt),  # using zero prompt as a proxy for DSP-style since it emphasizes reasoning
    ]

    for name, builder in methods:
        if builder is build_prompt:
            prompt = builder(topic, local_paths, n=n)
        elif builder is build_advanced_cot_prompt:
            prompt = builder(local_paths, topic, n=n)
        elif builder is build_dsp_prompt:
            prompt = builder(local_paths, topic, n=n)
        else:
            # few-shot usually follows the same style as your other prompt builders
            prompt = builder(topic, local_paths, n=n)

        questions = clean_questions(generate_questions(prompt))[:n]

        result = evaluate_question_set(
            method_name=name,
            topic=topic,
            paths=local_paths,
            questions=questions,
        )
        results.append(result)

    # ---------- WIKIDATA ----------
    wiki_paths = get_wikidata_context(topic)

    methods = [
        ("zero_shot_wikidata", build_zero_prompt),
        ("few_shot_wikidata", build_few_prompt),
        ("cot_wikidata", build_advanced_cot_prompt),
    ]

    for name, builder in methods:
        if builder is build_advanced_cot_prompt:
            prompt = builder(wiki_paths, topic, n=n)
        else:
            prompt = builder(topic, wiki_paths, n=n)

        questions = clean_questions(generate_questions(prompt))[:n]

        result = evaluate_question_set(
            method_name=name,
            topic=topic,
            paths=wiki_paths,
            questions=questions,
        )
        results.append(result)

    # ---------- SUMMARY ----------
    for r in results:
        summary_rows.append({
            "method": r["method"],
            "topic": r["topic"],
            **r["summary"],
            "num_paths": r["num_paths"],
            "num_questions": r["num_questions"],
        })

    print("\n=== SUMMARY ===")
    for row in summary_rows:
        print(row)

    save_json({"results": results}, "outputs/eval/full_results.json")
    save_summary_csv(summary_rows, "outputs/eval/summary.csv")


if __name__ == "__main__":
    run_all_experiments()