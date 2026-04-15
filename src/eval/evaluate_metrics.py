import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.kg.retrieve_context import get_context
from src.kg.neo4j_loader import insert_wiki_rows, query_wiki_paths, clear_wiki_subgraph
from src.kg.wikidata_sparql import get_wikidata_context_sparql

from src.rag.generate_questions import (
    generate_questions,
    generate_one_per_path,
    simplify_context,
    dedupe_paths,
    remove_bad_paths,
    dedupe_middle_related,
    limit_middle_reuse,
    dedupe_semantic_paths,
    select_diverse_paths,
    clean_questions,
)
from src.rag.bootstrap_pipeline import (
    bootstrap_zero_shot_examples,
    load_top_bootstrap_examples,
)

from src.rag.prompt.zero_shot import build_zero_prompt
from src.rag.prompt.frewShot import build_few_prompt
from src.rag.prompt.chain_prompt import build_cot_prompt
from src.rag.prompt.dspPrompt import build_dsp_prompt
from src.rag.prompt.judgePrompt import build_judge_prompt


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def path_to_text(path: dict) -> str:
    return (
        f"{path.get('event')} --{path.get('relation1')}--> {path.get('middle')} "
        f"--{path.get('relation2')}--> {path.get('related')}"
    )


def rows_to_context(rows: list[dict]) -> str:
    return "\n".join(path_to_text(r) for r in rows)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([])
    return embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def average_grounding_score(questions: list[str], paths: list[dict]) -> float:
    """
    For each question:
    - compute cosine similarity against all path texts
    - keep the best-matching path score
    - average across questions
    """
    if not questions or not paths:
        return 0.0

    q_emb = embed_texts(questions)
    p_emb = embed_texts([path_to_text(p) for p in paths])

    sims = cosine_similarity(q_emb, p_emb)
    best_per_question = sims.max(axis=1)
    return float(best_per_question.mean())


def question_path_grounding_scores(questions: list[str], paths: list[dict]) -> list[float]:
    if not questions or not paths:
        return []

    q_emb = embed_texts(questions)
    p_emb = embed_texts([path_to_text(p) for p in paths])

    sims = cosine_similarity(q_emb, p_emb)
    return [float(x) for x in sims.max(axis=1)]


def average_pairwise_similarity(questions: list[str]) -> float:
    if len(questions) < 2:
        return 0.0

    q_emb = embed_texts(questions)
    sims = cosine_similarity(q_emb, q_emb)

    vals = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            vals.append(float(sims[i][j]))

    return float(np.mean(vals)) if vals else 0.0


def redundant_pair_count(questions: list[str], threshold: float = 0.85) -> int:
    if len(questions) < 2:
        return 0

    q_emb = embed_texts(questions)
    sims = cosine_similarity(q_emb, q_emb)

    count = 0
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if sims[i][j] >= threshold:
                count += 1
    return count


def acceptance_rate(grounding_scores: list[float], threshold: float = 0.60) -> float:
    if not grounding_scores:
        return 0.0
    accepted = sum(1 for s in grounding_scores if s >= threshold)
    return float(accepted / len(grounding_scores))


def parse_judge_score(text: str) -> int:
    """
    Expects something like '6/7' somewhere in the output.
    """
    import re

    match = re.search(r"(\d+)\s*/\s*7", text)
    if match:
        return int(match.group(1))
    return 0


def llm_judge_scores(
    questions: list[str],
    paths: list[dict],
    llm_generate_fn: Callable[[str], list[str] | str],
) -> list[int]:
    """
    Pair each question with its best-matching path using SBERT,
    then judge with your existing judge prompt.
    """
    if not questions or not paths:
        return []

    path_texts = [path_to_text(p) for p in paths]
    q_emb = embed_texts(questions)
    p_emb = embed_texts(path_texts)
    sims = cosine_similarity(q_emb, p_emb)

    scores = []
    for i, q in enumerate(questions):
        best_path_idx = int(np.argmax(sims[i]))
        best_path = paths[best_path_idx]

        prompt = build_judge_prompt(q, best_path)
        result = llm_generate_fn(prompt)

        if isinstance(result, list):
            judge_text = " ".join(result)
        else:
            judge_text = str(result)

        scores.append(parse_judge_score(judge_text))

    return scores


def prepare_local_rows(topic: str, target_k: int = 10) -> list[dict]:
    raw_rows = get_context(topic, limit=200, split="train")
    rows = simplify_context(raw_rows)
    rows = dedupe_paths(rows)
    rows = remove_bad_paths(rows)
    rows = dedupe_middle_related(rows)
    rows = limit_middle_reuse(rows, max_per_middle=2)
    rows = dedupe_semantic_paths(rows, threshold=0.90)
    rows = select_diverse_paths(rows, final_k=max(target_k, 10))
    return rows[:target_k]


def prepare_wikidata_rows(topic: str, target_k: int = 10) -> list[dict]:
    wiki_rows = get_wikidata_context_sparql(topic, max_paths=max(target_k, 15), sparql_limit=300)
    if not wiki_rows:
        return []

    clear_wiki_subgraph()
    insert_wiki_rows(wiki_rows)
    structured_rows = query_wiki_paths(limit=300)

    if not structured_rows:
        cleaned = []
        for r in wiki_rows[:target_k]:
            cleaned.append(
                {
                    "event": r.get("event"),
                    "relation1": r.get("relation1"),
                    "middle": r.get("middle"),
                    "relation2": r.get("relation2"),
                    "related": r.get("related"),
                }
            )
        return cleaned

    structured_rows = dedupe_paths(structured_rows)
    structured_rows = remove_bad_paths(structured_rows)
    structured_rows = dedupe_middle_related(structured_rows)
    structured_rows = limit_middle_reuse(structured_rows, max_per_middle=1)

    cleaned = []
    for r in structured_rows[:target_k]:
        cleaned.append(
            {
                "event": r.get("event"),
                "relation1": r.get("relation1"),
                "middle": r.get("middle"),
                "relation2": r.get("relation2"),
                "related": r.get("related"),
            }
        )

    return cleaned


def generate_zero_shot(topic: str, rows: list[dict], n: int) -> list[str]:
    if not rows:
        return []

    context = rows_to_context(rows)
    prompt = build_zero_prompt(context=context, topic=topic, n=n)
    qs = generate_questions(prompt)
    qs = clean_questions(qs)
    return qs[:n]


def generate_few_shot(topic: str, rows: list[dict], n: int) -> list[str]:
    if not rows:
        return []

    csv_path = "outputs/bootstrap_examples.csv"

    bootstrap_zero_shot_examples(
        topic=topic,
        paths=rows,
        csv_path=csv_path,
        candidates_per_run=2,
        max_examples=5,
    )

    examples = load_top_bootstrap_examples(
        csv_path=csv_path,
        topic=topic,
        k=3,
    )

    prompt = build_few_prompt(
        event_keyword=topic,
        context_rows=rows,
        examples=examples,
        n=n,
    )

    qs = generate_questions(prompt)
    qs = clean_questions(qs)
    return qs[:n]


def generate_cot(topic: str, rows: list[dict], n: int) -> list[str]:
    if not rows:
        return []

    prompt = build_cot_prompt(topic, rows, n=n)
    qs = generate_questions(prompt)
    qs = clean_questions(qs)
    return qs[:n]


def generate_dsp(topic: str, rows: list[dict], n: int) -> list[str]:
    if not rows:
        return []

    prompt = build_dsp_prompt(topic, rows, n=n)
    qs = generate_questions(prompt)
    qs = clean_questions(qs)
    return qs[:n]


def generate_wiki_one_per_path(topic: str, rows: list[dict], n: int) -> list[str]:
    if not rows:
        return []

    qs = generate_one_per_path(topic, rows, n)
    qs = clean_questions(qs)
    return qs[:n]


def method_summary(
    method_name: str,
    topic: str,
    questions: list[str],
    paths: list[dict],
    use_llm_judge: bool = False,
) -> dict:
    grounding_scores = question_path_grounding_scores(questions, paths)

    summary = {
        "method": method_name,
        "topic": topic,
        "avg_grounding": float(np.mean(grounding_scores)) if grounding_scores else 0.0,
        "avg_pairwise_similarity": average_pairwise_similarity(questions),
        "redundant_pair_count": float(redundant_pair_count(questions)),
        "acceptance_rate": acceptance_rate(grounding_scores, threshold=0.60),
        "num_paths": len(paths),
        "num_questions": len(questions),
    }

    if use_llm_judge:
        judge_scores = llm_judge_scores(questions, paths, generate_questions)
        summary["avg_judge_score"] = float(np.mean(judge_scores)) if judge_scores else 0.0
    else:
        summary["avg_judge_score"] = 0.0

    return summary


def evaluate_topic(
    topic: str,
    n_questions: int = 10,
    target_paths: int = 15,
    use_llm_judge: bool = False,
) -> tuple[list[dict], list[dict]]:
    local_rows = prepare_local_rows(topic, target_k=target_paths)
    wiki_rows = prepare_wikidata_rows(topic, target_k=target_paths)

    summaries = []
    detailed_rows = []

    # Local KG
    local_zero = generate_zero_shot(topic, local_rows, n_questions)
    summaries.append(method_summary("zero_shot_findkg", topic, local_zero, local_rows, use_llm_judge))
    for q in local_zero:
        detailed_rows.append({"method": "zero_shot_findkg", "topic": topic, "question": q})

    local_few = generate_few_shot(topic, local_rows, n_questions)
    summaries.append(method_summary("few_shot_findkg", topic, local_few, local_rows, use_llm_judge))
    for q in local_few:
        detailed_rows.append({"method": "few_shot_findkg", "topic": topic, "question": q})

    local_cot = generate_cot(topic, local_rows, n_questions)
    summaries.append(method_summary("cot_findkg", topic, local_cot, local_rows, use_llm_judge))
    for q in local_cot:
        detailed_rows.append({"method": "cot_findkg", "topic": topic, "question": q})

    local_dsp = generate_dsp(topic, local_rows, n_questions)
    summaries.append(method_summary("dsp_findkg", topic, local_dsp, local_rows, use_llm_judge))
    for q in local_dsp:
        detailed_rows.append({"method": "dsp_findkg", "topic": topic, "question": q})

    # Wikidata
    if wiki_rows:
        wiki_zero = generate_zero_shot(topic, wiki_rows, n_questions)
        summaries.append(method_summary("zero_shot_wikidata", topic, wiki_zero, wiki_rows, use_llm_judge))
        for q in wiki_zero:
            detailed_rows.append({"method": "zero_shot_wikidata", "topic": topic, "question": q})

        wiki_few = generate_few_shot(topic, wiki_rows, n_questions)
        summaries.append(method_summary("few_shot_wikidata", topic, wiki_few, wiki_rows, use_llm_judge))
        for q in wiki_few:
            detailed_rows.append({"method": "few_shot_wikidata", "topic": topic, "question": q})

        wiki_cot = generate_cot(topic, wiki_rows, n_questions)
        summaries.append(method_summary("cot_wikidata", topic, wiki_cot, wiki_rows, use_llm_judge))
        for q in wiki_cot:
            detailed_rows.append({"method": "cot_wikidata", "topic": topic, "question": q})

        wiki_one = generate_wiki_one_per_path(topic, wiki_rows, n_questions)
        summaries.append(method_summary("one_per_path_wikidata", topic, wiki_one, wiki_rows, use_llm_judge))
        for q in wiki_one:
            detailed_rows.append({"method": "one_per_path_wikidata", "topic": topic, "question": q})

    return summaries, detailed_rows


def save_summary_csv(rows: list[dict], filename: str) -> None:
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "topic",
                "avg_grounding",
                "avg_pairwise_similarity",
                "redundant_pair_count",
                "acceptance_rate",
                "avg_judge_score",
                "num_paths",
                "num_questions",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_detailed_csv(rows: list[dict], filename: str) -> None:
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "topic", "question"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    topics = [
        "European Central Bank",
        "Inflation",
        "JPMorgan Chase",
    ]

    all_summaries = []
    all_detailed = []

    for topic in topics:
        print(f"\n=== Evaluating: {topic} ===")
        summaries, detailed = evaluate_topic(
            topic=topic,
            n_questions=10,
            target_paths=15,
            use_llm_judge=False,   # change to True if you want judge scoring too
        )

        for row in summaries:
            print(row)

        all_summaries.extend(summaries)
        all_detailed.extend(detailed)

    save_summary_csv(all_summaries, "outputs/evaluation_summary.csv")
    save_detailed_csv(all_detailed, "outputs/evaluation_questions.csv")

    Path("outputs/evaluation_summary.json").write_text(
        json.dumps(all_summaries, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print("- outputs/evaluation_summary.csv")
    print("- outputs/evaluation_questions.csv")
    print("- outputs/evaluation_summary.json")


if __name__ == "__main__":
    main()