from __future__ import annotations

from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer, util


_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def path_to_text(path: Dict[str, Any]) -> str:
    return (
        f"{path.get('event', '')} "
        f"{path.get('relation1', '')} "
        f"{path.get('middle', '')} "
        f"{path.get('relation2', '')} "
        f"{path.get('related', '')}"
    ).strip()


def compute_grounding_scores(
    paths: List[Dict[str, Any]],
    questions: List[str],
) -> List[float]:
    model = get_model()

    if not paths or not questions:
        return []

    n = min(len(paths), len(questions))
    path_texts = [path_to_text(p) for p in paths[:n]]
    qs = questions[:n]

    path_emb = model.encode(path_texts, convert_to_tensor=True)
    q_emb = model.encode(qs, convert_to_tensor=True)

    scores: List[float] = []
    for i in range(n):
        score = util.cos_sim(q_emb[i], path_emb[i]).item()
        scores.append(float(score))

    return scores


def compute_pairwise_question_similarities(
    questions: List[str],
) -> List[float]:
    model = get_model()

    if len(questions) < 2:
        return []

    q_emb = model.encode(questions, convert_to_tensor=True)
    sim_matrix = util.cos_sim(q_emb, q_emb)

    scores: List[float] = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            scores.append(float(sim_matrix[i][j].item()))
    return scores


def find_redundant_pairs(
    questions: List[str],
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    model = get_model()

    if len(questions) < 2:
        return []

    q_emb = model.encode(questions, convert_to_tensor=True)
    sim_matrix = util.cos_sim(q_emb, q_emb)

    redundant: List[Dict[str, Any]] = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = float(sim_matrix[i][j].item())
            if sim >= threshold:
                redundant.append(
                    {
                        "i": i,
                        "j": j,
                        "similarity": sim,
                        "q1": questions[i],
                        "q2": questions[j],
                    }
                )
    return redundant