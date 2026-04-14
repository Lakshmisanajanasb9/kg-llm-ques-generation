from __future__ import annotations

from typing import Dict, List


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_embedding_metrics(
    grounding_scores: List[float],
    pairwise_similarities: List[float],
    redundant_pairs: List[dict],
) -> Dict[str, float]:
    return {
        "avg_grounding": safe_mean(grounding_scores),
        "avg_pairwise_similarity": safe_mean(pairwise_similarities),
        "redundant_pair_count": float(len(redundant_pairs)),
    }