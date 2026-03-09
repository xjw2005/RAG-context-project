from __future__ import annotations

from typing import Any, Dict, List, Tuple


def normalize_search_results(results: List[Tuple[float, str, str]]) -> Dict[str, Dict[str, Any]]:
    if not results:
        return {}

    scores = [score for score, _, _ in results]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    normalized: Dict[str, Dict[str, Any]] = {}
    for score, chunk_id, snippet in results:
        normalized[chunk_id] = {
            "score": (score - min_score) / score_range,
            "snippet": snippet,
        }
    return normalized


def merge_and_rank(
    vector_scores: Dict[str, Dict[str, Any]],
    keyword_scores: Dict[str, Dict[str, Any]],
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
    top_k: int = 3,
) -> List[Tuple[str, str]]:
    if abs(vector_weight + keyword_weight - 1) > 1e-6:
        raise ValueError("vector_weight + keyword_weight must be 1")

    all_chunk_ids = set(vector_scores) | set(keyword_scores)
    final_scores: Dict[str, Tuple[float, str]] = {}

    for chunk_id in all_chunk_ids:
        vector_score = vector_scores.get(chunk_id, {}).get("score", 0.0)
        keyword_score = keyword_scores.get(chunk_id, {}).get("score", 0.0)
        snippet = vector_scores.get(chunk_id, {}).get("snippet", "") or keyword_scores.get(chunk_id, {}).get("snippet", "")
        final_scores[chunk_id] = (
            vector_weight * vector_score + keyword_weight * keyword_score,
            snippet,
        )

    ranked = sorted(final_scores.items(), key=lambda item: item[1][0], reverse=True)
    return [(chunk_id, snippet) for chunk_id, (_, snippet) in ranked[:top_k]]
