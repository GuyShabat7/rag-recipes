"""
metrics.py — Retrieval evaluation metrics.

Metrics implemented:
  - Recall@K     : fraction of relevant recipes appearing in top-K results
  - MRR          : Mean Reciprocal Rank — how high is the first relevant result?
  - NDCG@K       : Normalized Discounted Cumulative Gain — are top results most relevant?
  - Nutritional Precision@K : fraction of top-K results that satisfy the dietary constraint
                              (domain-specific metric, unique to this project)

All metrics are computed per-query and then macro-averaged across the eval set.

Interview talking point — why these metrics matter:
  Recall@K answers "will the user find what they need?"
  MRR answers "how quickly will they find it?"
  NDCG@K answers "are the best matches ranked highest?"
  Nutritional Precision@K answers "are the results actually safe for this user's condition?"
    — this last one is the most important for a health-oriented system.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """
    Recall@K = |retrieved_top_k ∩ relevant| / |relevant|

    Measures: of all relevant recipes, what fraction did we find in the top K?
    If relevant set is empty, returns 0.0.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_result

    Returns 0.0 if no relevant result appears in the retrieved list.
    MRR = mean of reciprocal rank across all queries.
    """
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list,
    relevance_scores: dict,  # {recipe_id: relevance_grade (e.g., 1–5 rating)}
    k: int,
) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    Uses graded relevance (e.g., user ratings 1–5) rather than binary.
    This rewards ranking higher-rated recipes above lower-rated ones.

    DCG@K  = Σ_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
    IDCG@K = DCG of the ideal (perfectly ranked) list
    """
    def dcg(ids, k):
        score = 0.0
        for i, rid in enumerate(ids[:k], 1):
            rel = relevance_scores.get(rid, 0)
            score += (2 ** rel - 1) / np.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids, k)
    # Ideal order: sort relevant items by descending grade
    ideal_ids = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    ideal_dcg = dcg(ideal_ids, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def nutritional_precision_at_k(
    retrieved_ids: list,
    recipe_flags_df: pd.DataFrame,
    constraint_flag: str,
    k: int,
) -> float:
    """
    Nutritional Precision@K = fraction of top-K retrieved results that
    actually satisfy the dietary constraint.

    This is a domain-specific metric designed for this project.
    It directly measures whether the system is safe for users with dietary
    requirements — e.g., if someone queries 'diabetic-friendly recipes',
    what fraction of the top-5 results are actually diabetic-friendly?

    Args:
        retrieved_ids: Ordered list of retrieved recipe IDs.
        recipe_flags_df: DataFrame indexed by recipe_id with boolean flag columns.
        constraint_flag: Dietary flag column name (e.g., 'is_diabetic_friendly').
        k: Number of top results to evaluate.
    """
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids or constraint_flag not in recipe_flags_df.columns:
        return 0.0

    valid_ids = [rid for rid in top_k_ids if rid in recipe_flags_df.index]
    if not valid_ids:
        return 0.0

    satisfying = recipe_flags_df.loc[valid_ids, constraint_flag].sum()
    return float(satisfying) / len(top_k_ids)


def evaluate_system(
    system_name: str,
    queries: list[dict],          # [{"query_id": ..., "query": ..., "constraints": [...]}]
    ground_truth: dict,           # {query_id: {"relevant_ids": [...], "relevance_scores": {...}}}
    retrieve_fn,                  # callable: (query, constraints) → list of recipe_ids
    recipe_flags_df: pd.DataFrame,
    k_values: list[int] = None,
) -> dict:
    """
    Run full evaluation of a retrieval system over all eval queries.

    Returns a dict with mean metric values across all queries:
        recall@5, recall@10, mrr, ndcg@10, nutritional_precision@5
    """
    if k_values is None:
        k_values = [5, 10]

    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["mrr"] = []
    results["nutritional_precision@5"] = []

    for q in queries:
        qid = q["query_id"]
        gt = ground_truth.get(qid, {})
        relevant_ids = set(gt.get("relevant_ids", []))
        relevance_scores = gt.get("relevance_scores", {})

        retrieved = retrieve_fn(q["query"], q.get("constraints", []))

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant_ids, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevance_scores, k))

        results["mrr"].append(reciprocal_rank(retrieved, relevant_ids))

        constraints = q.get("constraints", [])
        if constraints:
            nut_prec = nutritional_precision_at_k(
                retrieved, recipe_flags_df, constraints[0], k=5
            )
            results["nutritional_precision@5"].append(nut_prec)

    # Macro-average
    mean_results = {metric: float(np.mean(vals)) for metric, vals in results.items() if vals}
    logger.info(f"[{system_name}] " + " | ".join(f"{m}={v:.3f}" for m, v in mean_results.items()))

    return {"system": system_name, **mean_results}
