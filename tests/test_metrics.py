"""Tests for retrieval evaluation metrics."""

import pandas as pd
import pytest
from src.evaluation.metrics import (
    ndcg_at_k,
    nutritional_precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --- Recall@K ---
def test_recall_perfect():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_recall_partial():
    result = recall_at_k([1, 2, 5, 6, 7], {1, 2, 3, 4}, k=5)
    assert result == 0.5  # found 2 of 4 relevant


def test_recall_zero():
    assert recall_at_k([5, 6, 7], {1, 2, 3}, k=3) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_recall_k_smaller_than_retrieved():
    # Only top-2 considered, only 1 relevant found
    assert recall_at_k([1, 5, 2], {1, 2}, k=2) == 0.5


# --- MRR ---
def test_mrr_first_position():
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0


def test_mrr_second_position():
    assert reciprocal_rank([5, 1, 2], {1}) == 0.5


def test_mrr_not_found():
    assert reciprocal_rank([5, 6, 7], {1}) == 0.0


def test_mrr_multiple_relevant_uses_first():
    # First relevant is at position 2
    assert reciprocal_rank([5, 1, 2], {1, 2}) == 0.5


# --- NDCG@K ---
def test_ndcg_perfect_ranking():
    # Ideal order is exactly the retrieved order
    relevance = {1: 5, 2: 4, 3: 3}
    score = ndcg_at_k([1, 2, 3], relevance, k=3)
    assert abs(score - 1.0) < 1e-6


def test_ndcg_zero_relevance():
    relevance = {1: 0, 2: 0}
    score = ndcg_at_k([1, 2], relevance, k=2)
    assert score == 0.0


def test_ndcg_reversed_ranking():
    # Worst item ranked first — should be less than 1.0
    relevance = {1: 1, 2: 5}
    score_good = ndcg_at_k([2, 1], relevance, k=2)
    score_bad = ndcg_at_k([1, 2], relevance, k=2)
    assert score_good > score_bad


# --- Nutritional Precision@K ---
def make_flags_df(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data).set_index("id")
    return df


def test_nutritional_precision_perfect():
    flags = make_flags_df({"id": [1, 2, 3], "is_diabetic_friendly": [True, True, True]})
    result = nutritional_precision_at_k([1, 2, 3], flags, "is_diabetic_friendly", k=3)
    assert result == 1.0


def test_nutritional_precision_zero():
    flags = make_flags_df({"id": [1, 2, 3], "is_diabetic_friendly": [False, False, False]})
    result = nutritional_precision_at_k([1, 2, 3], flags, "is_diabetic_friendly", k=3)
    assert result == 0.0


def test_nutritional_precision_partial():
    flags = make_flags_df({"id": [1, 2, 3, 4, 5], "is_high_protein": [True, False, True, False, False]})
    # Top 5: ids 1,2,3,4,5 — 2 of 5 satisfy
    result = nutritional_precision_at_k([1, 2, 3, 4, 5], flags, "is_high_protein", k=5)
    assert abs(result - 0.4) < 1e-6


def test_nutritional_precision_unknown_flag():
    flags = make_flags_df({"id": [1, 2], "is_diabetic_friendly": [True, True]})
    result = nutritional_precision_at_k([1, 2], flags, "is_nonexistent_flag", k=2)
    assert result == 0.0
