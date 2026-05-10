"""
retriever.py — Query → top-K recipe retrieval with hybrid re-ranking.

Pipeline per query:
  1. Encode query with same SentenceTransformer model used to build the index
  2. FAISS ANN search → top-50 semantic candidates
  3. Hard-filter by dietary flags (e.g., must be is_diabetic_friendly)
  4. Re-rank using hybrid score:
       Without ingredients: 0.6*semantic + 0.3*nutritional + 0.1*popularity
       With ingredients:    0.4*semantic + 0.25*nutritional + 0.25*ingredient_match + 0.1*popularity
  5. Return top-K results with scores and metadata

Dietary constraint keywords → flag mapping:
  'diabetic'        → is_diabetic_friendly
  'gluten free'     → is_gluten_free
  'high protein'    → is_high_protein
  'workout'         → is_high_protein
  'low fat'         → is_low_fat
  'low sodium'      → is_low_sodium
  'low calorie'     → is_low_calorie
  'keto'            → is_keto_friendly
  'low carb'        → is_low_carb
  'low sugar'       → is_low_sugar
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.embeddings.encoder import RecipeEncoder
from src.embeddings.indexer import load_index
from src.scoring.ingredient_scorer import score_dataframe_ingredient_match
from src.scoring.nutritional_scorer import score_recipe

logger = logging.getLogger(__name__)

# Maps query keywords to dietary flag column names
KEYWORD_TO_FLAG = {
    "diabetic": "is_diabetic_friendly",
    "diabetes": "is_diabetic_friendly",
    "gluten free": "is_gluten_free",
    "gluten-free": "is_gluten_free",
    "celiac": "is_gluten_free",
    "high protein": "is_high_protein",
    "protein": "is_high_protein",
    "workout": "is_high_protein",
    "athlete": "is_high_protein",
    "muscle": "is_high_protein",
    "low fat": "is_low_fat",
    "low-fat": "is_low_fat",
    "low sodium": "is_low_sodium",
    "low-sodium": "is_low_sodium",
    "heart": "is_low_sodium",
    "low calorie": "is_low_calorie",
    "low-calorie": "is_low_calorie",
    "diet": "is_low_calorie",
    "weight loss": "is_low_calorie",
    "keto": "is_keto_friendly",
    "ketogenic": "is_keto_friendly",
    "low carb": "is_low_carb",
    "low-carb": "is_low_carb",
    "low sugar": "is_low_sugar",
    "low-sugar": "is_low_sugar",
}

DIETARY_FLAG_COLS = [
    "is_diabetic_friendly",
    "is_gluten_free",
    "is_high_protein",
    "is_low_fat",
    "is_low_sodium",
    "is_low_calorie",
    "is_keto_friendly",
    "is_low_carb",
    "is_low_sugar",
]


def parse_dietary_constraints(query: str) -> list[str]:
    """
    Extract dietary flag column names from a natural-language query.

    Example:
        "I want a high protein gluten free dinner"
        → ["is_high_protein", "is_gluten_free"]
    """
    query_lower = query.lower()
    flags = set()
    for keyword, flag in KEYWORD_TO_FLAG.items():
        if keyword in query_lower:
            flags.add(flag)
    return list(flags)


class RecipeRetriever:
    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.encoder = RecipeEncoder(self.cfg["embedding"]["model_name"])
        self.index, self.recipe_ids = load_index(config_path)

        logger.info("Loading featured recipe metadata for re-ranking...")
        cols = ["id", "name", "calories", "protein_pdv", "sugar_pdv",
                "total_fat_pdv", "carbohydrates_pdv", "sodium_pdv",
                "log_review_count", "mean_rating", "ingredients"] + DIETARY_FLAG_COLS
        self.recipes_df = pd.read_parquet(
            self.cfg["data"]["featured"], columns=cols
        ).set_index("id")
        logger.info(f"Retriever ready. {len(self.recipes_df):,} recipes loaded.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        dietary_constraints: Optional[list[str]] = None,
        user_ingredients: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Full retrieval pipeline for a single query.

        Args:
            query: Natural-language user query.
            top_k: Number of results to return (default from config).
            dietary_constraints: List of flag column names to enforce as hard
                filters. If None, parsed automatically from query text.
            user_ingredients: Optional list of ingredients the user has at home.
                When provided, an ingredient_match_score is computed and the
                score weights shift to favour ingredient overlap.

        Returns:
            DataFrame of top-K recipes sorted by hybrid score. When
            user_ingredients is provided, also includes columns:
              ingredient_match_score, matched_ingredients, missing_ingredients
        """
        cfg_r = self.cfg["retrieval"]
        k = top_k or cfg_r["top_k"]
        candidates = cfg_r["faiss_candidates"]

        # Choose weight set based on whether ingredients were provided
        if user_ingredients:
            weights = cfg_r["score_weights_with_ingredients"]
        else:
            weights = cfg_r["score_weights"]

        # Step 1: Parse dietary constraints from query if not provided
        if dietary_constraints is None:
            dietary_constraints = parse_dietary_constraints(query)
        logger.debug(f"Active dietary constraints: {dietary_constraints}")

        # Step 2: Encode query and search FAISS
        query_vec = self.encoder.encode_query(query)
        scores, indices = self.index.search(query_vec, candidates)
        semantic_scores = scores[0]  # shape (candidates,)
        candidate_ids = self.recipe_ids[indices[0]]

        # Step 3: Build candidate DataFrame with semantic scores
        candidate_df = self.recipes_df.loc[
            self.recipes_df.index.isin(candidate_ids)
        ].copy()
        id_to_score = dict(zip(candidate_ids, semantic_scores))
        candidate_df["semantic_score"] = candidate_df.index.map(id_to_score)

        # Step 4: Hard-filter by dietary flags
        for flag in dietary_constraints:
            if flag in candidate_df.columns:
                candidate_df = candidate_df[candidate_df[flag]]

        if candidate_df.empty:
            logger.warning("No candidates survived dietary filter. Relaxing constraints.")
            candidate_df = self.recipes_df.loc[
                self.recipes_df.index.isin(candidate_ids)
            ].copy()
            candidate_df["semantic_score"] = candidate_df.index.map(id_to_score)

        # Step 5: Nutritional score for each candidate
        candidate_df["nutritional_score"] = candidate_df.apply(
            lambda row: score_recipe(row, dietary_constraints)[0], axis=1
        )

        # Step 6: Ingredient match score (only when user provides ingredients)
        if user_ingredients:
            ing_scores, matched_lists, missing_lists = score_dataframe_ingredient_match(
                candidate_df, user_ingredients, ingredients_col="ingredients"
            )
            candidate_df["ingredient_match_score"] = ing_scores
            candidate_df["matched_ingredients"] = matched_lists
            candidate_df["missing_ingredients"] = missing_lists
        else:
            candidate_df["ingredient_match_score"] = 0.0
            candidate_df["matched_ingredients"] = None
            candidate_df["missing_ingredients"] = None

        # Step 7: Hybrid re-ranking
        pop_norm = candidate_df["log_review_count"].fillna(0)
        pop_max = max(float(pop_norm.max()), 1.0)  # avoid division by zero
        pop_normalized = pop_norm / pop_max

        if user_ingredients:
            candidate_df["hybrid_score"] = (
                weights["semantic"] * candidate_df["semantic_score"]
                + weights["nutritional"] * candidate_df["nutritional_score"]
                + weights["ingredient_match"] * candidate_df["ingredient_match_score"]
                + weights["popularity"] * pop_normalized
            )
        else:
            candidate_df["hybrid_score"] = (
                weights["semantic"] * candidate_df["semantic_score"]
                + weights["nutritional"] * candidate_df["nutritional_score"]
                + weights["popularity"] * pop_normalized
            )

        result = (
            candidate_df.sort_values("hybrid_score", ascending=False)
            .head(k)
            .reset_index()
            .rename(columns={"id": "recipe_id"})
        )

        return result
