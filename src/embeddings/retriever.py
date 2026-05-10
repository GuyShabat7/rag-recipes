"""
retriever.py — Query → top-K recipe retrieval with hybrid re-ranking.

Pipeline per query (no dietary constraints):
  1. Encode query with same SentenceTransformer model used to build the index
  2. FAISS search → top-N semantic candidates (faiss_candidates from config)
  3. Re-rank using hybrid score
  4. Return top-K

Pipeline per query (with dietary constraints):
  1. Encode query
  2. Look up pre-computed compliant recipe positions for each flag (built at startup)
  3. Intersect positions across all active flags → compliant pool
  4. FAISS IDSelectorBatch search restricted to compliant pool → every result is
     guaranteed compliant; no post-filter or auto-relax needed
  5. Adaptive k: search min(pool_size, max(top_k * multiplier, faiss_candidates))
  6. Re-rank using hybrid score within the compliant pool
  7. Return top-K — or empty DataFrame if the compliant pool is empty

The IDSelectorBatch approach is the only architecture that guarantees strict
dietary compliance. Post-retrieval filtering (the prior approach) cannot
guarantee compliance because it filters a semantically-selected pool, not the
full corpus — if keto recipes are 5% of 223k, only ~2-3 land in top-50.

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
from typing import Optional

import faiss
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

        # Pre-compute FAISS index positions (0-based row number in the index)
        # for each dietary flag. recipe_ids[i] = recipe ID stored at FAISS position i,
        # so we invert the mapping to go recipe_id → position in O(1).
        self._recipe_id_to_position: dict[int, int] = {
            int(rid): pos for pos, rid in enumerate(self.recipe_ids)
        }
        self._flag_positions: dict[str, np.ndarray] = {}
        for flag in DIETARY_FLAG_COLS:
            if flag not in self.recipes_df.columns:
                continue
            compliant_ids = self.recipes_df.index[self.recipes_df[flag]].tolist()
            positions = np.array(
                [self._recipe_id_to_position[int(rid)]
                 for rid in compliant_ids
                 if int(rid) in self._recipe_id_to_position],
                dtype=np.int64,
            )
            self._flag_positions[flag] = positions
            logger.info(f"  {flag}: {len(positions):,} compliant recipes pre-indexed")

    def _get_compliant_positions(self, constraints: list[str]) -> np.ndarray:
        """
        Return FAISS index positions of recipes satisfying ALL active constraints.

        Intersects the pre-computed position arrays, starting from the smallest
        (most restrictive) set so each successive intersection is cheap.
        """
        valid_flags = [f for f in constraints if f in self._flag_positions]
        if not valid_flags:
            return np.arange(len(self.recipe_ids), dtype=np.int64)

        sorted_flags = sorted(valid_flags, key=lambda f: len(self._flag_positions[f]))
        compliant = set(self._flag_positions[sorted_flags[0]].tolist())
        for flag in sorted_flags[1:]:
            compliant &= set(self._flag_positions[flag].tolist())

        return np.array(sorted(compliant), dtype=np.int64)

    def _faiss_search(
        self,
        query_vec: np.ndarray,
        k: int,
        compliant_positions: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run FAISS search, optionally restricted to compliant_positions.

        When compliant_positions is provided, uses IDSelectorBatch so FAISS
        only considers those positions — every returned index is guaranteed
        compliant before re-ranking even begins.

        Falls back to unrestricted search + np.isin filter if the FAISS build
        predates search_with_parameters (FAISS < 1.7.4).

        Returns (semantic_scores, faiss_indices) both of length ≤ k.
        """
        if compliant_positions is None or len(compliant_positions) == 0:
            scores, indices = self.index.search(query_vec, k)
            return scores[0], indices[0]

        k_actual = min(k, len(compliant_positions))
        try:
            sel = faiss.IDSelectorBatch(compliant_positions)
            params = faiss.SearchParameters()
            params.sel = sel
            scores, indices = self.index.search_with_parameters(query_vec, k_actual, params)
            return scores[0], indices[0]
        except AttributeError:
            # FAISS version doesn't expose search_with_parameters — fall back to
            # unrestricted search followed by masking. Correctness is preserved;
            # only efficiency suffers at very large k.
            logger.debug("search_with_parameters unavailable; using post-search mask fallback")
            k_extended = min(self.index.ntotal, k_actual * 10)
            scores_all, indices_all = self.index.search(query_vec, k_extended)
            mask = np.isin(indices_all[0], compliant_positions)
            return scores_all[0][mask][:k_actual], indices_all[0][mask][:k_actual]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        dietary_constraints: Optional[list[str]] = None,
        user_ingredients: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Full retrieval pipeline for a single query.

        When dietary_constraints are active, FAISS search is restricted to the
        pre-computed compliant pool via IDSelectorBatch. Every returned recipe
        satisfies all constraints by construction — no post-filter, no auto-relax.

        Returns an empty DataFrame (not a constrained violation) when the
        compliant pool is empty for the requested constraint combination.

        Args:
            query: Natural-language user query.
            top_k: Number of results to return (default from config).
            dietary_constraints: Flag column names to enforce. Parsed from query
                text if None.
            user_ingredients: Optional ingredients the user has at home.

        Returns:
            DataFrame of top-K recipes sorted by hybrid score, or empty
            DataFrame if no compliant recipes exist.
        """
        cfg_r = self.cfg["retrieval"]
        k = top_k or cfg_r["top_k"]

        weights = (
            cfg_r["score_weights_with_ingredients"] if user_ingredients
            else cfg_r["score_weights"]
        )

        # Step 1: Parse dietary constraints
        if dietary_constraints is None:
            dietary_constraints = parse_dietary_constraints(query)
        logger.debug(f"Active dietary constraints: {dietary_constraints}")

        # Step 2: Encode query
        query_vec = self.encoder.encode_query(query)

        # Step 3: Determine search scope
        if dietary_constraints:
            compliant_positions = self._get_compliant_positions(dietary_constraints)
            pool_size = len(compliant_positions)
            logger.info(f"Compliant pool: {pool_size:,} recipes satisfy {dietary_constraints}")

            if pool_size == 0:
                logger.warning(
                    f"No recipes satisfy all constraints: {dietary_constraints}. "
                    "Returning empty result — constraints cannot be relaxed silently."
                )
                return pd.DataFrame()

            # Adaptive k: search deeper into the compliant pool so we surface
            # the best semantic matches, not just the first ones encountered.
            # Without this, a pool of 11k keto recipes with k=50 sees only 0.5%.
            multiplier = cfg_r.get("constrained_candidates_multiplier", 20)
            cap = cfg_r.get("max_constrained_candidates", 500)
            k_search = min(pool_size, min(max(k * multiplier, cfg_r["faiss_candidates"]), cap))
        else:
            compliant_positions = None
            k_search = cfg_r["faiss_candidates"]

        # Step 4: FAISS search (constrained or unconstrained)
        semantic_scores, faiss_indices = self._faiss_search(query_vec, k_search, compliant_positions)

        # Drop FAISS sentinel -1 (returned when the index has fewer rows than k)
        valid = faiss_indices >= 0
        semantic_scores = semantic_scores[valid]
        faiss_indices = faiss_indices[valid]

        # Step 5: Build candidate DataFrame
        candidate_recipe_ids = self.recipe_ids[faiss_indices]
        candidate_df = self.recipes_df.loc[
            self.recipes_df.index.isin(candidate_recipe_ids)
        ].copy()
        id_to_score = dict(zip(candidate_recipe_ids, semantic_scores))
        candidate_df["semantic_score"] = candidate_df.index.map(id_to_score)

        # Step 6: Nutritional score
        candidate_df["nutritional_score"] = candidate_df.apply(
            lambda row: score_recipe(row, dietary_constraints)[0], axis=1
        )

        # Step 7: Ingredient match score (only when user provides ingredients)
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

        # Step 8: Hybrid re-ranking
        pop_norm = candidate_df["log_review_count"].fillna(0)
        pop_max = max(float(pop_norm.max()), 1.0)
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

        return (
            candidate_df.sort_values("hybrid_score", ascending=False)
            .head(k)
            .reset_index()
            .rename(columns={"id": "recipe_id"})
        )
