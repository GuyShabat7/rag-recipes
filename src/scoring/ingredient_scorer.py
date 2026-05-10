"""
ingredient_scorer.py — Score recipes based on overlap with user's available ingredients.

Two scores are computed per recipe:

  match_score (0–1):
    Jaccard-style coverage — what fraction of the recipe's ingredients does the user already have?
    = |user_ingredients ∩ recipe_ingredients| / |recipe_ingredients|
    High score = user can mostly make this recipe without shopping.

  coverage_score (0–1):
    How many of the user's ingredients are actually used by this recipe?
    = |user_ingredients ∩ recipe_ingredients| / |user_ingredients|
    High score = this recipe makes good use of what the user has.

  combined_ingredient_score = 0.7 * match_score + 0.3 * coverage_score
    Weighted toward match_score because minimizing missing ingredients
    is the primary goal. coverage_score is secondary (we don't penalize
    a recipe for not needing every single ingredient the user listed).

Matching is fuzzy: "chicken breast" matches "chicken", "olive oil" matches "oil".
We tokenize and check for substring containment so partial matches count.

Interview talking point:
  This is a set-intersection problem. The naive approach is exact string matching,
  but food ingredient names are messy — "all-purpose flour" vs "flour", "garlic clove"
  vs "garlic". We solve this with token-level substring matching, which is fast and
  interpretable. A more sophisticated approach would use word embeddings to find
  semantically similar ingredients (e.g., "chicken thigh" ≈ "chicken breast"),
  but that adds latency for marginal gain.
"""

import re
from typing import Optional


def _tokenize(ingredient: str) -> set[str]:
    """Lowercase and split ingredient string into meaningful tokens, removing stopwords."""
    stopwords = {"a", "an", "the", "of", "and", "or", "with", "fresh", "dried",
                 "large", "small", "medium", "cup", "cups", "tbsp", "tsp",
                 "clove", "cloves", "slice", "slices", "pound", "oz", "g"}
    tokens = set(re.findall(r"[a-z]+", ingredient.lower()))
    return tokens - stopwords


def _ingredients_match(user_tokens: set[str], recipe_ingredient: str) -> bool:
    """
    Returns True if any user ingredient token appears as a substring in the recipe ingredient.

    Example:
        user has "chicken" → matches "chicken breast", "rotisserie chicken", "diced chicken"
        user has "oil"     → matches "olive oil", "vegetable oil", "coconut oil"
    """
    recipe_tokens = _tokenize(recipe_ingredient)
    # Match if any user token appears in the recipe ingredient's tokens
    return bool(user_tokens & recipe_tokens)


def score_ingredient_match(
    recipe_ingredients: list[str],
    user_ingredients: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Compute ingredient match score for a single recipe.

    Args:
        recipe_ingredients: List of ingredient strings for the recipe (from the dataset).
        user_ingredients:   List of ingredient strings the user currently has.

    Returns:
        (combined_score, matched_ingredients, missing_ingredients)
          combined_score       — float in [0, 1]
          matched_ingredients  — recipe ingredients the user already has
          missing_ingredients  — recipe ingredients the user needs to buy
    """
    if not recipe_ingredients or not user_ingredients:
        return 0.0, [], recipe_ingredients or []

    # Build per-user-ingredient token sets once
    user_token_sets = [_tokenize(u) for u in user_ingredients]

    matched = []
    missing = []

    for recipe_ing in recipe_ingredients:
        found = any(
            _ingredients_match(u_tokens, recipe_ing)
            for u_tokens in user_token_sets
        )
        if found:
            matched.append(recipe_ing)
        else:
            missing.append(recipe_ing)

    n_recipe = len(recipe_ingredients)
    n_user = len(user_ingredients)

    # match_score: fraction of recipe ingredients the user has
    match_score = len(matched) / n_recipe if n_recipe > 0 else 0.0

    # coverage_score: fraction of user's ingredients used by this recipe
    user_ingredients_used = sum(
        1 for u_tokens in user_token_sets
        if any(_ingredients_match(u_tokens, ri) for ri in recipe_ingredients)
    )
    coverage_score = user_ingredients_used / n_user if n_user > 0 else 0.0

    combined = 0.7 * match_score + 0.3 * coverage_score

    return combined, matched, missing


def score_dataframe_ingredient_match(
    recipes_df,
    user_ingredients: list[str],
    ingredients_col: str = "ingredients",
) -> tuple:
    """
    Compute ingredient match scores for all rows in a DataFrame.

    Returns three parallel lists: scores, matched_lists, missing_lists.
    Used in retriever.py to vectorize scoring across candidates.
    """
    scores, matched_lists, missing_lists = [], [], []

    for ingredients in recipes_df[ingredients_col]:
        if isinstance(ingredients, list):
            recipe_ings = ingredients
        elif isinstance(ingredients, str):
            # Handle case where ingredients are stored as comma-separated string
            recipe_ings = [i.strip() for i in ingredients.split(",")]
        else:
            recipe_ings = []

        score, matched, missing = score_ingredient_match(recipe_ings, user_ingredients)
        scores.append(score)
        matched_lists.append(matched)
        missing_lists.append(missing)

    return scores, matched_lists, missing_lists
