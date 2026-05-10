"""
feature_engineering.py — Derive all features used by retrieval, scoring, and evaluation.

Three feature groups:
  1. Dietary flags      — boolean metadata for hard-filtering at retrieval time
  2. Numeric features   — engineered ratios and normalized values for the scoring model
  3. Composite text     — single string per recipe for embedding generation

Dietary profiles supported:
  - Diabetic-friendly  : low glycemic load (low sugar + low carbs combined)
  - Workout / Athlete  : high protein
  - Gluten-free        : no wheat/barley/rye in ingredients (ingredient-level check)
  - Low-fat            : low total fat
  - Low-sodium         : low sodium (heart health)
  - Low-calorie        : under 400 kcal (weight loss)
  - Keto-friendly      : very low carbs + high fat
  - Low-carb           : moderate carb restriction

Thresholds are based on FDA % Daily Value standards where applicable.
Gluten-free uses ingredient keyword matching (cannot be derived from nutrition vector).
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gluten detection helpers — shared logic with src/scoring/validator.py
# ---------------------------------------------------------------------------

# Complete ingredient names (lowercased) that contain hidden gluten.
# These are sources the nutrition vector cannot reveal — gluten comes from
# derived ingredients (soy sauce from wheat, malt vinegar from barley, etc.)
_GLUTEN_EXACT = frozenset({
    "soy sauce", "dark soy sauce", "light soy sauce",
    "low sodium soy sauce", "reduced sodium soy sauce",
    "teriyaki sauce", "teriyaki marinade", "hoisin sauce",
    "malt vinegar", "malt extract", "malt flavoring", "malted milk",
    "beer", "ale", "lager", "stout", "porter", "wheat beer",
    "worcestershire sauce",
    "panko", "panko breadcrumbs", "panko bread crumbs",
    "udon", "udon noodles",
    "ramen", "ramen noodles",
    "lo mein noodles", "chow mein noodles",
    "soba noodles",
    "egg noodles",
    "pasta",
    "pita", "pita bread",
    "naan", "naan bread",
    "phyllo", "phyllo dough", "filo", "filo dough",
    "puff pastry",
    "graham crackers",
    "pretzels",
    "saltine crackers", "saltines",
    "croutons",
    "flour tortillas",
})

# Words indicating gluten when present as a standalone token in an ingredient.
# Word-level tokenization prevents "buckwheat" → "wheat" false-positive and
# "almond flour" handled separately via _GF_FLOUR_MODIFIERS.
_GLUTEN_WORDS = frozenset({
    "wheat", "flour", "barley", "rye", "spelt", "farro", "bulgur",
    "durum", "semolina", "triticale", "kamut", "einkorn",
    "matzo", "matzoh", "couscous", "orzo", "breadcrumbs",
})

# When "flour" is the only gluten word and one of these modifiers is also
# present, the ingredient is a gluten-free flour (e.g., almond flour).
_GF_FLOUR_MODIFIERS = frozenset({
    "almond", "rice", "buckwheat", "coconut", "chickpea", "corn",
    "potato", "tapioca", "arrowroot", "amaranth", "teff", "sorghum",
    "cassava", "tigernut", "chestnut", "oat",
})


def _ingredient_contains_gluten(ingredient: str) -> bool:
    ing = ingredient.lower().strip()
    if ing in _GLUTEN_EXACT:
        return True
    words = frozenset(re.split(r"[\s\-]+", ing))
    gluten_hits = words & _GLUTEN_WORDS
    if not gluten_hits:
        return False
    if gluten_hits == {"flour"} and words & _GF_FLOUR_MODIFIERS:
        return False
    return True


# ---------------------------------------------------------------------------
# Group 1: Dietary flag features
# ---------------------------------------------------------------------------

def add_dietary_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add boolean dietary suitability flags to each recipe.

    These are used as hard metadata filters at retrieval time so that
    e.g. a query for 'diabetic-friendly dinner' only considers recipes
    where is_diabetic_friendly == True, before semantic re-ranking.
    """
    f = cfg["nutritional_flags"]

    # --- Diabetic-friendly ---
    # Combines low sugar AND low carbs into a single glycemic proxy score.
    # More nuanced than sugar alone — relevant for diabetes management.
    # Formula: weighted sum of sugar and carbs PDV, both must be low.
    df["glycemic_proxy"] = df["sugar_pdv"] * 0.6 + df["carbohydrates_pdv"] * 0.4
    df["is_diabetic_friendly"] = df["glycemic_proxy"] < 8

    # --- High protein (workout / athlete) ---
    df["is_high_protein"] = df["protein_pdv"] >= f["high_protein_pdv"]

    # --- Gluten-free ---
    # Ingredient-level check — cannot be inferred from the nutrition vector.
    # Checks each ingredient in the list individually (not the joined string)
    # to avoid substring false-positives ("buckwheat"→"wheat", "almond flour"→"flour")
    # and catches hidden gluten sources (soy sauce, malt vinegar, beer, udon, etc.)
    df["is_gluten_free"] = df["ingredients"].apply(
        lambda lst: not any(_ingredient_contains_gluten(i) for i in lst)
    )

    # --- Low-fat ---
    df["is_low_fat"] = df["total_fat_pdv"] < f["low_fat_pdv"]

    # --- Low-sodium (heart health) ---
    df["is_low_sodium"] = df["sodium_pdv"] < f["low_sodium_pdv"]

    # --- Low-calorie (weight loss) ---
    df["is_low_calorie"] = df["calories"] < f["low_calorie_kcal"]

    # --- Keto-friendly ---
    df["is_keto_friendly"] = (
        (df["carbohydrates_pdv"] < f["keto_max_carbs_pdv"])
        & (df["total_fat_pdv"] >= f["keto_min_fat_pdv"])
    )

    # --- Low-carb (general) ---
    df["is_low_carb"] = df["carbohydrates_pdv"] < f["low_carb_pdv"]

    # --- Low-sugar (standalone) ---
    df["is_low_sugar"] = df["sugar_pdv"] < f["low_sugar_pdv"]

    n_diabetic = df["is_diabetic_friendly"].sum()
    n_gf = df["is_gluten_free"].sum()
    n_hp = df["is_high_protein"].sum()
    logger.info(
        f"Dietary flags: diabetic_friendly={n_diabetic:,} | "
        f"gluten_free={n_gf:,} | high_protein={n_hp:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Group 2: Numeric engineered features
# ---------------------------------------------------------------------------

def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineered numeric features for the nutritional scoring model.

    protein_to_calorie_ratio — nutrient density signal:
        How much protein (% DV) per unit of caloric load.
        High values mean protein-dense relative to calories.

    sugar_to_carb_ratio — refined carbohydrate signal:
        What fraction of total carbs are sugars.
        High values indicate simple/refined carbs, bad for diabetics.

    log_minutes — log-transform of prep time:
        Prep time is heavily right-skewed; log1p normalizes it.

    complexity_score — weighted combination of recipe complexity signals:
        Useful as a feature for recommending "quick easy" vs "elaborate" recipes.
    """
    # Nutrient density
    df["protein_to_calorie_ratio"] = df["protein_pdv"] / (df["calories"].clip(lower=1) / 2000)

    # Refined carb signal
    df["sugar_to_carb_ratio"] = df["sugar_pdv"] / (df["carbohydrates_pdv"] + 1e-5)

    # Log-transform skewed fields
    df["log_minutes"] = np.log1p(df["minutes"])
    df["log_review_count"] = np.log1p(df["review_count"].fillna(0))

    # Complexity score (0–1 normalized)
    raw_complexity = (
        0.4 * df["n_steps"]
        + 0.3 * df["n_ingredients"]
        + 0.3 * df["log_minutes"]
    )
    df["complexity_score"] = (raw_complexity - raw_complexity.min()) / (
        raw_complexity.max() - raw_complexity.min() + 1e-8
    )

    return df


# ---------------------------------------------------------------------------
# Group 3: Composite text for embedding
# ---------------------------------------------------------------------------

def add_composite_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build recipe_text — the single string passed to the embedding model.

    Combines: name + description + ingredients + tags + first 2 steps.
    Including the first 2 steps captures cooking method (e.g., 'grilled',
    'baked', 'raw') which is semantically meaningful for queries like
    'no-cook high-protein lunch'.

    We include only the first 2 steps (not all) to keep token count reasonable
    while retaining method context.
    """
    def build_text(row):
        parts = [row["name"]]
        if row["description"]:
            parts.append(row["description"])
        parts.append(f"ingredients: {row['ingredients_str']}")
        if row["tags_str"]:
            parts.append(f"tags: {row['tags_str']}")
        # First 2 steps for cooking method context
        first_steps = row["steps"][:2] if isinstance(row["steps"], list) else []
        if first_steps:
            parts.append(f"method: {' '.join(first_steps)}")
        return ". ".join(parts)

    df["recipe_text"] = df.apply(build_text, axis=1)
    avg_len = df["recipe_text"].str.len().mean()
    logger.info(f"Built recipe_text field. Average length: {avg_len:.0f} chars")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    cleaned_path: str = None,
    output_path: str = None,
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Reads cleaned parquet, adds all feature groups, saves featured parquet.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    in_path = cleaned_path or cfg["data"]["cleaned"]
    out_path = output_path or cfg["data"]["featured"]

    logger.info(f"Loading cleaned data from {in_path}")
    df = pd.read_parquet(in_path)
    logger.info(f"Loaded {len(df):,} recipes")

    df = add_dietary_flags(df, cfg)
    df = add_numeric_features(df)
    df = add_composite_text(df)

    df.to_parquet(out_path, index=False)
    logger.info(f"Saved featured data to {out_path} ({len(df):,} rows, {len(df.columns)} columns)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engineer_features()
