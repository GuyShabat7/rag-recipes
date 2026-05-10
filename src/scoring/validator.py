"""
validator.py — Runtime dietary constraint re-validation.

Acts as Layer 2 in the dietary compliance stack:

  Layer 1 — IDSelectorBatch restricts FAISS search to the pre-computed
             compliant pool. Bounded by flag accuracy at parquet-build time.
  Layer 2 — This module re-applies threshold logic against live column values
             and re-checks gluten using an expanded keyword set. Catches:
               (a) Stale flags: parquet built with old config thresholds
               (b) Hidden gluten false-negatives: soy sauce, malt vinegar,
                   beer, udon etc. that the feature-engineering keyword set
                   missed but are now in the compliant pool
  Layer 3 — LLM narration in generator.py acts as a soft semantic guardrail.

This validator runs on at most max_constrained_candidates (default 500)
rows, all vectorized. It does not fix false-positives (buckwheat, almond
flour wrongly excluded from the GF pool) — those require a parquet rebuild
with the corrected feature_engineering.py. It only removes false-negatives
that slipped into the compliant pool.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gluten detection — expanded to cover hidden/derived gluten sources
# ---------------------------------------------------------------------------

# Complete ingredient names (lowercased) that contain hidden gluten.
# Soy sauce is the most impactful gap in the original set — it appears in
# thousands of Asian recipes and almost always contains wheat.
_GLUTEN_EXACT = frozenset({
    # Soy and derived sauces (typically wheat-thickened)
    "soy sauce", "dark soy sauce", "light soy sauce",
    "low sodium soy sauce", "reduced sodium soy sauce",
    "teriyaki sauce", "teriyaki marinade", "hoisin sauce",
    # Malt products (barley-derived)
    "malt vinegar", "malt extract", "malt flavoring", "malted milk",
    # Beer / ale (barley and often wheat)
    "beer", "ale", "lager", "stout", "porter", "wheat beer",
    # Sauces that typically contain malt vinegar or soy sauce
    "worcestershire sauce",
    # Japanese breadcrumbs and noodles
    "panko", "panko breadcrumbs", "panko bread crumbs",
    "udon", "udon noodles",
    "ramen", "ramen noodles",
    "lo mein noodles", "chow mein noodles",
    "soba noodles",   # traditionally buckwheat but commercial blends contain wheat
    "egg noodles",    # wheat-based
    "pasta",          # generic; GF versions are labelled "gluten-free pasta"
    # Bread products not caught by "flour" or "breadcrumbs"
    "pita", "pita bread",
    "naan", "naan bread",
    "phyllo", "phyllo dough", "filo", "filo dough",
    "puff pastry",
    "graham crackers",
    "pretzels",
    "saltine crackers", "saltines",
    "croutons",
    "flour tortillas",  # corn tortillas are GF; generic "tortillas" are wheat
})

# Words that, when they appear as a standalone token in an ingredient name,
# indicate gluten. Word-level tokenization (not substring) prevents:
#   "buckwheat" → "wheat" substring → false positive
#   "rice flour" → but "flour" as a WORD does flag gluten (correct)
_GLUTEN_WORDS = frozenset({
    "wheat", "flour", "barley", "rye", "spelt", "farro", "bulgur",
    "durum", "semolina", "triticale", "kamut", "einkorn",
    "matzo", "matzoh", "couscous", "orzo", "breadcrumbs",
})

# Modifier words that make a flour safe despite containing the word "flour".
# "almond flour" → "flour" is present BUT "almond" is in this set → GF.
# "all-purpose flour" → "all" and "purpose" are NOT in this set → not GF.
_GF_FLOUR_MODIFIERS = frozenset({
    "almond", "rice", "buckwheat", "coconut", "chickpea", "corn",
    "potato", "tapioca", "arrowroot", "amaranth", "teff", "sorghum",
    "cassava", "tigernut", "chestnut", "oat",
})


def _ingredient_contains_gluten(ingredient: str) -> bool:
    """
    Check a single ingredient string for gluten.

    Two-pass approach:
      1. Exact match against known hidden-gluten ingredient names.
      2. Word-level tokenization against the GLUTEN_WORDS set, with a
         GF-modifier exemption for recognised gluten-free flours.
    """
    ing = ingredient.lower().strip()

    if ing in _GLUTEN_EXACT:
        return True

    # Tokenise on whitespace and hyphens to avoid substring false-positives
    # ("wheat" in "buckwheat" is True as a substring, False as a word token)
    words = frozenset(re.split(r"[\s\-]+", ing))
    gluten_hits = words & _GLUTEN_WORDS

    if not gluten_hits:
        return False

    # "flour" is the only gluten hit AND a known GF modifier is present
    # → this is a GF flour (e.g., almond flour, rice flour, buckwheat flour)
    if gluten_hits == {"flour"} and words & _GF_FLOUR_MODIFIERS:
        return False

    return True


def _has_gluten(ingredients) -> bool:
    """Check a recipe's ingredient collection for any gluten source."""
    if isinstance(ingredients, list):
        return any(_ingredient_contains_gluten(i) for i in ingredients)
    # Fallback: ingredients stored as comma-separated string
    return any(
        _ingredient_contains_gluten(i.strip())
        for i in str(ingredients).split(",")
    )


# ---------------------------------------------------------------------------
# Runtime threshold re-validation
# ---------------------------------------------------------------------------

def _satisfies_constraint(row: pd.Series, flag: str, cfg: dict) -> bool:
    """
    Re-apply the threshold logic for one constraint against live column values.

    This is intentionally identical to the logic in feature_engineering.py.
    If config thresholds changed after the parquet was built, this catches
    recipes whose stored flag is stale.

    Returns False for NaN nutritional values — missing data is not compliant.
    """
    f = cfg.get("nutritional_flags", {})

    try:
        if flag == "is_high_protein":
            return float(row["protein_pdv"]) >= f.get("high_protein_pdv", 20)

        if flag == "is_low_fat":
            return float(row["total_fat_pdv"]) < f.get("low_fat_pdv", 10)

        if flag == "is_low_sodium":
            return float(row["sodium_pdv"]) < f.get("low_sodium_pdv", 10)

        if flag == "is_low_calorie":
            return float(row["calories"]) < f.get("low_calorie_kcal", 400)

        if flag == "is_keto_friendly":
            return (
                float(row["carbohydrates_pdv"]) < f.get("keto_max_carbs_pdv", 10)
                and float(row["total_fat_pdv"]) >= f.get("keto_min_fat_pdv", 25)
            )

        if flag == "is_low_carb":
            return float(row["carbohydrates_pdv"]) < f.get("low_carb_pdv", 15)

        if flag == "is_low_sugar":
            return float(row["sugar_pdv"]) < f.get("low_sugar_pdv", 5)

        if flag == "is_diabetic_friendly":
            proxy = (
                float(row["sugar_pdv"]) * 0.6
                + float(row["carbohydrates_pdv"]) * 0.4
            )
            return proxy < 8

        if flag == "is_gluten_free":
            return not _has_gluten(row.get("ingredients", []))

    except (TypeError, ValueError):
        # NaN or non-numeric value in a nutritional column — reject
        return False

    # Unknown flag: don't reject (conservative default)
    return True


def validate_candidates(
    candidate_df: pd.DataFrame,
    constraints: list[str],
    cfg: dict,
) -> tuple[pd.DataFrame, int]:
    """
    Re-validate dietary constraint compliance for all candidates.

    Applies _satisfies_constraint() for each active constraint to each row.
    Drops any row that fails, logging each discrepancy. A discrepancy means
    either the stored flag is stale (config drift) or a gluten false-negative
    slipped through the pre-computed pool.

    Args:
        candidate_df: DataFrame of FAISS-retrieved candidates (already flag-filtered).
        constraints:  Active dietary flag names for this query.
        cfg:          Parsed config.yaml dict (for live threshold values).

    Returns:
        (validated_df, n_discarded)
          validated_df  — candidates that pass all runtime checks
          n_discarded   — count of rows removed; >0 indicates a data quality issue
    """
    if not constraints or candidate_df.empty:
        return candidate_df, 0

    keep_mask = pd.Series(True, index=candidate_df.index)

    for flag in constraints:
        flag_mask = candidate_df.apply(
            lambda row: _satisfies_constraint(row, flag, cfg), axis=1
        )
        discarded = (~flag_mask) & keep_mask
        if discarded.any():
            n = discarded.sum()
            names = candidate_df.loc[discarded, "name"].tolist()[:5]
            logger.warning(
                f"Runtime validator: {n} recipe(s) failed '{flag}' re-check "
                f"despite being in the compliant pool — possible stale flag or "
                f"hidden gluten source. Examples: {names}"
            )
        keep_mask &= flag_mask

    validated = candidate_df[keep_mask]
    n_discarded = len(candidate_df) - len(validated)
    return validated, n_discarded
