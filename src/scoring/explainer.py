"""
explainer.py — Human-readable explanations for why a recipe matches a dietary profile.

Generates structured explanation text that is:
  1. Injected into the LLM prompt as context (so the LLM can reference real facts)
  2. Returned directly in the API response as the 'explanation' field

This is the 'explainability layer' of the system. Every claim traces back to a
specific nutritional threshold and a specific value in the data — there are no
vague claims like "this recipe is healthy".
"""

import pandas as pd


# Human-readable names for dietary flags (used in UI/API output)
FLAG_LABELS = {
    "is_diabetic_friendly": "Diabetic-Friendly",
    "is_gluten_free": "Gluten-Free",
    "is_high_protein": "High-Protein",
    "is_low_fat": "Low-Fat",
    "is_low_sodium": "Low-Sodium",
    "is_low_calorie": "Low-Calorie",
    "is_keto_friendly": "Keto-Friendly",
    "is_low_carb": "Low-Carb",
    "is_low_sugar": "Low-Sugar",
}


def get_active_flags(row: pd.Series) -> list[str]:
    """Return list of dietary flag column names that are True for this recipe."""
    return [
        flag for flag in FLAG_LABELS
        if row.get(flag, False)
    ]


def build_recipe_explanation(
    row: pd.Series,
    constraint_explanations: list[str],
) -> str:
    """
    Build a full explanation string for a recipe, combining:
      - Active dietary flags (what it qualifies for)
      - Constraint-specific score explanations (the why, with numbers)
      - Key nutritional facts

    This string is injected into the LLM prompt so the model can
    reference real numbers rather than hallucinating them.
    """
    active_flags = get_active_flags(row)
    flag_labels = [FLAG_LABELS[f] for f in active_flags if f in FLAG_LABELS]

    lines = []

    if flag_labels:
        lines.append(f"Dietary labels: {', '.join(flag_labels)}")

    lines.append(
        f"Nutrition: {row.get('calories', 0):.0f} kcal | "
        f"Protein {row.get('protein_pdv', 0):.0f}% DV | "
        f"Fat {row.get('total_fat_pdv', 0):.0f}% DV | "
        f"Sugar {row.get('sugar_pdv', 0):.0f}% DV | "
        f"Carbs {row.get('carbohydrates_pdv', 0):.0f}% DV | "
        f"Sodium {row.get('sodium_pdv', 0):.0f}% DV"
    )

    if constraint_explanations:
        lines.append("Why it qualifies:")
        for exp in constraint_explanations:
            lines.append(f"  • {exp}")

    rating = row.get("mean_rating")
    review_count = row.get("review_count", 0)
    if pd.notna(rating) and review_count > 0:
        lines.append(f"Community: {rating:.1f}/5 stars ({int(review_count)} reviews)")

    return "\n".join(lines)


def format_recipes_for_prompt(recipes: list[dict]) -> str:
    """
    Format a list of recipe dicts (from retriever output) into a structured
    text block for injection into the LLM prompt.

    Each recipe block includes name, ingredients preview, nutritional facts,
    dietary labels, and the why-it-qualifies explanation.
    """
    blocks = []
    for i, r in enumerate(recipes, 1):
        block = [
            f"[Recipe {i}: {r['name'].title()}]",
            f"Ingredients (preview): {r.get('ingredients_str', '')[:120]}...",
            r.get("explanation", ""),
            f"Nutritional score: {r.get('nutritional_score', 0):.2f}/1.0",
            f"Relevance score: {r.get('hybrid_score', 0):.3f}",
        ]
        blocks.append("\n".join(line for line in block if line))

    return "\n\n".join(blocks)
