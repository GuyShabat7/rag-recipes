"""
prompt_builder.py — Assemble the LLM prompt from retrieved recipe context.

The prompt has three parts:
  1. System message — defines the assistant's role as a nutritionist/chef
  2. Retrieved context — top-K recipes with nutritional facts and explanations
  3. User instruction — query + dietary constraints + output format request

Design decisions:
  - We inject pre-computed rule explanations into the context so the LLM can
    reference them verbatim rather than reasoning about raw numbers.
  - We include explicit nutritional values (not just labels) so the LLM can
    make graded comparisons (e.g., "Recipe 1 is higher in protein than Recipe 2").
  - We ask the LLM to rank and explain, not just list, to produce useful output.
  - Temperature is set to 0.3 (in config) — low enough for factual consistency,
    slightly above 0 to allow natural language variation.
"""

from typing import Optional

from src.scoring.explainer import format_recipes_for_prompt

SYSTEM_PROMPT = """You are a nutritionist and culinary assistant. You recommend recipes
based on specific dietary goals and health conditions. You explain your recommendations
using precise nutritional facts — never make vague claims like "this is healthy" without
citing specific values.

Your audience may include people managing diabetes, athletes, people with gluten
intolerance, or anyone with specific dietary requirements. Accuracy matters.

When recommending recipes:
1. Rank them from best to worst match for the user's goals
2. For each recipe, explain WHY it fits using the nutritional data provided
3. Note any caveats (e.g., borderline values, high sodium despite low sugar)
4. Keep explanations concise but specific — cite the actual numbers
5. If ingredient information is provided, mention which key ingredients the user already has
   and what they would need to buy
"""


def _format_ingredient_section(recipe: dict) -> str:
    """Build the ingredient match section for a single recipe in the prompt."""
    matched = recipe.get("matched_ingredients")
    missing = recipe.get("missing_ingredients")

    if not matched and not missing:
        return ""

    lines = []
    if matched:
        lines.append(f"  You already have ({len(matched)}): {', '.join(matched[:8])}")
    if missing:
        lines.append(f"  You need to buy ({len(missing)}): {', '.join(missing[:8])}")

    match_score = recipe.get("ingredient_match_score")
    if match_score is not None:
        lines.append(f"  Ingredient match: {match_score * 100:.0f}%")

    return "\n".join(lines)


def format_recipes_for_prompt_with_ingredients(recipes: list[dict]) -> str:
    """
    Extended recipe formatter that includes ingredient match info when available.
    Falls back to the standard formatter if no ingredient data is present.
    """
    has_ingredient_data = any(
        r.get("matched_ingredients") is not None for r in recipes
    )

    if not has_ingredient_data:
        return format_recipes_for_prompt(recipes)

    blocks = []
    for i, r in enumerate(recipes, 1):
        ing_section = _format_ingredient_section(r)
        block_lines = [
            f"[Recipe {i}: {r['name'].title()}]",
            f"Ingredients (preview): {r.get('ingredients_str', '')[:120]}...",
            r.get("explanation", ""),
        ]
        if ing_section:
            block_lines.append(ing_section)
        block_lines += [
            f"Nutritional score: {r.get('nutritional_score', 0):.2f}/1.0",
            f"Ingredient match: {r.get('ingredient_match_score', 0) * 100:.0f}%"
            if r.get("ingredient_match_score") is not None else "",
            f"Overall score: {r.get('hybrid_score', 0):.3f}",
        ]
        blocks.append("\n".join(line for line in block_lines if line))

    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    recipes: list[dict],
    dietary_constraints: list[str],
    user_ingredients: Optional[list[str]] = None,
) -> str:
    """
    Build the full user-turn message for the LLM.

    Args:
        query: Original user query string.
        recipes: List of recipe dicts from the retriever.
        dietary_constraints: Parsed dietary flag names (for display in prompt).
        user_ingredients: Optional list of ingredients the user already has.
    """
    constraint_labels = [c.replace("is_", "").replace("_", " ") for c in dietary_constraints]
    constraint_str = ", ".join(constraint_labels) if constraint_labels else "general healthy eating"

    context_block = format_recipes_for_prompt_with_ingredients(recipes)

    ingredient_context = ""
    if user_ingredients:
        ingredient_context = (
            f"\nIngredients the user already has: {', '.join(user_ingredients)}\n"
            "Please note which recipes make the best use of these ingredients "
            "and what (if anything) would need to be purchased."
        )

    prompt = f"""User request: "{query}"
Dietary goals identified: {constraint_str}{ingredient_context}

Here are the top {len(recipes)} matching recipes retrieved from a database of 223,000 recipes:

{context_block}

Based on the user's dietary goals ({constraint_str}) and the data above,
please provide a ranked recommendation. For each recipe:
- State why it is a good match (cite specific nutritional values)
- Note any weaknesses or caveats
- If ingredient data is shown, mention what the user already has vs. needs to buy
- Give a brief practical note (ease of cooking, meal occasion)

Start with the best match for the user's stated goals."""

    return prompt


def build_messages(
    query: str,
    recipes: list[dict],
    dietary_constraints: list[str],
    user_ingredients: Optional[list[str]] = None,
) -> list[dict]:
    """Return the messages list in OpenAI chat format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(query, recipes, dietary_constraints, user_ingredients)},
    ]
