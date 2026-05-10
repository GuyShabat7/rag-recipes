"""
nutritional_scorer.py — Rule-based nutritional scoring model.

Design philosophy (important for interview):
  This is deliberately rule-based rather than ML-learned. Nutritional thresholds
  are defined by clinical/regulatory standards (FDA % Daily Values), not learned
  from data. This makes the model:
    - Explainable: every score decision traces to a threshold and a reason
    - Auditable: a nutritionist can verify or override any rule
    - Robust: no risk of learning spurious correlations from the training data

  A learned re-ranker (e.g., cross-encoder like ms-marco-MiniLM-L-6-v2) could
  improve raw retrieval metrics, but rule-based scoring is the right choice here
  because dietary suitability has clinical ground truth, not just user preference.

Scoring logic:
  Each active constraint contributes 0.0, 0.5, or 1.0 to the score.
  The final score is the mean across all active constraints → always in [0, 1].

  1.0 = fully satisfies the threshold
  0.5 = partially satisfies (borderline)
  0.0 = does not satisfy

Constraint → threshold mapping uses FDA % Daily Value standards:
  - 'high protein': ≥20% DV = 1.0, ≥10% DV = 0.5
  - 'low sugar': <5% DV = 1.0, <10% DV = 0.5
  - 'diabetic': glycemic_proxy <8 = 1.0, <12 = 0.5
  - 'gluten free': is_gluten_free flag = 1.0 or 0.0 (binary)
  - 'low fat': <10% DV = 1.0, <20% DV = 0.5
  - 'low sodium': <10% DV = 1.0, <20% DV = 0.5
  - 'low calorie': <400 kcal = 1.0, <600 kcal = 0.5
  - 'keto': carbs<10% AND fat≥25% = 1.0, carbs<15% = 0.5
  - 'low carb': <15% DV = 1.0, <25% DV = 0.5
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Maps dietary flag column names to scorer functions
# Each function takes a recipe row and returns (score: float, explanation: str)
_SCORERS = {}


def register_scorer(flag: str):
    def decorator(fn):
        _SCORERS[flag] = fn
        return fn
    return decorator


@register_scorer("is_high_protein")
def _score_high_protein(row) -> tuple[float, str]:
    pdv = row["protein_pdv"]
    if pdv >= 20:
        return 1.0, f"High protein: {pdv:.0f}% DV (threshold ≥20% DV)"
    elif pdv >= 10:
        return 0.5, f"Moderate protein: {pdv:.0f}% DV (borderline, threshold ≥20% DV)"
    return 0.0, f"Low protein: {pdv:.0f}% DV (below 10% DV)"


@register_scorer("is_low_sugar")
def _score_low_sugar(row) -> tuple[float, str]:
    pdv = row["sugar_pdv"]
    if pdv < 5:
        return 1.0, f"Low sugar: {pdv:.0f}% DV (threshold <5% DV)"
    elif pdv < 10:
        return 0.5, f"Moderate sugar: {pdv:.0f}% DV (borderline, threshold <5% DV)"
    return 0.0, f"High sugar: {pdv:.0f}% DV (above 10% DV)"


@register_scorer("is_diabetic_friendly")
def _score_diabetic(row) -> tuple[float, str]:
    # glycemic_proxy = sugar_pdv * 0.6 + carbs_pdv * 0.4
    proxy = row["sugar_pdv"] * 0.6 + row["carbohydrates_pdv"] * 0.4
    if proxy < 8:
        return 1.0, (
            f"Diabetic-friendly: glycemic proxy={proxy:.1f} "
            f"(sugar={row['sugar_pdv']:.0f}%+carbs={row['carbohydrates_pdv']:.0f}%, threshold <8)"
        )
    elif proxy < 12:
        return 0.5, f"Borderline glycemic load: proxy={proxy:.1f} (threshold <8)"
    return 0.0, f"High glycemic load: proxy={proxy:.1f} (above 12)"


@register_scorer("is_gluten_free")
def _score_gluten_free(row) -> tuple[float, str]:
    if row.get("is_gluten_free", False):
        return 1.0, "Gluten-free: no wheat, barley, rye, or semolina detected in ingredients"
    return 0.0, "Contains gluten: wheat, barley, rye, or semolina found in ingredients"


@register_scorer("is_low_fat")
def _score_low_fat(row) -> tuple[float, str]:
    pdv = row["total_fat_pdv"]
    if pdv < 10:
        return 1.0, f"Low fat: {pdv:.0f}% DV (threshold <10% DV)"
    elif pdv < 20:
        return 0.5, f"Moderate fat: {pdv:.0f}% DV (borderline, threshold <10% DV)"
    return 0.0, f"High fat: {pdv:.0f}% DV (above 20% DV)"


@register_scorer("is_low_sodium")
def _score_low_sodium(row) -> tuple[float, str]:
    pdv = row["sodium_pdv"]
    if pdv < 10:
        return 1.0, f"Low sodium: {pdv:.0f}% DV (threshold <10% DV)"
    elif pdv < 20:
        return 0.5, f"Moderate sodium: {pdv:.0f}% DV (borderline, threshold <10% DV)"
    return 0.0, f"High sodium: {pdv:.0f}% DV (above 20% DV)"


@register_scorer("is_low_calorie")
def _score_low_calorie(row) -> tuple[float, str]:
    kcal = row["calories"]
    if kcal < 400:
        return 1.0, f"Low calorie: {kcal:.0f} kcal (threshold <400 kcal)"
    elif kcal < 600:
        return 0.5, f"Moderate calorie: {kcal:.0f} kcal (borderline, threshold <400 kcal)"
    return 0.0, f"High calorie: {kcal:.0f} kcal (above 600 kcal)"


@register_scorer("is_keto_friendly")
def _score_keto(row) -> tuple[float, str]:
    carbs = row["carbohydrates_pdv"]
    fat = row["total_fat_pdv"]
    if carbs < 10 and fat >= 25:
        return 1.0, f"Keto-friendly: carbs={carbs:.0f}% DV (<10%), fat={fat:.0f}% DV (≥25%)"
    elif carbs < 15:
        return 0.5, f"Low-carb but not fully keto: carbs={carbs:.0f}% DV, fat={fat:.0f}% DV"
    return 0.0, f"Not keto: carbs={carbs:.0f}% DV (above 15% DV)"


@register_scorer("is_low_carb")
def _score_low_carb(row) -> tuple[float, str]:
    pdv = row["carbohydrates_pdv"]
    if pdv < 15:
        return 1.0, f"Low carb: {pdv:.0f}% DV (threshold <15% DV)"
    elif pdv < 25:
        return 0.5, f"Moderate carb: {pdv:.0f}% DV (borderline, threshold <15% DV)"
    return 0.0, f"High carb: {pdv:.0f}% DV (above 25% DV)"


def score_recipe(
    row: Any,
    active_constraints: list[str],
) -> tuple[float, list[str]]:
    """
    Score a single recipe against a list of active dietary constraints.

    Args:
        row: A pandas Series or dict with nutritional and flag columns.
        active_constraints: List of dietary flag column names to score against.

    Returns:
        (score, explanations)
          score        — float in [0, 1], mean across all constraints
          explanations — list of human-readable explanation strings
    """
    if not active_constraints:
        return 0.5, ["No dietary constraints specified"]

    scores = []
    explanations = []
    for flag in active_constraints:
        scorer = _SCORERS.get(flag)
        if scorer is None:
            continue
        s, explanation = scorer(row)
        scores.append(s)
        explanations.append(explanation)

    if not scores:
        return 0.5, ["No matching scorers found for constraints"]

    final_score = sum(scores) / len(scores)
    return final_score, explanations
