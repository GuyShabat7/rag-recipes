"""
routes.py — FastAPI route handlers.

Endpoints:
  POST /recommend   — main recommendation endpoint
  GET  /explain/{recipe_id} — nutritional breakdown for a specific recipe
  GET  /health      — health check
"""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.models import ExplainResponse, RecommendRequest, RecommendResponse, RecipeResult
from src.rag.generator import RAGPipeline
from src.scoring.explainer import FLAG_LABELS, build_recipe_explanation, get_active_flags
from src.scoring.nutritional_scorer import score_recipe

logger = logging.getLogger(__name__)
router = APIRouter()

# Pipeline is initialized once at startup (expensive: loads FAISS index + model)
_pipeline: RAGPipeline = None
_recipes_df: pd.DataFrame = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_recipes_df() -> pd.DataFrame:
    global _recipes_df
    if _recipes_df is None:
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        _recipes_df = pd.read_parquet(cfg["data"]["featured"]).set_index("id")
    return _recipes_df


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Main recommendation endpoint.

    Given a natural-language query (and optional explicit dietary constraints),
    retrieves and re-ranks the top-K recipes, then generates an LLM explanation.

    Example request:
        POST /recommend
        {"query": "high protein gluten free dinner", "top_k": 5}

    Example response:
        {
          "query": "high protein gluten free dinner",
          "dietary_constraints": ["is_high_protein", "is_gluten_free"],
          "recipes": [...],
          "llm_response": "Based on your goals..."
        }
    """
    pipeline = get_pipeline()

    result = pipeline.recommend(
        query=request.query,
        top_k=request.top_k,
        user_ingredients=request.ingredients,
    )

    recipe_results = []
    for r in result["recipes"]:
        active_flags = [
            FLAG_LABELS[f] for f in FLAG_LABELS if r.get(f, False)
        ]
        recipe_results.append(RecipeResult(
            recipe_id=int(r["recipe_id"]),
            name=r["name"].title(),
            calories=round(r.get("calories", 0), 1),
            protein_pdv=round(r.get("protein_pdv", 0), 1),
            sugar_pdv=round(r.get("sugar_pdv", 0), 1),
            total_fat_pdv=round(r.get("total_fat_pdv", 0), 1),
            carbohydrates_pdv=round(r.get("carbohydrates_pdv", 0), 1),
            sodium_pdv=round(r.get("sodium_pdv", 0), 1),
            mean_rating=r.get("mean_rating"),
            semantic_score=round(r.get("semantic_score", 0), 4),
            nutritional_score=round(r.get("nutritional_score", 0), 4),
            ingredient_match_score=round(r["ingredient_match_score"], 4)
                if r.get("ingredient_match_score") is not None else None,
            hybrid_score=round(r.get("hybrid_score", 0), 4),
            matched_ingredients=r.get("matched_ingredients"),
            missing_ingredients=r.get("missing_ingredients"),
            explanation=r.get("explanation", ""),
            dietary_labels=active_flags,
        ))

    return RecommendResponse(
        query=result["query"],
        dietary_constraints=result["dietary_constraints"],
        user_ingredients=result.get("user_ingredients"),
        recipes=recipe_results,
        llm_response=result["response"],
    )


@router.get("/explain/{recipe_id}", response_model=ExplainResponse)
def explain(recipe_id: int):
    """
    Return the full nutritional breakdown and rule-based explanation for a recipe.

    Useful for users who want to understand why a recipe qualifies for a dietary label.
    Every claim is backed by a specific threshold and measured value.
    """
    df = get_recipes_df()

    if recipe_id not in df.index:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")

    row = df.loc[recipe_id]
    active_flags = get_active_flags(row)
    _, explanations = score_recipe(row, active_flags)
    explanation_text = build_recipe_explanation(row, explanations)

    return ExplainResponse(
        recipe_id=recipe_id,
        name=row["name"].title(),
        explanation=explanation_text,
        active_flags=[FLAG_LABELS.get(f, f) for f in active_flags],
        nutritional_facts={
            "calories": round(row.get("calories", 0), 1),
            "protein_pdv": round(row.get("protein_pdv", 0), 1),
            "sugar_pdv": round(row.get("sugar_pdv", 0), 1),
            "total_fat_pdv": round(row.get("total_fat_pdv", 0), 1),
            "carbohydrates_pdv": round(row.get("carbohydrates_pdv", 0), 1),
            "sodium_pdv": round(row.get("sodium_pdv", 0), 1),
            "saturated_fat_pdv": round(row.get("saturated_fat_pdv", 0), 1),
        },
    )
