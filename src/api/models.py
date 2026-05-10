"""
models.py — Pydantic request/response schemas for the FastAPI endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., description="Natural-language user query", min_length=3)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recipes to return")
    dietary_constraints: Optional[list[str]] = Field(
        default=None,
        description="Explicit dietary flag names (auto-parsed from query if not provided)"
    )
    ingredients: Optional[list[str]] = Field(
        default=None,
        description="Ingredients the user already has at home. When provided, recipes are "
                    "re-ranked to maximise ingredient overlap and minimise missing ingredients.",
        examples=[["chicken", "rice", "tomato", "garlic", "olive oil"]],
    )


class RecipeResult(BaseModel):
    recipe_id: int
    name: str
    calories: float
    protein_pdv: float
    sugar_pdv: float
    total_fat_pdv: float
    carbohydrates_pdv: float
    sodium_pdv: float
    mean_rating: Optional[float]
    semantic_score: float
    nutritional_score: float
    ingredient_match_score: Optional[float] = None
    hybrid_score: float
    matched_ingredients: Optional[list[str]] = None   # ingredients user already has
    missing_ingredients: Optional[list[str]] = None   # ingredients user needs to buy
    explanation: str
    dietary_labels: list[str]


class RecommendResponse(BaseModel):
    query: str
    dietary_constraints: list[str]
    user_ingredients: Optional[list[str]] = None
    recipes: list[RecipeResult]
    llm_response: str


class ExplainResponse(BaseModel):
    recipe_id: int
    name: str
    explanation: str
    active_flags: list[str]
    nutritional_facts: dict
