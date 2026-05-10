"""Tests for data cleaning logic."""

import pandas as pd
import pytest
from src.data.cleaner import (
    cap_nutrition_outliers,
    deduplicate,
    normalize_text,
    unpack_nutrition,
)

NUTRITION_COLS = [
    "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv",
]


def make_recipe_df(overrides=None):
    base = {
        "id": [1, 2, 3],
        "name": ["Chicken Salad", "Egg Toast", "Chicken Salad"],
        "submitted": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "minutes": [30, 15, 30],
        "n_steps": [3, 2, 3],
        "n_ingredients": [5, 4, 5],
        "ingredients": [["chicken", "lettuce"], ["egg", "bread"], ["chicken", "lettuce"]],
        "steps": [["step1", "step2", "step3"], ["step1", "step2"], ["step1", "step2", "step3"]],
        "tags": [["healthy"], ["quick"], ["healthy"]],
        "description": ["A salad", "Toast", "A salad"],
        "nutrition": [
            [300.0, 10.0, 5.0, 8.0, 20.0, 3.0, 15.0],
            [200.0, 8.0, 2.0, 5.0, 12.0, 2.0, 10.0],
            [300.0, 10.0, 5.0, 8.0, 20.0, 3.0, 15.0],
        ],
    }
    if overrides:
        base.update(overrides)
    return pd.DataFrame(base)


def test_unpack_nutrition():
    df = make_recipe_df()
    result = unpack_nutrition(df)
    for col in NUTRITION_COLS:
        assert col in result.columns
    assert "nutrition" not in result.columns
    assert result["calories"].iloc[0] == 300.0
    assert result["protein_pdv"].iloc[0] == 20.0


def test_deduplicate_removes_exact_dupes():
    df = make_recipe_df()
    # Recipes 0 and 2 are duplicates (same name + ingredients)
    result = deduplicate(df)
    assert len(result) == 2


def test_deduplicate_keeps_different_recipes():
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["recipe a", "recipe b"],
        "submitted": ["2020-01-01", "2020-01-02"],
        "ingredients": [["egg", "butter"], ["flour", "sugar"]],
    })
    result = deduplicate(df)
    assert len(result) == 2


def test_cap_nutrition_outliers():
    df = make_recipe_df()
    df = unpack_nutrition(df)
    # Add an extreme outlier
    df.loc[0, "calories"] = 99999.0
    result = cap_nutrition_outliers(df, percentile=99)
    assert result["calories"].max() < 99999.0


def test_normalize_text():
    df = make_recipe_df()
    df["description"] = ["Hello <b>World</b>", "Test", "Another"]
    result = normalize_text(df)
    assert result["name"].iloc[0] == "chicken salad"
    assert "<b>" not in result["description"].iloc[0]
    assert "ingredients_str" in result.columns
    assert "tags_str" in result.columns
    assert isinstance(result["ingredients_str"].iloc[0], str)
