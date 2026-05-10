"""Tests for feature engineering logic."""

import pandas as pd
import pytest
from src.data.feature_engineering import (
    add_dietary_flags,
    add_numeric_features,
    add_composite_text,
    GLUTEN_KEYWORDS,
)


MOCK_CFG = {
    "nutritional_flags": {
        "low_sugar_pdv": 5,
        "high_protein_pdv": 20,
        "low_fat_pdv": 10,
        "low_sodium_pdv": 10,
        "low_calorie_kcal": 400,
        "keto_max_carbs_pdv": 10,
        "keto_min_fat_pdv": 25,
        "low_carb_pdv": 15,
    }
}


def make_df(**overrides):
    base = {
        "name": ["grilled chicken"],
        "description": ["healthy chicken dish"],
        "ingredients_str": ["chicken breast, olive oil, garlic"],
        "tags_str": ["healthy, quick"],
        "steps": [["marinate chicken", "grill for 20 minutes"]],
        "protein_pdv": [25.0],
        "sugar_pdv": [3.0],
        "total_fat_pdv": [8.0],
        "carbohydrates_pdv": [10.0],
        "sodium_pdv": [7.0],
        "calories": [350.0],
        "minutes": [30],
        "n_steps": [4],
        "n_ingredients": [5],
        "review_count": [50.0],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


# --- Dietary flags ---
def test_diabetic_friendly_flag():
    df = make_df(sugar_pdv=3.0, carbohydrates_pdv=5.0)  # proxy = 3*0.6 + 5*0.4 = 3.8 < 8
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_diabetic_friendly"].iloc[0] is True


def test_not_diabetic_friendly_flag():
    df = make_df(sugar_pdv=12.0, carbohydrates_pdv=18.0)  # proxy = 14.4 > 8
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_diabetic_friendly"].iloc[0] is False


def test_gluten_free_flag_no_gluten():
    df = make_df(ingredients_str=["chicken, rice, olive oil"])
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_gluten_free"].iloc[0] is True


def test_gluten_free_flag_with_flour():
    df = make_df(ingredients_str=["chicken, flour, butter"])
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_gluten_free"].iloc[0] is False


def test_high_protein_flag():
    df = make_df(protein_pdv=25.0)
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_high_protein"].iloc[0] is True


def test_keto_friendly_flag():
    df = make_df(carbohydrates_pdv=8.0, total_fat_pdv=30.0)
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_keto_friendly"].iloc[0] is True


def test_not_keto_high_carbs():
    df = make_df(carbohydrates_pdv=25.0, total_fat_pdv=30.0)
    result = add_dietary_flags(df, MOCK_CFG)
    assert result["is_keto_friendly"].iloc[0] is False


# --- Numeric features ---
def test_protein_to_calorie_ratio():
    df = make_df(protein_pdv=20.0, calories=200.0)
    result = add_numeric_features(df)
    assert result["protein_to_calorie_ratio"].iloc[0] > 0


def test_log_minutes():
    df = make_df(minutes=30)
    result = add_numeric_features(df)
    import numpy as np
    assert abs(result["log_minutes"].iloc[0] - np.log1p(30)) < 1e-6


def test_complexity_score_bounded():
    df = make_df()
    result = add_numeric_features(df)
    assert 0 <= result["complexity_score"].iloc[0] <= 1


# --- Composite text ---
def test_composite_text_contains_name():
    df = make_df()
    result = add_composite_text(df)
    assert "grilled chicken" in result["recipe_text"].iloc[0]


def test_composite_text_contains_ingredients():
    df = make_df()
    result = add_composite_text(df)
    assert "chicken breast" in result["recipe_text"].iloc[0]


def test_composite_text_contains_steps():
    df = make_df()
    result = add_composite_text(df)
    assert "grill" in result["recipe_text"].iloc[0]
