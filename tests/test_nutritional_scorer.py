"""Tests for the rule-based nutritional scoring model."""

import pytest
from src.scoring.nutritional_scorer import score_recipe


def make_row(**kwargs):
    defaults = {
        "protein_pdv": 15.0,
        "sugar_pdv": 7.0,
        "total_fat_pdv": 15.0,
        "carbohydrates_pdv": 20.0,
        "sodium_pdv": 15.0,
        "calories": 450.0,
        "saturated_fat_pdv": 10.0,
        "is_gluten_free": True,
    }
    defaults.update(kwargs)
    return defaults


# --- High protein ---
def test_high_protein_full_score():
    row = make_row(protein_pdv=25.0)
    score, explanations = score_recipe(row, ["is_high_protein"])
    assert score == 1.0
    assert "25" in explanations[0]


def test_high_protein_partial_score():
    row = make_row(protein_pdv=12.0)
    score, _ = score_recipe(row, ["is_high_protein"])
    assert score == 0.5


def test_high_protein_zero_score():
    row = make_row(protein_pdv=5.0)
    score, _ = score_recipe(row, ["is_high_protein"])
    assert score == 0.0


# --- Low sugar ---
def test_low_sugar_full_score():
    row = make_row(sugar_pdv=3.0)
    score, explanations = score_recipe(row, ["is_low_sugar"])
    assert score == 1.0


def test_low_sugar_zero_score():
    row = make_row(sugar_pdv=20.0)
    score, _ = score_recipe(row, ["is_low_sugar"])
    assert score == 0.0


# --- Diabetic friendly ---
def test_diabetic_full_score():
    # glycemic_proxy = 4*0.6 + 5*0.4 = 4.4 < 8
    row = make_row(sugar_pdv=4.0, carbohydrates_pdv=5.0)
    score, _ = score_recipe(row, ["is_diabetic_friendly"])
    assert score == 1.0


def test_diabetic_zero_score():
    # glycemic_proxy = 15*0.6 + 20*0.4 = 17 > 12
    row = make_row(sugar_pdv=15.0, carbohydrates_pdv=20.0)
    score, _ = score_recipe(row, ["is_diabetic_friendly"])
    assert score == 0.0


# --- Gluten free ---
def test_gluten_free_score():
    row = make_row(is_gluten_free=True)
    score, explanations = score_recipe(row, ["is_gluten_free"])
    assert score == 1.0
    assert "Gluten-free" in explanations[0]


def test_not_gluten_free_score():
    row = make_row(is_gluten_free=False)
    score, _ = score_recipe(row, ["is_gluten_free"])
    assert score == 0.0


# --- Multiple constraints ---
def test_multi_constraint_averaging():
    # high protein (1.0) + low sugar (1.0) → mean = 1.0
    row = make_row(protein_pdv=25.0, sugar_pdv=3.0)
    score, explanations = score_recipe(row, ["is_high_protein", "is_low_sugar"])
    assert score == 1.0
    assert len(explanations) == 2


def test_multi_constraint_partial():
    # high protein (1.0) + low sugar (0.0) → mean = 0.5
    row = make_row(protein_pdv=25.0, sugar_pdv=20.0)
    score, _ = score_recipe(row, ["is_high_protein", "is_low_sugar"])
    assert score == 0.5


# --- Edge cases ---
def test_no_constraints_returns_default():
    row = make_row()
    score, explanations = score_recipe(row, [])
    assert score == 0.5


def test_unknown_constraint_ignored():
    row = make_row(protein_pdv=25.0)
    score, _ = score_recipe(row, ["is_high_protein", "is_unknown_flag"])
    assert score == 1.0  # only high_protein scored, unknown ignored
