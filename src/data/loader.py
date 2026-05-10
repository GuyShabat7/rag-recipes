"""
loader.py — Load and validate raw Food.com CSV files.

Raw data: RAW_recipes.csv (~230k rows) and RAW_interactions.csv (~1.1M rows)
from https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
"""

import ast
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _safe_literal_eval(val):
    """Parse a stringified Python list/dict. Returns None on failure."""
    if pd.isna(val):
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None


def load_recipes(path: str) -> pd.DataFrame:
    """
    Load RAW_recipes.csv and parse list columns from string literals.

    Columns with stored Python lists: nutrition, steps, tags, ingredients
    ~0.3% of rows have malformed literals and are dropped here.
    """
    logger.info(f"Loading recipes from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw recipe count: {len(df):,}")

    list_cols = ["nutrition", "steps", "tags", "ingredients"]
    for col in list_cols:
        df[col] = df[col].apply(_safe_literal_eval)

    before = len(df)
    df = df.dropna(subset=list_cols)
    dropped = before - len(df)
    logger.info(f"Dropped {dropped:,} rows with malformed list literals ({dropped/before:.2%})")

    # Validate nutrition vector has exactly 7 elements
    df = df[df["nutrition"].apply(lambda x: isinstance(x, list) and len(x) == 7)]
    logger.info(f"After literal parse validation: {len(df):,} recipes")

    return df.reset_index(drop=True)


def load_interactions(path: str) -> pd.DataFrame:
    """
    Load RAW_interactions.csv (~1.1M user-recipe interactions).
    Returns columns: user_id, recipe_id, date, rating, review
    """
    logger.info(f"Loading interactions from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw interaction count: {len(df):,}")
    return df


def load_interaction_aggregates(interactions_path: str) -> pd.DataFrame:
    """
    Aggregate interactions to per-recipe mean_rating and review_count.
    Used for joining popularity signals onto the recipe table.
    """
    df = load_interactions(interactions_path)
    agg = (
        df.groupby("recipe_id")
        .agg(mean_rating=("rating", "mean"), review_count=("rating", "count"))
        .reset_index()
    )
    logger.info(f"Aggregated interactions for {len(agg):,} unique recipes")
    return agg
