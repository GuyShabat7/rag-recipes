"""
cleaner.py — Clean and normalize raw recipe data.

Cleaning steps and row counts (starting from ~230k):
  Step 1: Parse list literals          → drops ~700  (malformed strings)
  Step 2: Deduplicate                  → drops ~4,200 (same name + ingredients)
  Step 3: Filter stubs/outliers        → drops ~2,800 (too few ingredients/steps, extreme times)
  Step 4: Unpack + cap nutrition       → no row drops, adds 7 named columns
  Step 5: Normalize text               → no row drops
  Step 6: Join interaction aggregates  → no row drops (left join)
  Final:  ~223,000 clean recipes
"""

import logging
import re

import numpy as np
import pandas as pd

from src.data.loader import load_config, load_interaction_aggregates, load_recipes

logger = logging.getLogger(__name__)

# Maps position in nutrition vector to column name
NUTRITION_COLS = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
]


def unpack_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the 7-element nutrition list into named columns."""
    nutrition_df = pd.DataFrame(df["nutrition"].tolist(), columns=NUTRITION_COLS, index=df.index)
    return pd.concat([df.drop(columns=["nutrition"]), nutrition_df], axis=1)


def cap_nutrition_outliers(df: pd.DataFrame, percentile: int = 99) -> pd.DataFrame:
    """Cap nutritional columns at the given percentile to remove data entry errors."""
    for col in NUTRITION_COLS:
        cap = df[col].quantile(percentile / 100)
        df[col] = df[col].clip(upper=cap)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate recipes using (lowercase name, frozenset of ingredients).
    Keeps the first occurrence (oldest submission date).
    """
    before = len(df)
    df = df.dropna(subset=["name"])  # some rows have NaN name
    df = df.sort_values("submitted")  # keep earliest submission
    key = df.apply(
        lambda r: (str(r["name"]).lower().strip(), frozenset(r["ingredients"])), axis=1
    )
    df = df[~key.duplicated(keep="first")]
    logger.info(f"Deduplication: removed {before - len(df):,} duplicate recipes")
    return df.reset_index(drop=True)


def filter_stubs_and_outliers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Remove:
      - Recipes with fewer than min_ingredients ingredients (stub entries)
      - Recipes with fewer than min_steps steps (stub entries)
      - Recipes with more than max_minutes prep time (data entry errors)
    """
    c = cfg["cleaning"]
    before = len(df)

    mask = (
        (df["n_ingredients"] >= c["min_ingredients"])
        & (df["n_steps"] >= c["min_steps"])
        & (df["minutes"] <= c["max_minutes"])
    )
    df = df[mask]
    logger.info(f"Stub/outlier filter: removed {before - len(df):,} recipes")
    return df.reset_index(drop=True)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and clean text fields; join list fields to strings."""
    df["name"] = df["name"].str.lower().str.strip()
    df["description"] = df["description"].fillna("").apply(_strip_html).str.lower()
    df["ingredients_str"] = df["ingredients"].apply(lambda lst: ", ".join(lst))
    df["tags_str"] = df["tags"].apply(lambda lst: ", ".join(lst))
    df["steps_str"] = df["steps"].apply(lambda lst: " ".join(lst))
    return df


def clean(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a clean DataFrame saved to parquet.
    """
    cfg = load_config(config_path)

    # Step 1: Load (list literal parsing happens inside load_recipes)
    df = load_recipes(cfg["data"]["raw_recipes"])

    # Step 2: Deduplicate
    df = deduplicate(df)

    # Step 3: Filter stubs and outliers
    df = filter_stubs_and_outliers(df, cfg)

    # Step 4: Unpack nutrition vector + cap outliers
    df = unpack_nutrition(df)
    df = cap_nutrition_outliers(df, cfg["cleaning"]["nutrition_cap_percentile"])

    # Step 5: Normalize text fields
    df = normalize_text(df)

    # Step 6: Join popularity signals from interactions
    agg = load_interaction_aggregates(cfg["data"]["raw_interactions"])
    df = df.merge(agg, left_on="id", right_on="recipe_id", how="left").drop(
        columns=["recipe_id"]
    )

    logger.info(f"Final clean recipe count: {len(df):,}")

    out_path = cfg["data"]["cleaned"]
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned data to {out_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean()
