"""
indexer.py — Build and persist a FAISS vector index over all recipes.

Index type: faiss.IndexFlatIP (exact inner product search)
  - Exact search, no approximation error
  - After L2 normalization, inner product == cosine similarity
  - For 223k recipes at 384 dims: index size ~330 MB, search time ~50ms per query

Scaling note (for interview):
  For production at 10M+ recipes, switch to IndexIVFFlat with nlist=512 clusters.
  This reduces search from O(n) to O(n/nlist), trading a small recall drop for
  10–20x speedup.
"""

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.embeddings.encoder import RecipeEncoder

logger = logging.getLogger(__name__)


def build_index(
    featured_path: str = None,
    index_path: str = None,
    ids_path: str = None,
    config_path: str = "configs/config.yaml",
) -> tuple[faiss.Index, np.ndarray]:
    """
    Build a FAISS index from the featured recipe dataset.

    Steps:
      1. Load recipe_text column from featured parquet
      2. Encode all recipe_text fields in batches with SentenceTransformer
      3. Build FAISS IndexFlatIP (cosine similarity via normalized inner product)
      4. Save index (.bin) and recipe ID mapping (.npy) to disk

    Returns:
        (faiss_index, recipe_ids_array)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    in_path = featured_path or cfg["data"]["featured"]
    idx_path = index_path or cfg["data"]["faiss_index"]
    ids_path = ids_path or cfg["data"]["recipe_ids"]

    logger.info(f"Loading featured recipes from {in_path}")
    df = pd.read_parquet(in_path, columns=["id", "recipe_text"])
    logger.info(f"Encoding {len(df):,} recipes with all-MiniLM-L6-v2...")

    encoder = RecipeEncoder(cfg["embedding"]["model_name"])
    batch_size = cfg["embedding"]["batch_size"]
    dimension = cfg["embedding"]["dimension"]

    texts = df["recipe_text"].tolist()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize=True,
        show_progress=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info(f"FAISS index built. Total vectors: {index.ntotal:,}")

    # Persist index and ID mapping
    Path(idx_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, idx_path)
    recipe_ids = df["id"].values
    np.save(ids_path, recipe_ids)
    logger.info(f"Saved FAISS index to {idx_path}")
    logger.info(f"Saved recipe IDs to {ids_path}")

    return index, recipe_ids


def load_index(
    config_path: str = "configs/config.yaml",
) -> tuple[faiss.Index, np.ndarray]:
    """Load a pre-built FAISS index and its recipe ID mapping from disk."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    idx_path = cfg["data"]["faiss_index"]
    ids_path = cfg["data"]["recipe_ids"]

    logger.info(f"Loading FAISS index from {idx_path}")
    index = faiss.read_index(idx_path)
    recipe_ids = np.load(ids_path)
    logger.info(f"Index loaded. {index.ntotal:,} vectors, {len(recipe_ids):,} IDs")
    return index, recipe_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index()
