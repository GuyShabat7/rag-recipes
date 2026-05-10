"""
encoder.py — SentenceTransformer wrapper for embedding recipes and queries.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional output vectors
  - Trained on 1B+ sentence pairs with contrastive learning
  - ~14,000 sentences/second on CPU
  - Understands semantic similarity: 'grilled chicken salad' and
    'poultry with greens' will have high cosine similarity

The same model is used for BOTH recipe indexing and query encoding.
This is critical — query and document vectors must be in the same embedding space.
"""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RecipeEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 512,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into L2-normalized embedding vectors.

        Normalization is required for FAISS IndexFlatIP (inner product) to
        compute cosine similarity correctly. After L2 normalization,
        inner product == cosine similarity.

        Args:
            texts: Single string or list of strings to encode.
            batch_size: Number of texts per encoding batch.
            normalize: L2-normalize output vectors (required for cosine sim).
            show_progress: Show tqdm progress bar (useful for large corpora).

        Returns:
            np.ndarray of shape (n_texts, dimension), dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single user query. Returns shape (1, dimension)."""
        return self.encode(query, normalize=True)
