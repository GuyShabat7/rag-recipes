# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (use the venv/ in the repo root)
pip install -r requirements.txt

# Run FastAPI server (must be run from the repo root)
uvicorn src.api.main:app --reload --port 8000

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_metrics.py -v

# Run evaluation across all three systems
python -m src.evaluation.evaluator
```

API docs are auto-generated at `http://localhost:8000/docs` (Swagger UI).

## Environment

Copy `.env` and populate your key:
- `GEMINI_API_KEY` — default LLM (Gemini 2.5 Flash via Google's OpenAI-compatible endpoint)
- `OPENAI_API_KEY` — alternative; switch by editing `configs/config.yaml` (`llm.model`, remove `llm.base_url`)

All paths in the code are **relative to the repo root** — always run commands from the top-level directory.

## Architecture

This is a RAG-based recipe recommendation system built over ~223k Food.com recipes. There is no LangChain; every component is explicit and custom.

### Data pipeline (must run in order before anything else works)

Raw CSVs (`data/raw/`) → `src/data/loader.py` → `src/data/cleaner.py` → `data/processed/recipes_cleaned.parquet` → `src/data/feature_engineering.py` → `data/processed/recipes_featured.parquet`

The featured parquet is the single source of truth consumed by embeddings and the API. Key columns added by feature engineering:
- Boolean dietary flags: `is_diabetic_friendly`, `is_gluten_free`, `is_high_protein`, `is_low_fat`, `is_low_sodium`, `is_low_calorie`, `is_keto_friendly`, `is_low_carb`, `is_low_sugar`
- Composite `recipe_text` field (name + description + ingredients + tags + first two steps) — this is what gets embedded
- `log_minutes`, `complexity_score`, `protein_to_calorie_ratio`, `sugar_to_carb_ratio`

### Embedding + FAISS index

`src/embeddings/encoder.py` wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim). `src/embeddings/indexer.py` builds and persists a FAISS `IndexFlatIP` to `data/processed/faiss_index.bin`. The index must be rebuilt if the featured parquet changes.

### Retrieval pipeline (the core of the system)

`src/embeddings/retriever.py` — `RecipeRetriever.retrieve()`:
1. Parse dietary keywords from query text → map to flag column names (see `KEYWORD_TO_FLAG`)
2. Encode query → FAISS search → top-50 semantic candidates
3. Hard-filter candidates by dietary flags (relaxes automatically if no candidates survive)
4. Score each candidate: `nutritional_score` (rule-based, `src/scoring/nutritional_scorer.py`), optional `ingredient_match_score` (`src/scoring/ingredient_scorer.py`)
5. Hybrid re-rank: `0.6*semantic + 0.3*nutritional + 0.1*popularity` (or `0.4/0.25/0.25/0.1` when ingredients are provided)
6. Return top-K

### RAG generation layer

`src/rag/generator.py` — `RAGPipeline.recommend()` calls `RecipeRetriever`, builds a structured prompt via `src/rag/prompt_builder.py`, and calls the LLM. The LLM is accessed via the OpenAI SDK regardless of whether Gemini or OpenAI is configured (Gemini uses a compatible endpoint).

### API

`src/api/main.py` mounts the FastAPI router. Two main endpoints in `src/api/routes.py`:
- `POST /recommend` — full pipeline; accepts `query`, `top_k`, optional `ingredients` list
- `GET /explain/{recipe_id}` — nutritional breakdown and rule-based explanation for a recipe

The pipeline and recipes DataFrame are initialized lazily on first request (expensive: loads FAISS index + SentenceTransformer model).

### Evaluation

`src/evaluation/evaluator.py` compares three systems: BM25 (baseline), embedding-only, and the full hybrid pipeline. Ground truth is built from interactions: recipes with the dietary flag + `mean_rating > 4.0` + `review_count > 10`. Metrics (`src/evaluation/metrics.py`): Recall@K, MRR, NDCG@K, and Nutritional Precision@K (domain-specific: fraction of top-K that satisfy the stated dietary constraint).

## Config

All thresholds, paths, model names, and score weights live in `configs/config.yaml`. Dietary flag thresholds are FDA % Daily Value standards. Score weight sets differ depending on whether the user provides an explicit ingredient list (see `retrieval.score_weights` vs `retrieval.score_weights_with_ingredients`).
