# rag-recipes

A retrieval-augmented recipe recommendation system over ~223,000 Food.com recipes. Ask for food in plain English — *"high protein gluten free dinner"* — and get back recipes that are semantically relevant, nutritionally scored, and **guaranteed** to satisfy the dietary constraint you asked for.

Built without LangChain. Every component — retrieval, filtering, re-ranking, scoring, prompt construction — is explicit and inspectable.

---

## What it does

```
POST /recommend
{"query": "high protein gluten free dinner", "top_k": 5}
```

Response shape (illustrative values, one of `top_k` results shown):

```json
{
  "query": "high protein gluten free dinner",
  "dietary_constraints": ["is_high_protein", "is_gluten_free"],
  "recipes": [
    {
      "recipe_id": 137739,
      "name": "grilled lemon herb chicken",
      "calories": 312.0,
      "protein_pdv": 61.0,
      "semantic_score": 0.71,
      "nutritional_score": 0.88,
      "hybrid_score": 0.79,
      "dietary_labels": ["High protein", "Gluten free", "Low calorie"],
      "explanation": "61% DV protein with only 312 kcal; no gluten-containing ingredients."
    }
  ],
  "llm_response": "Based on your goals, these five recipes..."
}
```

You can also pass `ingredients` — a list of what's already in your kitchen — and the ranking shifts to maximise overlap, returning `matched_ingredients` and `missing_ingredients` per recipe.

---

## Architecture

```
RAW_recipes.csv ──► loader ──► cleaner ──► feature_engineering ──► recipes_featured.parquet
                                                                            │
                                                                            ▼
                                                              encoder (MiniLM-L6-v2, 384d)
                                                                            │
                                                                            ▼
                                                              FAISS IndexFlatIP
                                                                            │
  query ──► keyword→flag parse ──► constrained FAISS search ──► re-validate ──► hybrid re-rank ──► LLM
```

### 1. Feature engineering

Food.com ships nutrition as a 7-element `% Daily Value` vector. Feature engineering unpacks it and derives nine boolean dietary flags using **FDA %DV thresholds** (all configurable in `configs/config.yaml`):

`is_diabetic_friendly` · `is_gluten_free` · `is_high_protein` · `is_low_fat` · `is_low_sodium` · `is_low_calorie` · `is_keto_friendly` · `is_low_carb` · `is_low_sugar`

It also builds `recipe_text` — name + description + ingredients + tags + first two steps — which is the field that actually gets embedded, plus derived features (`log_minutes`, `complexity_score`, `protein_to_calorie_ratio`, `sugar_to_carb_ratio`).

### 2. Retrieval

`RecipeRetriever.retrieve()` runs five stages:

1. **Parse constraints** — map keywords in the query to flag columns (`"celiac"` → `is_gluten_free`, `"workout"` → `is_high_protein`, …)
2. **Constrained ANN search** — FAISS `IndexFlatIP` search restricted to the compliant pool
3. **Runtime re-validation** — re-check flags against live column values
4. **Score** — rule-based `nutritional_score`, plus `ingredient_match_score` when an ingredient list is supplied
5. **Hybrid re-rank**

| weight | no ingredients | with ingredients |
|---|---|---|
| semantic | 0.6 | 0.4 |
| nutritional | 0.3 | 0.25 |
| ingredient match | — | 0.25 |
| popularity | 0.1 | 0.1 |

### 3. Three-layer dietary compliance

Getting "gluten free" *actually* gluten free is the hard part of this project. A recipe that violates a medical constraint is worse than no recommendation at all, so compliance is enforced three times:

| Layer | Mechanism | Catches |
|---|---|---|
| **1** | FAISS `IDSelectorBatch` restricts the search itself to the pre-computed compliant pool | Everything the parquet flags correctly |
| **2** | `src/scoring/validator.py` re-applies threshold logic to live values and re-checks gluten against an expanded keyword set | Stale flags from old config thresholds; hidden gluten — soy sauce, malt vinegar, beer, udon — missed at parquet-build time |
| **3** | LLM narration is instructed to respect the constraint | Residual semantic slips |

Layer 1 is a hard filter, not a post-hoc rerank — non-compliant recipes are never scored in the first place. Layer 2 runs vectorized over at most 500 candidate rows, so it costs effectively nothing.

---

## Evaluation

`src/evaluation/evaluator.py` benchmarks three systems on the same queries and ground truth:

1. **BM25** — TF-IDF keyword search (baseline)
2. **Embedding-only** — pure semantic similarity, no re-ranking
3. **Full hybrid pipeline** — constraints + nutritional scoring + hybrid re-rank

Ground truth is derived from the 1.1M-row interactions table: a recipe is relevant to a dietary query if it carries the flag **and** has `mean_rating > 4.0` **and** `review_count > 10`.

Metrics (`src/evaluation/metrics.py`): **Recall@K**, **MRR**, **NDCG@K**, and **Nutritional Precision@K** — a domain-specific metric measuring the fraction of top-K results that actually satisfy the stated dietary constraint. The last one is the metric that matters here: a system can score well on relevance while still recommending gluten to a celiac user.

```bash
python -m src.evaluation.evaluator
```

---

## Setup

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download the [Food.com Recipes & Interactions dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) from Kaggle and place `RAW_recipes.csv` and `RAW_interactions.csv` in `data/raw/`. The data is not in this repo — it's ~700MB.

### 3. Configure your API key

```bash
cp .env.example .env
```

Then fill in `GEMINI_API_KEY` (default — Gemini 2.5 Flash via Google's OpenAI-compatible endpoint). To use OpenAI instead, set `OPENAI_API_KEY` and edit `configs/config.yaml`: change `llm.model` and remove `llm.base_url`.

### 4. Build the pipeline

Run in order — each step depends on the previous one's output:

```bash
python -m src.data.cleaner               # → data/processed/recipes_cleaned.parquet
python -m src.data.feature_engineering   # → data/processed/recipes_featured.parquet
python -m src.embeddings.indexer         # → data/processed/faiss_index.bin
```

### 5. Serve

```bash
uvicorn src.api.main:app --reload --port 8000
```

Interactive API docs at **http://localhost:8000/docs**.

> All paths in the code are relative to the repo root — always run commands from the top-level directory.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/recommend` | Full pipeline. Body: `query`, `top_k`, optional `ingredients`, optional `dietary_constraints` |
| `GET` | `/explain/{recipe_id}` | Nutritional breakdown and rule-based explanation for one recipe |
| `GET` | `/health` | Health check |

The pipeline (FAISS index + SentenceTransformer) loads lazily on first request, so the first call is slow and subsequent ones are fast.

---

## Tests

```bash
pytest tests/
```

Covers cleaning, feature engineering, nutritional scoring, and the evaluation metrics.

---

## Layout

```
src/
├── data/          loader, cleaner, feature_engineering
├── embeddings/    encoder (MiniLM), indexer (FAISS), retriever
├── scoring/       nutritional_scorer, ingredient_scorer, validator, explainer
├── rag/           prompt_builder, generator
├── evaluation/    metrics, evaluator
└── api/           FastAPI app, routes, Pydantic schemas
notebooks/         EDA → feature engineering → indexing → pipeline demo → evaluation
configs/           config.yaml — all thresholds, paths, model names, score weights
tests/
```

Every threshold, path, model name and score weight lives in `configs/config.yaml`. Nothing is hardcoded in the modules.

---

## Stack

Python · FAISS · sentence-transformers · FastAPI · pandas · Gemini 2.5 Flash (or OpenAI) · pytest · rank-bm25 · ranx
