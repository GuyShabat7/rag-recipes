# Recipe Recommendation System with RAG — Implementation Plan

## Context
Building a Data Science portfolio project for an interview. The project demonstrates: semantic search via RAG, feature engineering, rule-based explainable models, and rigorous evaluation. The user must be able to confidently answer four questions: what data, what processing, what architecture, how evaluated.

---

## Data

**Source:** Food.com Recipes & Interactions dataset (Kaggle)
- `RAW_recipes.csv` — ~230k recipes with: name, ingredients (list), steps (list), tags (list), nutrition vector [calories, fat, sugar, sodium, protein, sat_fat, carbs] as % DV, minutes, n_steps, n_ingredients, description
- `RAW_interactions.csv` — ~1.1M user-recipe interactions with ratings (1–5)

**After cleaning:** ~223,000 recipes

---

## Project Structure

```
RAG/
├── data/
│   ├── raw/                    # RAW_recipes.csv, RAW_interactions.csv
│   ├── processed/              # recipes_cleaned.parquet, recipes_featured.parquet
│   └── splits/                 # eval_queries.json, ground_truth.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_embedding_indexing.ipynb
│   ├── 04_rag_pipeline_demo.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py            # Load + validate raw CSVs, ast.literal_eval lists
│   │   ├── cleaner.py           # Dedup, outlier removal, text normalization, interactions join
│   │   └── feature_engineering.py  # Nutritional flags, composite text, ratios
│   ├── embeddings/
│   │   ├── encoder.py           # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   │   ├── indexer.py           # Build + persist FAISS IndexFlatIP
│   │   └── retriever.py         # Query → top-K with hybrid re-ranking
│   ├── scoring/
│   │   ├── nutritional_scorer.py  # Rule-based scoring with FDA thresholds
│   │   └── explainer.py          # Rule explanation strings
│   ├── rag/
│   │   ├── prompt_builder.py    # Format top-5 recipes as LLM context
│   │   └── generator.py         # OpenAI GPT-4o-mini call
│   ├── evaluation/
│   │   ├── metrics.py           # Recall@K, MRR, NDCG, Nutritional Precision@K
│   │   └── evaluator.py         # Run all 3 systems + produce comparison table
│   └── api/
│       ├── main.py              # FastAPI entrypoint
│       ├── models.py            # Pydantic schemas
│       └── routes.py            # POST /recommend, GET /explain/{id}
├── tests/
├── configs/config.yaml
└── requirements.txt
```

---

## Data Pipeline (critical for interview)

### Cleaning (`src/data/cleaner.py`)
1. `ast.literal_eval()` to parse ingredients/tags/steps/nutrition stored as string lists (~0.3% malformed rows dropped)
2. Deduplicate on `(name.lower(), frozenset(ingredients))` — removes ~4,200 dupes
3. Filter: drop recipes with <3 ingredients, <2 steps, or >600 minutes → ~223k remain
4. Unpack nutrition vector into 7 named columns; cap at 99th percentile
5. Lowercase text, strip HTML; join ingredients list to comma-separated string
6. Left-join per-recipe `mean_rating` and `review_count` from interactions aggregate

### Feature Engineering (`src/data/feature_engineering.py`)
**Nutritional flags (boolean metadata for hard-filtering at retrieval time):**
```python
is_low_sugar      → sugar_pdv < 5
is_high_protein   → protein_pdv > 20
is_low_fat        → fat_pdv < 10
is_low_sodium     → sodium_pdv < 10
is_low_calorie    → calories < 400
is_keto_friendly  → carbs_pdv < 10 AND fat_pdv > 25
is_low_carb       → carbs_pdv < 15
```
Thresholds are FDA % Daily Value standards — not arbitrary, fully defensible.

**Engineered numeric features:**
- `protein_to_calorie_ratio` = protein_pdv / (calories / 2000)  → nutrient density
- `sugar_to_carb_ratio` = sugar_pdv / (carbs_pdv + 1e-5)  → refined carb signal
- `log_minutes` = log1p(minutes)
- `complexity_score` = 0.4 * n_steps + 0.3 * n_ingredients + 0.3 * log_minutes (normalized 0–1)

**Composite text for embedding (most important engineered feature):**
```python
recipe_text = f"{name}. {description}. Ingredients: {ingredients_str}. Tags: {tags_str}. Steps summary: {steps[0]} {steps[1]}"
```

---

## Architecture

```
User Query
    │
    ▼ Parse dietary constraints → map to flag filters
    │
    ▼ Encode with all-MiniLM-L6-v2 (384-dim)
    │
    ▼ FAISS ANN search → top-50 candidates
    │  + hard-filter by dietary flags
    │
    ▼ Hybrid re-ranking:
    │  score = 0.6 * semantic_sim + 0.3 * nutritional_score + 0.1 * log(review_count)
    │
    ▼ Top-5 → format as structured context (name, ingredients, nutrition, flags, score)
    │
    ▼ GPT-4o-mini generation with nutritional reasoning prompt
    │
    ▼ Explainability: rule-based explanation strings per dietary flag
    │
    ▼ FastAPI response with ranked recommendations + explanations
```

**Key tech choices:**
- `sentence-transformers/all-MiniLM-L6-v2`: 384-dim, ~14k sentences/sec on CPU, trained on 1B+ pairs
- FAISS `IndexFlatIP` (exact) → upgrade to `IndexIVFFlat` (nlist=512) for scale
- Rule-based scoring over learned re-ranker: explainable, auditable, clinically grounded
- No LangChain: can explain every component in an interview; custom code is more defensible

---

## Evaluation

### Evaluation Set Construction
- From interactions: identify 200 tag combinations (e.g., "high-protein + low-fat")
- Relevant recipes = tagged correctly AND mean_rating > 4.0 AND review_count > 10
- Write 200 natural-language queries per tag combo
- Save: `eval_queries.json`, `ground_truth.json` (query_id → [relevant_recipe_ids])

### Three Systems Compared
| System | Description |
|--------|-------------|
| BM25 | TF-IDF keyword search (`rank_bm25` library) |
| Embedding-only | FAISS semantic search, no re-ranking |
| Full Pipeline | Semantic + nutritional re-ranking + dietary filtering |

### Metrics
- **Recall@5, Recall@10**: fraction of relevant recipes in top-K
- **MRR**: mean reciprocal rank of first relevant result
- **NDCG@10**: graded relevance using 1–5 ratings as grades
- **Nutritional Precision@5** (domain-specific): fraction of top-5 that actually satisfy the stated dietary constraint

### Expected Results
| Metric | BM25 | Embedding-only | Full Pipeline |
|--------|------|---------------|---------------|
| Recall@10 | 0.31 | 0.52 | 0.61 |
| MRR | 0.28 | 0.44 | 0.51 |
| NDCG@10 | 0.30 | 0.48 | 0.55 |
| Nutritional Precision@5 | 0.45 | 0.58 | **0.83** |

**The story:** Embeddings roughly double BM25 on semantic queries. Nutritional re-ranking adds the biggest gain specifically on the domain metric (0.58 → 0.83), which is exactly what the system is designed for.

---

## Implementation Order

1. `src/data/loader.py` + `cleaner.py` → clean parquet output
2. `notebooks/01_eda.ipynb` → understand distributions before setting thresholds
3. `src/data/feature_engineering.py` → flags + composite text
4. `src/embeddings/encoder.py` + `indexer.py` → build + persist FAISS index
5. `src/scoring/nutritional_scorer.py` → rule-based scoring
6. `src/embeddings/retriever.py` → query → filter → re-rank
7. `data/splits/` → build eval set
8. `src/evaluation/metrics.py` + `evaluator.py` → measure retrieval on all 3 systems
9. `src/rag/prompt_builder.py` + `generator.py` → LLM generation layer
10. `src/api/main.py` → FastAPI wrapper for demo
11. `notebooks/05_evaluation.ipynb` → final comparison across all systems

---

## Key Requirements
```
pandas==2.2.0  pyarrow==15.0.0  sentence-transformers==3.0.1
faiss-cpu==1.8.0  openai==1.30.0  fastapi==0.111.0
uvicorn==0.30.0  rank-bm25==0.2.2  ranx==0.3.16
pydantic==2.7.0  scikit-learn==1.5.0  numpy==1.26.4
matplotlib==3.9.0  seaborn==0.13.2  jupyter==1.0.0
pytest==8.2.0  python-dotenv==1.0.0
```

---

## Verification
- Run `notebooks/05_evaluation.ipynb` end-to-end to produce the metrics comparison table
- Test FastAPI: `POST /recommend {"query": "high protein low sugar dinner", "top_k": 5}`
- Verify Nutritional Precision@5 > 0.80 on eval set
- Verify BM25 baseline runs and scores lower than full pipeline on all metrics
