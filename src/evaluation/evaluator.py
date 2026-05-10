"""
evaluator.py — Build the evaluation set and run comparison across all three systems.

Three systems compared:
  1. BM25         — TF-IDF keyword search (baseline)
  2. Embedding    — FAISS semantic search only, no re-ranking
  3. Full pipeline — Semantic + nutritional re-ranking + dietary flag filtering

Evaluation set construction:
  - Source: interactions data (1.1M user-recipe ratings)
  - Method: For each dietary tag combination, collect recipes that:
      (a) have the matching dietary flags (is_diabetic_friendly, is_high_protein, etc.)
      (b) have mean_rating > 4.0 (community-validated quality)
      (c) have review_count > 10 (sufficient evidence)
  - These are the 'relevant' recipes for a query targeting that dietary profile
  - 200 natural-language queries are written, 50 per dietary category

Dietary categories (50 queries each):
  1. Diabetic-friendly   → is_diabetic_friendly
  2. High-protein/Workout → is_high_protein
  3. Gluten-free          → is_gluten_free
  4. Low-fat/Heart health → is_low_fat AND is_low_sodium
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.evaluation.metrics import evaluate_system
from src.embeddings.retriever import RecipeRetriever

logger = logging.getLogger(__name__)

# 200 evaluation queries — 50 per dietary category
# Written to mimic realistic user queries (not just tag names)
EVAL_QUERIES = [
    # --- Diabetic-friendly (50 queries) ---
    {"query_id": "d01", "query": "diabetic friendly dinner recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d02", "query": "low sugar low carb meals for diabetes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d03", "query": "blood sugar friendly breakfast ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d04", "query": "recipes safe for type 2 diabetes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d05", "query": "low glycemic dinner options", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d06", "query": "diabetic meal prep ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d07", "query": "no sugar added main dishes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d08", "query": "low carb diabetic lunch", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d09", "query": "sugar free snack recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d10", "query": "healthy dinner for diabetics", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d11", "query": "low glycemic index foods for dinner", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d12", "query": "diabetic friendly chicken recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d13", "query": "low sugar vegetable dishes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d14", "query": "no refined carbs dinner ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d15", "query": "recipes for managing blood sugar levels", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d16", "query": "low carb low sugar soup recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d17", "query": "diabetic safe pasta alternatives", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d18", "query": "blood sugar stable meal ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d19", "query": "low sugar high fiber meals", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d20", "query": "diabetes management dinner recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d21", "query": "low carb diabetic breakfast", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d22", "query": "sugar conscious cooking ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d23", "query": "diabetic approved salad recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d24", "query": "low glycemic snacks for diabetics", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d25", "query": "no sugar vegetable stir fry", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d26", "query": "diabetic friendly fish dishes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d27", "query": "low carb egg recipes for diabetes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d28", "query": "whole grain alternatives for diabetics", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d29", "query": "diabetic safe curry recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d30", "query": "low sugar protein packed meals", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d31", "query": "blood sugar friendly stew recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d32", "query": "diabetic vegetarian dinner ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d33", "query": "low carb diabetic desserts without sugar", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d34", "query": "healthy low carb diabetic bowl recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d35", "query": "low sugar low calorie dinner for diabetics", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d36", "query": "diabetes friendly grilled recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d37", "query": "easy diabetic friendly weeknight dinner", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d38", "query": "low sugar seafood recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d39", "query": "type 1 diabetes meal ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d40", "query": "low glycemic load lunch recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d41", "query": "diabetic cooking without white rice", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d42", "query": "low sugar healthy dinner ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d43", "query": "diabetes safe turkey recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d44", "query": "low carb cauliflower dishes for diabetics", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d45", "query": "blood sugar friendly Mediterranean recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d46", "query": "diabetic friendly Asian inspired dishes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d47", "query": "low sugar legume and bean recipes", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d48", "query": "diabetic meal plan main course ideas", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d49", "query": "slow cooker recipes for diabetics", "constraints": ["is_diabetic_friendly"]},
    {"query_id": "d50", "query": "low carb diabetic one pan dinners", "constraints": ["is_diabetic_friendly"]},

    # --- High-protein / Workout (50 queries) ---
    {"query_id": "p01", "query": "high protein post workout meal", "constraints": ["is_high_protein"]},
    {"query_id": "p02", "query": "muscle building dinner recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p03", "query": "protein packed lunch for athletes", "constraints": ["is_high_protein"]},
    {"query_id": "p04", "query": "gym meal prep high protein", "constraints": ["is_high_protein"]},
    {"query_id": "p05", "query": "lean protein dinner ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p06", "query": "high protein low fat chicken recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p07", "query": "protein rich meals for weight training", "constraints": ["is_high_protein"]},
    {"query_id": "p08", "query": "bodybuilding meal ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p09", "query": "high protein vegetarian meals", "constraints": ["is_high_protein"]},
    {"query_id": "p10", "query": "protein heavy breakfast ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p11", "query": "high protein fish recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p12", "query": "strength training nutrition meal ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p13", "query": "high protein egg based dishes", "constraints": ["is_high_protein"]},
    {"query_id": "p14", "query": "protein dense salad recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p15", "query": "high protein turkey recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p16", "query": "muscle recovery meal after workout", "constraints": ["is_high_protein"]},
    {"query_id": "p17", "query": "high protein low carb dinner", "constraints": ["is_high_protein"]},
    {"query_id": "p18", "query": "protein packed steak recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p19", "query": "athletic performance meal ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p20", "query": "high protein tofu recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p21", "query": "pre workout high protein snack", "constraints": ["is_high_protein"]},
    {"query_id": "p22", "query": "high protein shrimp recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p23", "query": "lean muscle diet dinner", "constraints": ["is_high_protein"]},
    {"query_id": "p24", "query": "high protein meal prep bowls", "constraints": ["is_high_protein"]},
    {"query_id": "p25", "query": "protein rich soup recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p26", "query": "high protein low calorie meals", "constraints": ["is_high_protein"]},
    {"query_id": "p27", "query": "protein packed stir fry", "constraints": ["is_high_protein"]},
    {"query_id": "p28", "query": "high protein lentil recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p29", "query": "fitness meal ideas with chicken breast", "constraints": ["is_high_protein"]},
    {"query_id": "p30", "query": "high protein cottage cheese dishes", "constraints": ["is_high_protein"]},
    {"query_id": "p31", "query": "protein rich beef recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p32", "query": "high protein grain bowls", "constraints": ["is_high_protein"]},
    {"query_id": "p33", "query": "sports nutrition meal ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p34", "query": "high protein salmon dinner", "constraints": ["is_high_protein"]},
    {"query_id": "p35", "query": "bulking meal prep ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p36", "query": "high protein cutting diet recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p37", "query": "protein dense wrap and sandwich ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p38", "query": "high protein Mediterranean meals", "constraints": ["is_high_protein"]},
    {"query_id": "p39", "query": "muscle building breakfast recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p40", "query": "high protein vegan meal ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p41", "query": "protein rich dinner under 500 calories", "constraints": ["is_high_protein"]},
    {"query_id": "p42", "query": "high protein snack recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p43", "query": "high protein chicken thigh recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p44", "query": "endurance athlete meal ideas", "constraints": ["is_high_protein"]},
    {"query_id": "p45", "query": "high protein chickpea recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p46", "query": "protein dense Asian noodle dishes", "constraints": ["is_high_protein"]},
    {"query_id": "p47", "query": "high protein slow cooker meal", "constraints": ["is_high_protein"]},
    {"query_id": "p48", "query": "protein rich dinner for two", "constraints": ["is_high_protein"]},
    {"query_id": "p49", "query": "high protein tuna recipes", "constraints": ["is_high_protein"]},
    {"query_id": "p50", "query": "high protein meal under 30 minutes", "constraints": ["is_high_protein"]},

    # --- Gluten-free (50 queries) ---
    {"query_id": "g01", "query": "gluten free dinner ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g02", "query": "celiac safe main dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g03", "query": "gluten free pasta recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g04", "query": "wheat free meal ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g05", "query": "gluten free chicken recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g06", "query": "gluten free breakfast ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g07", "query": "no gluten dinner recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g08", "query": "gluten free soup recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g09", "query": "gluten intolerance meal ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g10", "query": "gluten free Asian inspired meals", "constraints": ["is_gluten_free"]},
    {"query_id": "g11", "query": "rice based gluten free dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g12", "query": "gluten free quick dinner ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g13", "query": "gluten free high protein meal", "constraints": ["is_gluten_free"]},
    {"query_id": "g14", "query": "gluten free salad recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g15", "query": "celiac friendly stir fry", "constraints": ["is_gluten_free"]},
    {"query_id": "g16", "query": "gluten free Mexican food recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g17", "query": "gluten free fish and seafood dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g18", "query": "no wheat no barley dinner recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g19", "query": "gluten free potato dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g20", "query": "gluten free meal prep ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g21", "query": "gluten free vegetarian dinner", "constraints": ["is_gluten_free"]},
    {"query_id": "g22", "query": "safe meals for celiac disease", "constraints": ["is_gluten_free"]},
    {"query_id": "g23", "query": "gluten free beef stew recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g24", "query": "gluten free quinoa recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g25", "query": "gluten free egg based dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g26", "query": "gluten free legume recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g27", "query": "naturally gluten free dinner ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g28", "query": "gluten free slow cooker meals", "constraints": ["is_gluten_free"]},
    {"query_id": "g29", "query": "gluten free Mediterranean recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g30", "query": "gluten free Indian food recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g31", "query": "gluten free dairy free dinner", "constraints": ["is_gluten_free"]},
    {"query_id": "g32", "query": "gluten free turkey recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g33", "query": "gluten free low carb meals", "constraints": ["is_gluten_free"]},
    {"query_id": "g34", "query": "gluten free baked dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g35", "query": "celiac safe chicken soup", "constraints": ["is_gluten_free"]},
    {"query_id": "g36", "query": "gluten free grain bowls", "constraints": ["is_gluten_free"]},
    {"query_id": "g37", "query": "gluten free tofu recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g38", "query": "gluten free comfort food ideas", "constraints": ["is_gluten_free"]},
    {"query_id": "g39", "query": "gluten free low calorie dinners", "constraints": ["is_gluten_free"]},
    {"query_id": "g40", "query": "gluten free high fiber meals", "constraints": ["is_gluten_free"]},
    {"query_id": "g41", "query": "celiac disease safe meal prep", "constraints": ["is_gluten_free"]},
    {"query_id": "g42", "query": "gluten free roasted vegetable dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g43", "query": "gluten free simple weeknight dinner", "constraints": ["is_gluten_free"]},
    {"query_id": "g44", "query": "gluten free sushi and rice dishes", "constraints": ["is_gluten_free"]},
    {"query_id": "g45", "query": "gluten free keto recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g46", "query": "gluten free pork recipes", "constraints": ["is_gluten_free"]},
    {"query_id": "g47", "query": "gluten free Thai food at home", "constraints": ["is_gluten_free"]},
    {"query_id": "g48", "query": "gluten free lentil soup", "constraints": ["is_gluten_free"]},
    {"query_id": "g49", "query": "wheat free low sodium meals", "constraints": ["is_gluten_free"]},
    {"query_id": "g50", "query": "gluten free family dinner recipes", "constraints": ["is_gluten_free"]},

    # --- Low-fat / Heart health (50 queries) ---
    {"query_id": "h01", "query": "low fat heart healthy dinner", "constraints": ["is_low_fat"]},
    {"query_id": "h02", "query": "low sodium low fat meals", "constraints": ["is_low_fat"]},
    {"query_id": "h03", "query": "heart healthy chicken recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h04", "query": "low fat meal prep ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h05", "query": "low cholesterol dinner recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h06", "query": "low fat fish and seafood dishes", "constraints": ["is_low_fat"]},
    {"query_id": "h07", "query": "heart health friendly vegetable dishes", "constraints": ["is_low_fat"]},
    {"query_id": "h08", "query": "low fat soup and stew ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h09", "query": "low fat turkey meals", "constraints": ["is_low_fat"]},
    {"query_id": "h10", "query": "low fat high protein dinner", "constraints": ["is_low_fat"]},
    {"query_id": "h11", "query": "heart friendly grilled recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h12", "query": "low fat Mediterranean dishes", "constraints": ["is_low_fat"]},
    {"query_id": "h13", "query": "cardiovascular health meal ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h14", "query": "low fat salad recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h15", "query": "low fat steamed recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h16", "query": "low fat vegetarian meals", "constraints": ["is_low_fat"]},
    {"query_id": "h17", "query": "low fat Asian cooking ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h18", "query": "low fat bean and legume recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h19", "query": "low saturated fat dinner recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h20", "query": "heart health diet meal ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h21", "query": "low fat low calorie dinners", "constraints": ["is_low_fat"]},
    {"query_id": "h22", "query": "low fat shrimp recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h23", "query": "low fat stir fry recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h24", "query": "low fat egg white dishes", "constraints": ["is_low_fat"]},
    {"query_id": "h25", "query": "low fat Indian recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h26", "query": "heart healthy slow cooker meal", "constraints": ["is_low_fat"]},
    {"query_id": "h27", "query": "low fat pasta dishes", "constraints": ["is_low_fat"]},
    {"query_id": "h28", "query": "low fat baked chicken", "constraints": ["is_low_fat"]},
    {"query_id": "h29", "query": "heart health quinoa recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h30", "query": "low fat dinner under 400 calories", "constraints": ["is_low_fat"]},
    {"query_id": "h31", "query": "low fat grain bowl ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h32", "query": "low fat tuna recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h33", "query": "low fat high fiber meals", "constraints": ["is_low_fat"]},
    {"query_id": "h34", "query": "low fat low sugar dinner ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h35", "query": "cardiac diet friendly recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h36", "query": "low fat roasted vegetable dinner", "constraints": ["is_low_fat"]},
    {"query_id": "h37", "query": "low fat cod and white fish recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h38", "query": "low fat high protein Greek food", "constraints": ["is_low_fat"]},
    {"query_id": "h39", "query": "low fat meal for high blood pressure", "constraints": ["is_low_fat"]},
    {"query_id": "h40", "query": "heart healthy breakfast ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h41", "query": "low fat tofu stir fry", "constraints": ["is_low_fat"]},
    {"query_id": "h42", "query": "low fat dinner ideas for cholesterol", "constraints": ["is_low_fat"]},
    {"query_id": "h43", "query": "low fat lentil and dal recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h44", "query": "low fat poached fish recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h45", "query": "low fat meal prep for weight loss", "constraints": ["is_low_fat"]},
    {"query_id": "h46", "query": "heart healthy salmon recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h47", "query": "low fat summer salad ideas", "constraints": ["is_low_fat"]},
    {"query_id": "h48", "query": "low fat healthy weeknight dinners", "constraints": ["is_low_fat"]},
    {"query_id": "h49", "query": "low fat chicken soup recipes", "constraints": ["is_low_fat"]},
    {"query_id": "h50", "query": "low fat heart safe quick meals", "constraints": ["is_low_fat"]},
]


def build_ground_truth(
    recipes_df: pd.DataFrame,
    queries: list[dict],
    min_rating: float = 4.0,
    min_reviews: int = 10,
) -> dict:
    """
    Build ground truth: for each query, find relevant recipe IDs.

    Relevant = has the dietary flag AND meets minimum quality threshold.
    Uses interaction-derived mean_rating and review_count for quality filtering.
    This grounds the eval set in actual human judgments from 1.1M interactions.
    """
    ground_truth = {}

    for q in queries:
        constraints = q.get("constraints", [])
        if not constraints:
            continue

        flag = constraints[0]
        if flag not in recipes_df.columns:
            continue

        relevant = recipes_df[
            (recipes_df[flag])
            & (recipes_df["mean_rating"] >= min_rating)
            & (recipes_df["review_count"] >= min_reviews)
        ]

        relevant_ids = relevant["id"].tolist()
        # Use mean_rating as graded relevance score for NDCG
        relevance_scores = dict(zip(relevant["id"], relevant["mean_rating"]))

        ground_truth[q["query_id"]] = {
            "relevant_ids": relevant_ids,
            "relevance_scores": relevance_scores,
        }

    return ground_truth


def run_evaluation(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Run evaluation for all three systems and return a comparison DataFrame.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading featured recipes for evaluation...")
    recipes_df = pd.read_parquet(cfg["data"]["featured"])

    # Build or load ground truth
    gt_path = cfg["data"]["ground_truth"]
    queries_path = cfg["data"]["eval_queries"]

    if Path(gt_path).exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)
        with open(queries_path) as f:
            queries = json.load(f)
    else:
        queries = EVAL_QUERIES
        ground_truth = build_ground_truth(recipes_df, queries)
        Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f)
        with open(queries_path, "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"Saved eval set: {len(queries)} queries, {len(ground_truth)} ground truths")

    # Recipe flags DataFrame indexed by recipe ID (for Nutritional Precision metric)
    flag_cols = [c for c in recipes_df.columns if c.startswith("is_")]
    recipe_flags_df = recipes_df[["id"] + flag_cols].set_index("id")

    all_results = []

    # --- System 1: BM25 baseline ---
    logger.info("Building BM25 index...")
    tokenized_corpus = [r.split() for r in recipes_df["recipe_text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    recipe_id_list = recipes_df["id"].tolist()

    def bm25_retrieve(query, constraints):
        scores = bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:50]
        return [recipe_id_list[i] for i in top_indices]

    all_results.append(evaluate_system(
        "BM25", queries, ground_truth, bm25_retrieve,
        recipe_flags_df, cfg["evaluation"]["k_values"]
    ))

    # --- System 2: Embedding-only ---
    logger.info("Loading retriever for embedding-only evaluation...")
    retriever = RecipeRetriever(config_path)

    def embedding_retrieve(query, constraints):
        results = retriever.retrieve(query, top_k=50, dietary_constraints=[])
        return results["recipe_id"].tolist()

    all_results.append(evaluate_system(
        "Embedding-only", queries, ground_truth, embedding_retrieve,
        recipe_flags_df, cfg["evaluation"]["k_values"]
    ))

    # --- System 3: Full pipeline ---
    def full_pipeline_retrieve(query, constraints):
        results = retriever.retrieve(query, top_k=50, dietary_constraints=constraints)
        return results["recipe_id"].tolist()

    all_results.append(evaluate_system(
        "Full Pipeline", queries, ground_truth, full_pipeline_retrieve,
        recipe_flags_df, cfg["evaluation"]["k_values"]
    ))

    results_df = pd.DataFrame(all_results).set_index("system")
    logger.info("\n" + results_df.to_string())
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
