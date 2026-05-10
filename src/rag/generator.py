"""
generator.py — LLM generation layer for the RAG pipeline.

Supports both Gemini (default) and OpenAI via the OpenAI-compatible client.
Gemini is accessed through Google's OpenAI-compatible endpoint:
  base_url: https://generativelanguage.googleapis.com/v1beta/openai/
  model: gemini-2.0-flash
  key env var: GEMINI_API_KEY

To switch back to OpenAI, change config.yaml:
  model: gpt-4o-mini
  remove base_url line
  set OPENAI_API_KEY in .env
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.embeddings.retriever import RecipeRetriever
from src.rag.prompt_builder import build_messages
from src.scoring.explainer import build_recipe_explanation
from src.scoring.nutritional_scorer import score_recipe

load_dotenv()
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline: query → LLM response with grounded recipe explanations.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.recommend("high protein gluten free dinner")
        print(result["response"])        # LLM-generated recommendation
        print(result["recipes"])         # top-K retrieved recipes with scores
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.retriever = RecipeRetriever(config_path)
        self.llm_cfg = self.cfg["llm"]

        # Support both Gemini (via OpenAI-compatible endpoint) and OpenAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = self.llm_cfg.get("base_url")  # set in config for Gemini, absent for OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"RAG pipeline initialized. Model: {self.llm_cfg['model']}")

    def recommend(
        self,
        query: str,
        top_k: int = None,
        user_ingredients: list[str] = None,
    ) -> dict:
        """
        Full pipeline: natural-language query → ranked recommendations with explanations.

        Args:
            query: Natural-language user query.
            top_k: Number of recipes to return.
            user_ingredients: Optional list of ingredients the user already has.
                When provided, recipes are re-ranked to maximise ingredient overlap
                and the LLM prompt includes match/missing breakdown.

        Returns dict with:
          query               — original query
          dietary_constraints — parsed flag names
          user_ingredients    — passed through for context
          recipes             — list of recipe dicts with scores + explanations
          response            — LLM-generated recommendation text
        """
        from src.embeddings.retriever import parse_dietary_constraints

        dietary_constraints = parse_dietary_constraints(query)

        # Retrieve top-K candidates (with optional ingredient scoring)
        results_df = self.retriever.retrieve(
            query,
            top_k=top_k,
            dietary_constraints=dietary_constraints,
            user_ingredients=user_ingredients,
        )

        # Build per-recipe explanation text
        recipes = []
        for _, row in results_df.iterrows():
            _, explanations = score_recipe(row, dietary_constraints)
            explanation_text = build_recipe_explanation(row, explanations)
            recipe_dict = row.to_dict()
            recipe_dict["explanation"] = explanation_text
            recipes.append(recipe_dict)

        # Build prompt and call LLM
        messages = build_messages(query, recipes, dietary_constraints, user_ingredients)
        logger.info(f"Calling {self.llm_cfg['model']} for query: '{query}'")

        completion = self.client.chat.completions.create(
            model=self.llm_cfg["model"],
            messages=messages,
            max_tokens=self.llm_cfg["max_tokens"],
            temperature=self.llm_cfg["temperature"],
        )
        response_text = completion.choices[0].message.content

        return {
            "query": query,
            "dietary_constraints": dietary_constraints,
            "user_ingredients": user_ingredients,
            "recipes": recipes,
            "response": response_text,
        }
