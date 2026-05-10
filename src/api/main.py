"""
main.py — FastAPI application entrypoint.

Run with:
    uvicorn src.api.main:app --reload --port 8000

API docs auto-generated at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import logging

from fastapi import FastAPI

from src.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Recipe Recommendation System",
    description=(
        "RAG-powered recipe recommendation with semantic search and "
        "rule-based nutritional scoring for dietary profiles including "
        "diabetic-friendly, high-protein, gluten-free, and low-fat."
    ),
    version="1.0.0",
)

app.include_router(router)
