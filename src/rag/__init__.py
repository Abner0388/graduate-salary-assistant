"""
RAG (Retrieval-Augmented Generation) module.
Uses TF-IDF similarity to retrieve relevant student profiles
and aggregate statistics from the training dataset.
"""
from .corpus import load_corpus, query_to_text
from .retriever import (
    retrieve_similar_profiles,
    retrieve_aggregate_stats,
    format_retrieved_context,
)

__all__ = [
    "load_corpus",
    "query_to_text",
    "retrieve_similar_profiles",
    "retrieve_aggregate_stats",
    "format_retrieved_context",
]
