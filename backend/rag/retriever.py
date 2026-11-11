"""
retriever.py
-------------
Query interface used by agents to fetch relevant curriculum chunks.
Uses a structured query (concept + error_type + misconception) and a
concept-level ChromaDB filter. Returns "" silently on any failure so
the tutoring pipeline always degrades gracefully.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rag.retriever")


def get_curriculum_context(
    concept: str,
    error_type: str = "",
    misconception: str = "",
    k: int = 3,
) -> str:
    """
    Returns the top-k most relevant curriculum chunks for a concept as a
    formatted string ready to inject into an agent prompt.

    Args:
        concept:       Topic being taught, e.g. "Fractions".
        error_type:    Diagnostic classification ("Careless", "Conceptual", "Prerequisite").
        misconception: Specific misconception text from the diagnostic result.
        k:             Number of chunks to retrieve (default 3).
    """
    if not concept:
        return ""

    try:
        from backend.rag.indexer import get_index

        index = get_index()

        parts = [f"topic: {concept}"]
        if error_type and error_type not in ("None", "none", ""):
            parts.append(f"error type: {error_type}")
        if misconception:
            parts.append(f"misconception: {misconception}")
        query = " | ".join(parts)

        results = index.similarity_search(query, k=k, filter={"concept": concept})

        if not results:
            logger.debug("No RAG results for concept=%r, query=%r", concept, query)
            return ""

        formatted = "\n\n---\n\n".join(
            f"[{r.metadata.get('section', 'General')}]\n{r.page_content}"
            for r in results
        )
        logger.info("RAG retrieved %d chunks for concept=%r (error=%r)", len(results), concept, error_type)
        return formatted

    except Exception as exc:
        logger.warning("RAG retrieval failed (non-fatal): %s", exc)
        return ""
