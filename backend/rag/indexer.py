"""
indexer.py
-----------
Builds and caches a ChromaDB vector index from the curriculum markdown files in
backend/curriculum/. Each file covers one topic and contains the lesson,
rubrics, and worked examples in combined ## sections.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger("rag.indexer")

CURRICULUM_DIR  = Path(__file__).parent.parent / "curriculum"
CHROMA_DIR      = Path(__file__).parent / ".chromadb"
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Maps filename stem → display concept name used as metadata filter key
_CONCEPT_MAP: dict[str, str] = {
    "fractions":                "Fractions",
    "probability":              "Probability",
    "algebra":                  "Algebra",
    "geometry":                 "Geometry",
    "decimals_and_percentages": "Decimals and Percentages",
}

_HEADER_SPLITS = [("##", "section")]


def _load_curriculum_docs() -> list[Document]:
    """
    Reads every .md file in the curriculum directory, splits by ## headers,
    and attaches concept + section metadata to each chunk.
    """
    docs: list[Document] = []
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADER_SPLITS)

    for md_file in sorted(CURRICULUM_DIR.glob("*.md")):
        stem    = md_file.stem.lower()
        text    = md_file.read_text(encoding="utf-8")
        concept = _CONCEPT_MAP.get(stem, stem.replace("_", " ").title())

        chunks = splitter.split_text(text)
        for chunk in chunks:
            chunk.metadata["concept"] = concept
            chunk.metadata.setdefault("section", "General")
        docs.extend(chunks)
        logger.info("Loaded %d chunks from %s (concept=%r)", len(chunks), md_file.name, concept)

    return docs


_index = None   # ChromaDB singleton, loaded once per process


def get_index():
    """
    Returns the singleton ChromaDB index.
    - First run: builds from curriculum docs and persists to CHROMA_DIR.
    - Subsequent runs: loads from CHROMA_DIR (fast — no re-embedding needed).
    """
    global _index
    if _index is not None:
        return _index

    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "langchain-chroma is not installed. "
            "Run: pip install chromadb langchain-chroma"
        ) from exc

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # Load from disk if already built — avoids re-embedding on every restart.
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        logger.info("Loading curriculum index from %s ...", CHROMA_DIR)
        _index = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        count = _index._collection.count()
        logger.info("Curriculum index loaded: %d vectors.", count)
        return _index

    # First run — build and persist.
    logger.info("Building curriculum index with embed model=%r ...", EMBED_MODEL)
    docs = _load_curriculum_docs()

    if not docs:
        raise RuntimeError(
            f"No curriculum documents found in {CURRICULUM_DIR}. "
            "Add .md files to backend/curriculum/ and restart."
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _index = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("Curriculum index built and persisted: %d chunks.", len(docs))
    return _index


def reset_index() -> None:
    """
    Deletes the persisted index and resets the singleton. Call this after
    adding new curriculum files so the next get_index() rebuilds from scratch.
    """
    global _index
    _index = None
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        logger.info("ChromaDB index deleted — will rebuild on next get_index() call.")


def available_concepts() -> list[str]:
    """Sorted list of concept names the curriculum covers."""
    return sorted(_CONCEPT_MAP.values())
