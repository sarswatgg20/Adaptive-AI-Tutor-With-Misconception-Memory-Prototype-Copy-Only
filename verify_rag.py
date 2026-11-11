"""
RAG verification script — run from project root:
    python verify_rag.py
Tests each layer independently so you can see exactly where a failure occurs.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def check(label, fn):
    try:
        detail = fn()
        results.append((PASS, label, detail or ""))
        print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))
    except Exception as exc:
        results.append((FAIL, label, str(exc)))
        print(f"  [FAIL] {label} — {exc}")


# ── 1. Dependencies ────────────────────────────────────────────────────────────
print("\n=== 1. Dependencies ===")

def _check_chromadb():
    import chromadb  # noqa: F401
    return f"chromadb version: {chromadb.__version__}"

def _check_langchain_chroma():
    from langchain_chroma import Chroma  # noqa: F401
    return "langchain_chroma.Chroma importable"

def _check_text_splitters():
    from langchain_text_splitters import MarkdownHeaderTextSplitter  # noqa: F401
    return "MarkdownHeaderTextSplitter importable"

def _check_ollama_embeddings():
    from langchain_ollama import OllamaEmbeddings  # noqa: F401
    return "OllamaEmbeddings importable"

check("chromadb", _check_chromadb)
check("langchain-chroma", _check_langchain_chroma)
check("langchain-text-splitters", _check_text_splitters)
check("langchain-ollama embeddings", _check_ollama_embeddings)


# ── 2. Curriculum files ────────────────────────────────────────────────────────
print("\n=== 2. Curriculum files ===")

from pathlib import Path
CURRICULUM_DIR = Path(__file__).parent / "backend" / "curriculum"
EXPECTED = ["fractions.md", "probability.md", "algebra.md",
            "geometry.md", "decimals_and_percentages.md"]

for fname in EXPECTED:
    path = CURRICULUM_DIR / fname
    def _check_file(p=path, n=fname):
        size = p.stat().st_size
        return f"{size} bytes"
    check(fname, _check_file if path.exists() else lambda n=fname: (_ for _ in ()).throw(FileNotFoundError(f"{n} not found")))


# ── 3. Document loading & chunking ─────────────────────────────────────────────
print("\n=== 3. Document loading & chunking ===")

def _check_doc_loading():
    from backend.rag.indexer import _load_curriculum_docs
    docs = _load_curriculum_docs()
    if not docs:
        raise RuntimeError("No documents loaded")
    concepts = set(d.metadata.get("concept") for d in docs)
    return f"{len(docs)} chunks across concepts: {sorted(concepts)}"

check("_load_curriculum_docs()", _check_doc_loading)


# ── 4. Ollama connectivity ─────────────────────────────────────────────────────
print("\n=== 4. Ollama connectivity ===")

def _check_ollama_ping():
    import urllib.request
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
        import json
        data = json.loads(r.read())
    models = [m["name"] for m in data.get("models", [])]
    return f"Ollama running. Models: {models}"

check("Ollama reachable", _check_ollama_ping)


# ── 5. Embedding model availability ───────────────────────────────────────────
print("\n=== 5. Embedding model ===")

def _check_embed_model():
    import urllib.request, json
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
        data = json.loads(r.read())
    models = [m["name"] for m in data.get("models", [])]
    # match by prefix (e.g. "nomic-embed-text:latest" matches "nomic-embed-text")
    matched = [m for m in models if m.startswith(embed_model.split(":")[0])]
    if not matched:
        raise RuntimeError(
            f"'{embed_model}' not found in Ollama. Available: {models}\n"
            f"  Fix: ollama pull {embed_model}  OR  set EMBED_MODEL=<another model> in .env"
        )
    return f"Found: {matched}"

check(f"embed model ({os.getenv('EMBED_MODEL', 'nomic-embed-text')})", _check_embed_model)


# ── 6. ChromaDB index build ────────────────────────────────────────────────────
print("\n=== 6. ChromaDB index build ===")

def _check_index_build():
    import backend.rag.indexer as idx_mod
    idx_mod._index = None
    index = idx_mod.get_index()
    count = index._collection.count()
    return f"Index loaded: {count} vectors"

check("get_index() (ChromaDB)", _check_index_build)


# ── 7. Retrieval ───────────────────────────────────────────────────────────────
print("\n=== 7. Retrieval ===")

TEST_CASES = [
    {
        "concept": "Fractions",
        "error_type": "Conceptual",
        "misconception": "Adds numerators and denominators directly",
    },
    {
        "concept": "Probability",
        "error_type": "Conceptual",
        "misconception": "Gambler's fallacy — past outcomes affect future independent events",
    },
    {
        "concept": "Algebra",
        "error_type": "Prerequisite",
        "misconception": "Cannot isolate variable correctly",
    },
]

def _make_retrieval_check(tc):
    def _check():
        from backend.rag.retriever import get_curriculum_context
        ctx = get_curriculum_context(
            concept=tc["concept"],
            error_type=tc["error_type"],
            misconception=tc["misconception"],
            k=2,
        )
        if not ctx:
            raise RuntimeError("Empty context returned — check concept filter or embedding model")
        preview = ctx[:120].replace("\n", " ")
        return f"{len(ctx)} chars | preview: {preview!r}..."
    return _check

for tc in TEST_CASES:
    label = f"retrieve({tc['concept']!r}, {tc['error_type']!r})"
    check(label, _make_retrieval_check(tc))


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")

if failed:
    print("\nFailed checks:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"  - {label}: {detail}")
    sys.exit(1)
else:
    print("\nAll checks passed — RAG pipeline is working correctly.")
