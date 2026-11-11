"""
misconception_graph.py
----------------------
Manages the prerequisite dependency graph for the curriculum concepts, backed
by Neo4j. Helps the Diagnostic Agent (backend/agents/diagnostic_agent.py) and
Tutor Planner identify missing prerequisite skills using an actual curriculum
structure instead of letting the LLM invent prerequisite names from scratch
each turn.

Data model
==========
    (:Concept {name})-[:REQUIRES]->(:Concept {name})

`SEED_CONCEPT_DEPENDENCIES` below is the initial curriculum graph — call
`seed_concept_graph()` once against a fresh Neo4j instance to load it in.
It is a seed for the graph, not the runtime source of truth: once seeded,
`get_prerequisites`/`get_deep_prerequisites` read from Neo4j, so edits to
the curriculum should happen via Cypher (or by re-running the seed) rather
than by editing this dict after the first seed.

Every function here degrades safely (logs a warning, returns `[]`) if Neo4j
is unreachable, matching the fallback contract used throughout backend/memory
and backend/agents — a missing graph must never crash a tutoring turn.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from backend.memory.neo4j_client import is_available, run_query

logger = logging.getLogger("misconception_graph")
logging.basicConfig(level=logging.INFO)

# Initial curriculum seed data: concept -> list of direct prerequisites.
SEED_CONCEPT_DEPENDENCIES: Dict[str, List[str]] = {
    "Fraction Addition": ["Comparing Unit Fractions", "Common Denominators"],
    "Comparing Unit Fractions": ["Part-Whole Division", "Understanding Denominators"],
    "Algebra": ["Arithmetic Operations", "Fractions"],
}


def seed_concept_graph(dependencies: Dict[str, List[str]] = SEED_CONCEPT_DEPENDENCIES) -> None:
    """
    One-time (idempotent) setup: MERGEs a Concept node per key/value and a
    REQUIRES relationship for each dependency. Safe to re-run — MERGE means
    running it twice doesn't create duplicates.

    Run this once against a fresh Neo4j instance, e.g.:
        $ python backend/memory/misconception_graph.py --seed
    """
    if not is_available():
        logger.warning("Neo4j unavailable — skipping concept graph seeding.")
        return

    for concept, prereqs in dependencies.items():
        run_query("MERGE (c:Concept {name: $name})", name=concept)
        for prereq in prereqs:
            run_query(
                """
                MERGE (c:Concept {name: $concept})
                MERGE (p:Concept {name: $prereq})
                MERGE (c)-[:REQUIRES]->(p)
                """,
                concept=concept,
                prereq=prereq,
            )
    logger.info("Seeded concept graph with %d concepts.", len(dependencies))


def get_prerequisites(concept: str) -> List[str]:
    """
    Get direct prerequisites for a given concept from the Neo4j graph.
    Returns [] if Neo4j is unreachable or the concept has none on record.
    """
    if not is_available():
        return []

    try:
        rows = run_query(
            """
            MATCH (c:Concept)-[:REQUIRES]->(p:Concept)
            WHERE toLower(c.name) = toLower($concept)
            RETURN p.name AS name
            """,
            concept=concept,
        )
        return [row["name"] for row in rows]
    except Exception as exc:  # noqa: BLE001 - never let a graph hiccup break the caller
        logger.warning("get_prerequisites(%r) failed: %s", concept, exc)
        return []


def get_deep_prerequisites(concept: str) -> List[str]:
    """
    Recursively fetch all deep prerequisites for a given concept — the
    entire REQUIRES chain, however many levels deep, deduplicated. Neo4j's
    variable-length path match (`*1..`) does the traversal natively, which
    is what the original in-memory recursive DFS version of this function
    was doing by hand.
    """
    if not is_available():
        return []

    try:
        rows = run_query(
            """
            MATCH (c:Concept)-[:REQUIRES*1..]->(p:Concept)
            WHERE toLower(c.name) = toLower($concept)
            RETURN DISTINCT p.name AS name
            """,
            concept=concept,
        )
        return [row["name"] for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_deep_prerequisites(%r) failed: %s", concept, exc)
        return []


def get_all_concepts() -> List[str]:
    """
    Every Concept node's name in the curriculum graph — the full node set for
    a visualization (backend/memory/graph_viz.py). Returns [] if Neo4j is
    unreachable or nothing has been seeded yet.
    """
    if not is_available():
        return []

    try:
        rows = run_query("MATCH (c:Concept) RETURN c.name AS name")
        return [row["name"] for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_all_concepts() failed: %s", exc)
        return []


def get_all_edges() -> List[Tuple[str, str]]:
    """
    Every (concept, prerequisite) REQUIRES edge in the curriculum graph —
    the full edge set for a visualization (backend/memory/graph_viz.py).
    Returns [] if Neo4j is unreachable or nothing has been seeded yet.
    """
    if not is_available():
        return []

    try:
        rows = run_query(
            "MATCH (c:Concept)-[:REQUIRES]->(p:Concept) RETURN c.name AS concept, p.name AS prereq"
        )
        return [(row["concept"], row["prereq"]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_all_edges() failed: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Standalone test / one-time seed entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python backend/memory/misconception_graph.py --seed   # populate a fresh instance
    $ python backend/memory/misconception_graph.py          # just run the lookups
    """
    import sys

    if "--seed" in sys.argv:
        seed_concept_graph()

    print("=" * 80)
    if not is_available():
        print("Neo4j is unavailable — get_prerequisites/get_deep_prerequisites")
        print("will return [] below. This is expected without a live instance.")
    for concept in ["Fraction Addition", "Algebra", "Comparing Unit Fractions"]:
        print(f"\nConcept: {concept}")
        print(f"  direct prerequisites: {get_prerequisites(concept)}")
        print(f"  deep prerequisites:   {get_deep_prerequisites(concept)}")

    print("\nAll concepts:", get_all_concepts())
    print("All edges (concept -> prerequisite):", get_all_edges())
    print("=" * 80)
