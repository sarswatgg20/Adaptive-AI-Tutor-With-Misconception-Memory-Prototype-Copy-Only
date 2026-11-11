"""
neo4j_client.py
----------------
Thin Neo4j driver wrapper shared by every graph-backed memory module
(misconception_graph.py, student_graph_store.py).

Role
====
Owns the one thing every other module in this package needs and shouldn't
each reimplement: a lazily-created driver singleton, a `run_query` helper
that hides session/transaction boilerplate, and an `is_available()` check
so callers can degrade gracefully instead of crashing a tutoring turn when
Neo4j isn't reachable (e.g. not provisioned yet, or briefly down).

Nothing in this module raises on connection failure except `run_query`
itself — callers are expected to check `is_available()` first, or catch
exceptions around `run_query` the same way the rest of this codebase
already guards LLM calls (see e.g. escalation_agent.py's fallback pattern).

Tech stack: Python 3.10+, the official `neo4j` driver, Bolt protocol.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

logger = logging.getLogger("neo4j_client")
logging.basicConfig(level=logging.INFO)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    """Lazily creates and reuses a single driver for the process lifetime."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def is_available() -> bool:
    """
    Cheap connectivity check. Callers should use this to decide whether to
    hit the graph at all, rather than letting a connection error surface
    mid-turn.
    """
    try:
        get_driver().verify_connectivity()
        return True
    except Exception as exc:  # noqa: BLE001 - any driver/connection error counts as "unavailable"
        logger.warning("Neo4j is not reachable at %s: %s", NEO4J_URI, exc)
        return False


def run_query(cypher: str, **params: Any) -> list[dict]:
    """
    Runs a single Cypher statement in its own session/transaction and
    returns the results as a list of plain dicts.

    Raises whatever the underlying driver raises on failure — callers that
    need to degrade gracefully should guard this with `is_available()` or a
    try/except, matching the fallback pattern used elsewhere in
    backend/memory and backend/agents.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, **params)
        return [record.data() for record in result]


def close_driver() -> None:
    """Closes the singleton driver, if one was ever created. Mostly for tests."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python backend/memory/neo4j_client.py

    Requires a running Neo4j instance reachable at NEO4J_URI. If none is
    available, this should print "unavailable" rather than raising.
    """
    print("=" * 80)
    print(f"Checking connectivity to {NEO4J_URI} ...")
    if is_available():
        print("Neo4j is reachable. Running a trivial query...")
        rows = run_query("RETURN 1 AS one")
        print(rows)
    else:
        print("Neo4j is unavailable — this is expected if you haven't provisioned")
        print("an instance yet. Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env")
        print("once you have one (Neo4j Desktop or the free Aura tier both work).")
    print("=" * 80)
