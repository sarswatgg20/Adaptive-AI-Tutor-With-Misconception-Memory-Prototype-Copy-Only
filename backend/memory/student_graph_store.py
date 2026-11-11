"""
student_graph_store.py
-----------------------
Loads and saves each student's mastery graph in Neo4j.

Role
====
Reshapes to/from the exact `student_memory_profile` dict shape used
everywhere else in this codebase (`concept_mastery`, `weak_prerequisites`,
`recent_attempts` — see backend/agents/orchestrator.py's TutorState and
backend/memory/student_profile.py's run_memory_update). Every agent,
prompt, and the LangGraph pipeline itself only ever sees this same dict
shape — this module is the *only* place that knows Neo4j exists at all.

Data model
==========
    (:Student {id})
    (:Student)-[:MASTERY {mastery, consecutive_misses}]->(:Concept {name})
    (:Student)-[:WEAK_IN]->(:Concept {name})

`recent_attempts` is a small rolling window (last 10), not itself a useful
graph relationship, so it's stored as a JSON string property on the
Student node (`recent_attempts_json`) and decoded/encoded here.

Every function degrades safely (logs a warning, returns/no-ops) if Neo4j is
unreachable — persistence is additive, never a hard dependency for a
tutoring turn to complete. Callers (app.py, student_profile.py) don't need
their own try/except around these calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.memory.neo4j_client import is_available, run_query

logger = logging.getLogger("student_graph_store")
logging.basicConfig(level=logging.INFO)

DEFAULT_MASTERY = 0.3


def _empty_profile(student_id: str) -> dict:
    return {
        "student_id": student_id,
        "concept_mastery": {},
        "weak_prerequisites": [],
        "recent_attempts": [],
    }


def load_student_profile(student_id: str) -> dict:
    """
    Loads a student's mastery graph from Neo4j and reshapes it into the
    standard student_memory_profile dict. Returns a fresh empty-shaped
    profile (same shape app.py has always defaulted to) if Neo4j is
    unreachable or the student has no data yet.
    """
    profile = _empty_profile(student_id)

    if not is_available():
        logger.warning("Neo4j unavailable — returning a fresh profile for %r.", student_id)
        return profile

    try:
        mastery_rows = run_query(
            """
            MATCH (s:Student {id: $student_id})-[m:MASTERY]->(c:Concept)
            RETURN c.name AS concept, m.mastery AS mastery, m.consecutive_misses AS consecutive_misses
            """,
            student_id=student_id,
        )
        for row in mastery_rows:
            profile["concept_mastery"][row["concept"]] = {
                "mastery": row["mastery"] if row["mastery"] is not None else DEFAULT_MASTERY,
                "consecutive_misses": row["consecutive_misses"] or 0,
            }

        weak_rows = run_query(
            """
            MATCH (s:Student {id: $student_id})-[:WEAK_IN]->(c:Concept)
            RETURN c.name AS concept
            """,
            student_id=student_id,
        )
        profile["weak_prerequisites"] = [row["concept"] for row in weak_rows]

        student_rows = run_query(
            "MATCH (s:Student {id: $student_id}) RETURN s.recent_attempts_json AS recent_attempts_json",
            student_id=student_id,
        )
        if student_rows and student_rows[0].get("recent_attempts_json"):
            profile["recent_attempts"] = json.loads(student_rows[0]["recent_attempts_json"])

    except Exception as exc:  # noqa: BLE001 - a read failure should return a safe empty profile, not crash
        logger.warning("load_student_profile(%r) failed: %s", student_id, exc)
        return _empty_profile(student_id)

    return profile


def save_student_profile(student_id: str, profile: dict) -> None:
    """
    Persists the (already-updated, e.g. by student_profile.py's
    run_memory_update) profile dict back to Neo4j. No-ops with a logged
    warning if Neo4j is unreachable — this must never raise into the
    caller's tutoring turn.
    """
    if not is_available():
        logger.warning("Neo4j unavailable — skipping profile save for %r.", student_id)
        return

    try:
        run_query("MERGE (s:Student {id: $student_id})", student_id=student_id)

        for concept, data in (profile.get("concept_mastery") or {}).items():
            run_query(
                """
                MERGE (s:Student {id: $student_id})
                MERGE (c:Concept {name: $concept})
                MERGE (s)-[m:MASTERY]->(c)
                SET m.mastery = $mastery, m.consecutive_misses = $consecutive_misses
                """,
                student_id=student_id,
                concept=concept,
                mastery=data.get("mastery", DEFAULT_MASTERY),
                consecutive_misses=data.get("consecutive_misses", 0),
            )

        for concept in profile.get("weak_prerequisites") or []:
            run_query(
                """
                MERGE (s:Student {id: $student_id})
                MERGE (c:Concept {name: $concept})
                MERGE (s)-[:WEAK_IN]->(c)
                """,
                student_id=student_id,
                concept=concept,
            )

        run_query(
            "MATCH (s:Student {id: $student_id}) SET s.recent_attempts_json = $recent_attempts_json",
            student_id=student_id,
            recent_attempts_json=json.dumps(profile.get("recent_attempts") or []),
        )

    except Exception as exc:  # noqa: BLE001 - persistence failures must not break the turn
        logger.warning("save_student_profile(%r) failed: %s", student_id, exc)


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python backend/memory/student_graph_store.py

    Requires a running Neo4j instance to see real round-trip behavior;
    otherwise this demonstrates the safe-fallback path.
    """
    test_student_id = "stu_test_1024"

    test_profile = {
        "student_id": test_student_id,
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.42, "consecutive_misses": 1},
        },
        "weak_prerequisites": ["Common Denominators"],
        "recent_attempts": [{"concept": "Fraction Addition", "correct": False}],
    }

    print("=" * 80)
    if not is_available():
        print("Neo4j is unavailable — save/load below will safely no-op / return empty.")
    print(f"Saving profile for {test_student_id}...")
    save_student_profile(test_student_id, test_profile)

    print(f"Loading profile for {test_student_id}...")
    loaded = load_student_profile(test_student_id)
    print(json.dumps(loaded, indent=2))
    print("=" * 80)
