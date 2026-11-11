"""
students.py
------------
The demo student registry — closes the "everything is hardcoded to one
demo_student" gap the roadmap calls out. A real login system is out of
scope for a hackathon demo; a small fixed roster is enough to make the
Teacher Dashboard read as a real classroom instead of a single-user toy.

No new persistence mechanism: seeding just calls the existing
student_graph_store.save_student_profile and spaced_repetition.
schedule_next_review functions for the two non-interactive demo students,
so the roster/mastery/review-schedule views have real data immediately
without anyone needing to chat as them first.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("students")
logging.basicConfig(level=logging.INFO)

# id -> display name. "demo_student" is the interactive one the Student tab
# defaults to; the other two are seeded with distinct profiles below so the
# Teacher Dashboard roster shows visibly different students immediately.
DEMO_STUDENTS: dict[str, str] = {
    "demo_student": "Aarav R.",
    "priya_s": "Priya S.",
    "kabir_m": "Kabir M.",
}


def list_students() -> dict[str, str]:
    """Returns the demo student registry (id -> display name)."""
    return dict(DEMO_STUDENTS)


def seed_demo_students() -> None:
    """
    One-time (idempotent — just overwrites) seed for "priya_s" (doing
    well) and "kabir_m" (struggling), so the Teacher Dashboard roster and
    spaced-repetition schedule look like a real classroom immediately.
    "demo_student" is left alone since it's the live interactive one.

    Run this once against a fresh Neo4j instance, e.g.:
        $ python -m backend.memory.students --seed
    """
    from backend.memory.student_graph_store import save_student_profile
    from backend.memory.spaced_repetition import schedule_next_review

    priya_profile = {
        "student_id": "priya_s",
        "concept_mastery": {
            "Fractions": {"mastery": 0.82, "consecutive_misses": 0},
            "Algebra": {"mastery": 0.55, "consecutive_misses": 1},
        },
        "weak_prerequisites": [],
        "recent_attempts": [
            {"concept": "Fractions", "correct": True},
            {"concept": "Fractions", "correct": True},
            {"concept": "Algebra", "correct": True},
        ],
    }
    kabir_profile = {
        "student_id": "kabir_m",
        "concept_mastery": {
            "Fractions": {"mastery": 0.28, "consecutive_misses": 3},
        },
        "weak_prerequisites": ["Common Denominators"],
        "recent_attempts": [
            {"concept": "Fractions", "correct": False},
            {"concept": "Fractions", "correct": False},
            {"concept": "Fractions", "correct": False},
        ],
    }

    save_student_profile("priya_s", priya_profile)
    for concept, data in priya_profile["concept_mastery"].items():
        schedule_next_review("priya_s", concept, data["mastery"])

    save_student_profile("kabir_m", kabir_profile)
    for concept, data in kabir_profile["concept_mastery"].items():
        schedule_next_review("kabir_m", concept, data["mastery"])

    logger.info("Seeded demo students: priya_s, kabir_m.")


# --------------------------------------------------------------------------- #
# Standalone test / one-time seed entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python -m backend.memory.students --seed   # populate the two demo profiles
    $ python -m backend.memory.students          # just print the registry
    """
    import sys

    if "--seed" in sys.argv:
        seed_demo_students()

    print("=" * 80)
    print("Demo student registry:")
    for student_id, name in list_students().items():
        print(f"  {student_id!r} -> {name!r}")
    print("=" * 80)
