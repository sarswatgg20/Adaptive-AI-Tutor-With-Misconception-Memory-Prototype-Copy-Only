"""
student_profile.py
-------------------
Memory Update node for the Adaptive AI Tutor with Misconception Memory.

Role
====
Final node in the tutoring graph, downstream of both terminal branches
(tutor_planner_node and escalator_node). It takes the turn's outcome
(evaluator/diagnostic/escalation results) and folds it into the durable
`student_memory_profile` dict that persists across turns — mastery levels,
consecutive-miss streaks, weak prerequisites, and recent attempt history.

This used to live in app.py as `update_memory_from_turn`, coupling the
memory model to the Streamlit frontend. Moving it into the graph means any
caller of `tutor_app.invoke(...)` gets the same profile-update behavior, not
just the Streamlit UI.

Note: because evaluator_node now always runs first (see orchestrator.py),
`evaluator_result` is always present in state by the time this node runs,
even on the escalation branch.

Once the in-memory dict is updated (unchanged logic from before), this node
also persists it to Neo4j via backend/memory/student_graph_store.py — that
call degrades safely on its own (logs a warning, no-ops) if Neo4j isn't
reachable, so this node's core dict-update responsibility never breaks.

Tech stack: Python 3.10+, plain dict manipulation (no LLM call) + a Neo4j
write at the end.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.memory.student_graph_store import save_student_profile
from backend.memory.eval_store import record_interaction
from backend.memory.spaced_repetition import schedule_next_review

logger = logging.getLogger("student_profile")
logging.basicConfig(level=logging.INFO)

DEFAULT_MASTERY = 0.3
MASTERY_INCREMENT = 0.15
MAX_RECENT_ATTEMPTS = 10


def run_memory_update(state: dict) -> dict:
    """
    LangGraph node function. Runs last, after tutor_planner_node or
    escalator_node, right before END.

    Reads from state:
        - "current_concept" (str): required
        - "student_memory_profile" (dict): optional, defaults to {}
        - "evaluator_result" (dict): required in practice (evaluator always
          runs first), used to determine correctness and mastery evidence
        - "mastery_signal" (bool): positive mastery evidence flag
        - "diagnostic_result" (dict): optional, may contribute a
          missing_prerequisite
        - "escalation_result" (dict): optional, presence just distinguishes
          the escalation branch for logging

    Writes to state:
        - "student_memory_profile" (dict): the updated profile, with
          concept_mastery / weak_prerequisites / recent_attempts refreshed.
    """
    current_concept = state.get("current_concept")
    if not current_concept:
        raise ValueError("run_memory_update requires 'current_concept' in state.")

    profile: dict[str, Any] = state.get("student_memory_profile") or {}
    profile.setdefault("concept_mastery", {})
    profile.setdefault("weak_prerequisites", [])
    profile.setdefault("recent_attempts", [])

    concept_entry = profile["concept_mastery"].setdefault(
        current_concept, {"mastery": DEFAULT_MASTERY, "consecutive_misses": 0}
    )
    mastery_before = concept_entry["mastery"]
    consecutive_misses_before = concept_entry["consecutive_misses"]

    evaluator_result = state.get("evaluator_result") or {}
    diagnostic_result = state.get("diagnostic_result") or {}

    was_correct = evaluator_result.get("answer_quality") == "correct"
    if state.get("mastery_signal"):
        concept_entry["mastery"] = min(1.0, concept_entry["mastery"] + MASTERY_INCREMENT)
        concept_entry["consecutive_misses"] = 0
    else:
        concept_entry["consecutive_misses"] += 1

    missing_prereq = diagnostic_result.get("missing_prerequisite")
    if missing_prereq and missing_prereq not in profile["weak_prerequisites"]:
        profile["weak_prerequisites"].append(missing_prereq)

    profile["recent_attempts"].append({"concept": current_concept, "correct": was_correct})
    profile["recent_attempts"] = profile["recent_attempts"][-MAX_RECENT_ATTEMPTS:]

    logger.info(
        "Memory update for concept=%r -> mastery=%.2f, consecutive_misses=%d, escalated=%s",
        current_concept,
        concept_entry["mastery"],
        concept_entry["consecutive_misses"],
        "escalation_result" in state,
    )

    state["student_memory_profile"] = profile

    student_id = state.get("student_id") or profile.get("student_id")
    if student_id:
        save_student_profile(student_id, profile)
        schedule_next_review(student_id, current_concept, concept_entry["mastery"])

        escalated = "escalation_result" in state
        if escalated:
            path = "escalation"
        elif "diagnostic_result" in state:
            path = "hint"
        else:
            path = "praise"

        tutor_plan_result = state.get("tutor_plan_result") or {}
        applied_strategy = tutor_plan_result.get("applied_strategy") or (
            "escalate_to_teacher" if escalated else diagnostic_result.get("recommended_strategy")
        )

        record_interaction(
            student_id=student_id,
            concept=current_concept,
            path=path,
            mastery_signal=bool(state.get("mastery_signal")),
            distress_detected=bool(state.get("distress_detected")),
            cheating_risk_detected=bool(state.get("cheating_risk_detected")),
            escalated=escalated,
            consecutive_misses_before=consecutive_misses_before,
            mastery_before=mastery_before,
            mastery_after=concept_entry["mastery"],
            answer_quality=evaluator_result.get("answer_quality"),
            confidence_score=evaluator_result.get("confidence_score"),
            error_type=diagnostic_result.get("error_type"),
            identified_misconception=diagnostic_result.get("identified_misconception"),
            applied_strategy=applied_strategy,
            teacher_summary=state.get("teacher_summary"),
        )
    else:
        logger.warning("No student_id in state or profile — skipping Neo4j persistence.")

    return state


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python backend/memory/student_profile.py
    """
    import json

    base_profile = {
        "student_id": "stu_1024",
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.42, "consecutive_misses": 1},
        },
        "weak_prerequisites": [],
        "recent_attempts": [],
    }

    test_cases = [
        {
            "label": "Correct + confident -> mastery up, misses reset",
            "state": {
                "current_concept": "Fraction Addition",
                "student_memory_profile": json.loads(json.dumps(base_profile)),
                "evaluator_result": {"answer_quality": "correct"},
                "mastery_signal": True,
            },
        },
        {
            "label": "Incorrect, prerequisite gap -> misses increment, prereq recorded",
            "state": {
                "current_concept": "Fraction Addition",
                "student_memory_profile": json.loads(json.dumps(base_profile)),
                "evaluator_result": {"answer_quality": "incorrect"},
                "mastery_signal": False,
                "diagnostic_result": {"missing_prerequisite": "Common Denominators"},
            },
        },
        {
            "label": "Escalation branch -> still folds in as a miss",
            "state": {
                "current_concept": "Fraction Addition",
                "student_memory_profile": json.loads(json.dumps(base_profile)),
                "evaluator_result": {"answer_quality": "incorrect"},
                "mastery_signal": False,
                "diagnostic_result": {"missing_prerequisite": None},
                "escalation_result": {"teacher_summary": "..."},
            },
        },
    ]

    for case in test_cases:
        print("=" * 80)
        print(f"TEST: {case['label']}")
        updated = run_memory_update(case["state"])
        print(json.dumps(updated["student_memory_profile"], indent=2))

    print("=" * 80)
