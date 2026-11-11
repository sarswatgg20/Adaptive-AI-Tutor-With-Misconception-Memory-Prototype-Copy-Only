"""
orchestrator.py
----------------
Tutor Orchestrator — wires the 4 agents + memory update into a single
LangGraph StateGraph, Evaluator-first.

Graph topology
==============

    START
      │
      ▼
  evaluator_node            (scores answer_quality, confidence, distress)
      │
      ▼
  [route_after_evaluation]
      ├── correct + confidence >= 0.6 ──────────────────► tutor_planner_node (praise mode)
      │
      └── everything else ──► diagnostic_node             (classifies error type + prereq)
                                    │
                                    ▼
                              [route_after_diagnosis]      (escalation gate — deterministic,
                                                             reads consecutive_misses + distress)
                                    ├── escalate ──► escalator_node
                                    └── otherwise ──► tutor_planner_node (hint-ladder mode)

  tutor_planner_node / escalator_node ──► memory_update_node ──► END

Agent responsibilities
======================
  evaluator_node      — EvaluatorAgent: scores answer_quality, confidence_score,
                         mastery_signal, distress_detected. Runs FIRST, always.
  diagnostic_node     — DiagnosticAgent: classifies error (Careless/Conceptual/
                         Prerequisite/None) and recommended_strategy. Only runs
                         when the evaluator didn't find a confident-correct answer.
  tutor_planner_node  — TutorPlannerAgent: two modes —
                           * praise mode (no diagnostic_result): reinforcement +
                             required next practice_question.
                           * hint-ladder mode (diagnostic_result present): uses
                             BOTH diagnostic + evaluator results to generate the
                             student-facing hint.
  escalator_node      — EscalationAgent: only runs on the escalation branch;
                         produces teacher_summary + student holding message.
  memory_update_node  — folds the turn's outcome into student_memory_profile
                         (mastery, consecutive_misses, weak_prerequisites,
                         recent_attempts). Runs last, before END.

Routing (both deterministic, no LLM call)
==========================================
  route_after_evaluation — correct + confidence >= CONFIDENCE_THRESHOLD skips
                            diagnosis entirely.
  route_after_diagnosis  — the escalation gate: consecutive_misses on the
                            current concept >= ESCALATION_MISS_THRESHOLD, OR
                            the evaluator's distress_detected flag, sends the
                            turn to the escalator instead of the tutor planner.

Tech stack: Python 3.10+, LangGraph.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.diagnostic_agent import run_diagnostic
from backend.agents.evaluator_agent import run_evaluator
from backend.agents.tutor_planner import run_tutor_planner
from backend.agents.escalation_agent import run_escalation
from backend.memory.student_profile import run_memory_update

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Routing thresholds
# --------------------------------------------------------------------------- #

CONFIDENCE_THRESHOLD = 0.6      # correct + confidence >= this -> skip diagnosis
ESCALATION_MISS_THRESHOLD = 4   # consecutive_misses >= this -> escalate


# --------------------------------------------------------------------------- #
# Shared graph state schema
# --------------------------------------------------------------------------- #

class TutorState(TypedDict, total=False):
    """
    All fields that flow through the tutoring graph for one student turn.
    `total=False` — callers only need to supply the first four; agents fill
    the rest as the graph progresses.
    """

    # --- Provided by caller at turn start ---
    student_id: str                 # Identity key for Neo4j/conversation persistence
    current_question: str           # The problem the student is solving
    current_concept: str            # Topic being taught, e.g. "Probability"
    student_response: str           # Student's latest answer text
    student_memory_profile: dict    # Mastery levels, weak prerequisites, history
    response_time_seconds: Optional[float]  # Wall-clock time app.py measured for this reply

    # --- Written by evaluator_node (runs first) ---
    evaluator_result: dict          # Full EvaluatorResult as dict
    answer_quality: str              # "correct" | "partially_correct" | "incorrect"
    confidence_score: float          # 0.0-1.0
    mastery_signal: bool             # True if positive mastery evidence
    distress_detected: bool          # True if frustration/disengagement language
    cheating_risk_detected: bool     # True if the response looks suspicious (speed/style/formatting)

    # --- Written by diagnostic_node (conditional) ---
    diagnostic_result: dict         # Full DiagnosticResult as dict
    error_type: str                  # "None" | "Careless" | "Conceptual" | "Prerequisite"
    recommended_strategy: str        # Drives the hint ladder

    # --- Written by tutor_planner_node ---
    tutor_plan_result: dict         # Full TutorPlanResult as dict
    hint_text: str                   # Student-facing tutoring response (markdown)
    practice_question: Optional[str]  # Follow-up practice Q, or None
    escalation_flag: bool            # True if this turn needed a human teacher

    # --- Written by escalator_node (conditional) ---
    escalation_result: dict         # Full EscalationResult as dict
    teacher_summary: str              # Alert text for teacher_dashboard.py


# --------------------------------------------------------------------------- #
# Node wrappers
# --------------------------------------------------------------------------- #

def evaluator_node(state: TutorState) -> TutorState:
    """Scores answer quality, confidence, and distress. Always runs first."""
    logger.info(">evaluator_node")
    return run_evaluator(dict(state))  # type: ignore[return-value]


def diagnostic_node(state: TutorState) -> TutorState:
    """Classifies the student's error (only reached when not confident-correct)."""
    logger.info(">diagnostic_node")
    return run_diagnostic(dict(state))  # type: ignore[return-value]


def tutor_planner_node(state: TutorState) -> TutorState:
    """Generates the student-facing hint / praise + next question."""
    logger.info(">tutor_planner_node")
    return run_tutor_planner(dict(state))  # type: ignore[return-value]


def escalator_node(state: TutorState) -> TutorState:
    """Produces teacher alert summary + student holding message."""
    logger.info(">escalator_node")
    return run_escalation(dict(state))  # type: ignore[return-value]


def memory_update_node(state: TutorState) -> TutorState:
    """Folds this turn's outcome into student_memory_profile. Runs last."""
    logger.info(">memory_update_node")
    return run_memory_update(dict(state))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #

def route_after_evaluation(state: TutorState) -> str:
    """
    After evaluator_node: skip diagnosis entirely if the answer was correct
    and the student was confident about it. Otherwise, find out why.

    Exception: a cheating-flagged answer never praise-skips, even if it's
    correct and confident — that combination (too-perfect + suspicious) is
    exactly the highest-risk case. It still has to pass through
    diagnostic_node first (run_escalation requires diagnostic_result in
    state), which is where route_after_diagnosis's escalation gate lives.
    """
    answer_quality = state.get("answer_quality")
    confidence = state.get("confidence_score") or 0.0
    cheating_risk = bool(state.get("cheating_risk_detected"))

    if answer_quality == "correct" and confidence >= CONFIDENCE_THRESHOLD and not cheating_risk:
        logger.info(
            "route -> tutor_planner (praise) [quality=%s, confidence=%.2f]",
            answer_quality,
            confidence,
        )
        return "tutor_planner"

    logger.info(
        "route -> diagnostic_node [quality=%s, confidence=%.2f, cheating_risk=%s]",
        answer_quality,
        confidence,
        cheating_risk,
    )
    return "diagnostic_node"


def _consecutive_misses(state: TutorState) -> int:
    """Reads the current concept's consecutive_misses from the (pre-turn) memory profile."""
    profile = state.get("student_memory_profile") or {}
    concept = state.get("current_concept")
    concept_mastery = profile.get("concept_mastery", {}) or {}
    entry = concept_mastery.get(concept, {}) or {}
    return entry.get("consecutive_misses", 0)


def route_after_diagnosis(state: TutorState) -> str:
    """
    The escalation gate: deterministic, no LLM call. Reads the student's
    consecutive-miss streak on this concept (from the memory profile, before
    this turn's update), the evaluator's distress_detected flag, and the
    evaluator's cheating_risk_detected flag — any one of the three escalates.
    """
    misses = _consecutive_misses(state)
    distress = bool(state.get("distress_detected"))
    cheating_risk = bool(state.get("cheating_risk_detected"))

    if misses >= ESCALATION_MISS_THRESHOLD or distress or cheating_risk:
        logger.info(
            "route -> escalator [consecutive_misses=%d, distress=%s, cheating_risk=%s]",
            misses,
            distress,
            cheating_risk,
        )
        return "escalator"

    logger.info(
        "route -> tutor_planner (hint) [consecutive_misses=%d, distress=%s, cheating_risk=%s]",
        misses,
        distress,
        cheating_risk,
    )
    return "tutor_planner"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_tutor_graph() -> StateGraph:
    """
    Builds (but does not compile) the tutoring StateGraph.
    Exposed separately so tests/evals can inspect the graph before compilation.
    """
    graph = StateGraph(TutorState)

    # Register all nodes
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("diagnostic_node", diagnostic_node)
    graph.add_node("tutor_planner", tutor_planner_node)
    graph.add_node("escalator", escalator_node)
    graph.add_node("memory_update", memory_update_node)

    # Entry point: evaluator always runs first
    graph.add_edge(START, "evaluator")

    # Router #1: correct + confident skips diagnosis
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "tutor_planner": "tutor_planner",
            "diagnostic_node": "diagnostic_node",
        },
    )

    # Router #2: escalation gate
    graph.add_conditional_edges(
        "diagnostic_node",
        route_after_diagnosis,
        {
            "escalator": "escalator",
            "tutor_planner": "tutor_planner",
        },
    )

    # Both terminal branches converge on the memory update, then end
    graph.add_edge("tutor_planner", "memory_update")
    graph.add_edge("escalator", "memory_update")
    graph.add_edge("memory_update", END)

    return graph


def compile_tutor_app():
    """Builds and compiles the graph into an executable LangGraph app."""
    return build_tutor_graph().compile()


# Module-level compiled app — import this elsewhere:
#   from backend.agents.orchestrator import tutor_app
#   result_state = tutor_app.invoke(initial_state)
tutor_app = compile_tutor_app()


# --------------------------------------------------------------------------- #
# End-to-end test block
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    Run all scenarios end-to-end:

        $ ollama serve                        # Ollama must be running
        $ python backend/agents/orchestrator.py

    Requires the models set in .env (e.g. qwen2.5:7b-instruct) to be pulled
    in Ollama.
    """

    base_memory = {
        "student_id": "stu_1024",
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.42, "consecutive_misses": 1},
            "Comparing Unit Fractions": {"mastery": 0.31, "consecutive_misses": 2},
        },
        "weak_prerequisites": ["Comparing Unit Fractions", "Common Denominators"],
        "recent_attempts": [{"concept": "Fraction Addition", "correct": False}],
    }

    escalation_memory = {
        "student_id": "stu_1024",
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.22, "consecutive_misses": 4},
        },
        "weak_prerequisites": ["Comparing Unit Fractions", "Common Denominators"],
        "recent_attempts": [
            {"concept": "Fraction Addition", "correct": False},
            {"concept": "Fraction Addition", "correct": False},
            {"concept": "Fraction Addition", "correct": False},
            {"concept": "Fraction Addition", "correct": False},
        ],
    }

    cheating_test_memory = {
        "student_id": "stu_1024",
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.25, "consecutive_misses": 3},
        },
        "weak_prerequisites": ["Comparing Unit Fractions", "Common Denominators"],
        "recent_attempts": [
            {"concept": "Fraction Addition", "correct": False},
            {"concept": "Fraction Addition", "correct": False},
            {"concept": "Fraction Addition", "correct": False},
        ],
    }

    scenarios = [
        {
            "label": "Scenario 1 — Correct + confident (praise mode, no diagnostic)",
            "state": {
                "student_id": "stu_1024",
                "current_question": "What is 1/2 + 1/3?",
                "current_concept": "Fraction Addition",
                "student_response": "1/2 + 1/3 = 3/6 + 2/6 = 5/6.",
                "student_memory_profile": base_memory,
            },
        },
        {
            "label": "Scenario 2 — Conceptual error (evaluator -> diagnostic -> tutor_planner hint)",
            "state": {
                "student_id": "stu_1024",
                "current_question": "What is 1/2 + 1/3?",
                "current_concept": "Fraction Addition",
                "student_response": "1/2 + 1/3 = 2/5 because you add the tops and bottoms.",
                "student_memory_profile": base_memory,
            },
        },
        {
            "label": "Scenario 3 — Correct but low confidence (evaluator sends to diagnostic; planner should strengthen)",
            "state": {
                "student_id": "stu_1024",
                "current_question": "What is 1/2 + 1/3?",
                "current_concept": "Fraction Addition",
                "student_response": "Maybe 5/6? I think I need a common denominator but I'm not sure.",
                "student_memory_profile": base_memory,
            },
        },
        {
            "label": "Scenario 4 — Careless slip",
            "state": {
                "student_id": "stu_1024",
                "current_question": "What is 1/4 + 1/4?",
                "current_concept": "Fraction Addition",
                "student_response": "1/4 + 1/4 = 2/8.",
                "student_memory_profile": base_memory,
            },
        },
        {
            "label": "Scenario 5 — Persistent failure + distress -> escalation gate fires",
            "state": {
                "student_id": "stu_1024",
                "current_question": "What is 1/2 + 1/3?",
                "current_concept": "Fraction Addition",
                "student_response": "I give up, I hate fractions, this is impossible.",
                "student_memory_profile": escalation_memory,
            },
        },
        {
            "label": "Scenario 6 — Suspiciously fast, polished answer on a concept the student has never gotten right (cheating-risk gate)",
            "state": {
                "student_id": "stu_1024",
                "current_question": "What is 1/2 + 1/3?",
                "current_concept": "Fraction Addition",
                "student_response": (
                    "To add fractions with unlike denominators, we first determine the least "
                    "common multiple of the denominators, which is 6. Converting each fraction "
                    "to an equivalent fraction with denominator 6 yields 3/6 and 2/6. Summing "
                    "the numerators gives 5/6, which is already in lowest terms since gcd(5, 6) = 1."
                ),
                "student_memory_profile": cheating_test_memory,
                "response_time_seconds": 2.5,
            },
        },
    ]

    def _run(label: str, initial_state: dict) -> None:
        print("\n" + "=" * 80)
        print(f"  {label}")
        print(f"  Student: {initial_state['student_response']!r}")
        print("=" * 80)

        final = tutor_app.invoke(initial_state)

        ev = final.get("evaluator_result")
        if ev:
            print(
                f"\n[evaluator]   quality={ev.get('answer_quality')}  "
                f"confidence={ev.get('confidence_score'):.2f}  "
                f"mastery_signal={ev.get('mastery_signal')}  "
                f"distress={ev.get('distress_detected')}  "
                f"cheating_risk={ev.get('cheating_risk_detected')}"
            )
            if ev.get("cheating_risk_detected"):
                print(f"  cheating_risk_reasoning: {ev.get('cheating_risk_reasoning')}")

        if "diagnostic_result" in final:
            print(
                f"[diagnostic]  error_type={final.get('error_type')}  "
                f"strategy={final.get('recommended_strategy')}"
            )
        else:
            print("[diagnostic]  skipped (evaluator: correct + confident)")

        if "escalation_result" in final:
            print("\n[ESCALATION]")
            print(f"  teacher_summary : {final.get('teacher_summary')}")
        else:
            print(f"\n[tutor_planner] strategy={final.get('tutor_plan_result', {}).get('applied_strategy')}")
            if final.get("practice_question"):
                print(f"  practice_question: {final['practice_question']}")

        print(f"\n[hint_text shown to student]\n{final.get('hint_text', '<none>')}")

        print("\n[student_memory_profile after memory_update]")
        print(json.dumps(final.get("student_memory_profile", {}), indent=2))

    for scenario in scenarios:
        try:
            _run(scenario["label"], scenario["state"])
        except Exception as exc:
            print(f"\nERROR in {scenario['label']}: {exc}")

    print("\n" + "=" * 80)
    print("Done.")
