"""
server.py
---------
FastAPI backend for the web dashboards (frontend_web/simple,
frontend_web/gamified), replacing Streamlit as the primary UI. Every
endpoint below is a thin wrapper around existing backend functions -- no new
tutoring/grading/escalation logic lives here, only request/response shaping
and the small bit of per-student session state in sessions.py.

Run: uvicorn backend.api.server:app --reload --port 8000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agents.orchestrator import tutor_app
from backend.agents.quiz_agent import grade_quiz_answer
from backend.agents.tutor_planner import run_explain
from backend.api.auth import infer_role, is_valid_email
from backend.api.sessions import get_session
from backend.evals.metrics import compute_all_metrics
from backend.memory.conversation_store import get_recent_messages, save_message
from backend.memory.eval_store import get_interactions, set_escalation_warranted
from backend.memory.graph_viz import build_prerequisite_graph_json
from backend.memory.quiz_store import (
    get_or_generate_questions,
    get_round_summary,
    next_round_number,
    record_attempt,
)
from backend.memory.spaced_repetition import get_due_reviews, get_schedule, schedule_next_review
from backend.memory.student_graph_store import load_student_profile, save_student_profile
from frontend.theme import strategy_to_ladder_level

app = FastAPI(title="Adaptive AI Tutor API")

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #

class LoginBody(BaseModel):
    email: str


class ChatBody(BaseModel):
    student_id: str
    concept: str
    message: str


class QuizStartBody(BaseModel):
    student_id: str
    concept: str
    variant: str  # "pre" | "post"


class QuizSkipBody(BaseModel):
    student_id: str
    concept: str
    variant: str


class QuizAnswer(BaseModel):
    question_id: int
    question_text: str
    answer_text: str


class QuizSubmitBody(BaseModel):
    student_id: str
    concept: str
    variant: str
    answers: list[QuizAnswer]


class EscalationReviewBody(BaseModel):
    student_id: str
    id: int
    warranted: bool


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _hint_ladder_payload(final_state: dict[str, Any]) -> dict[str, Any]:
    if "escalation_result" in final_state:
        return {"kind": "note", "text": "Escalated to teacher — this has been flagged for follow-up."}
    strategy = (final_state.get("tutor_plan_result") or {}).get("applied_strategy") or final_state.get(
        "recommended_strategy"
    )
    level = strategy_to_ladder_level(strategy)
    if level == 0:
        return {"kind": "none"}
    return {
        "kind": "steps",
        "current_index": level - 1,
        "labels": ["Small clue", "Stronger hint", "Worked example"],
    }


def _round_scores(student_id: str, concept: str, round_number: Optional[int]) -> tuple[Optional[list[int]], Optional[list[int]]]:
    """Returns ([correct, total], [correct, total]) for pre/post of a round, or (None, None) entries not yet attempted."""
    if round_number is None:
        return None, None
    summary = get_round_summary(student_id, concept, round_number)
    pre = None
    if summary["pre_count"]:
        pre = [round(summary["pre_score"] * summary["pre_count"]), summary["pre_count"]]
    post = None
    if summary["post_count"]:
        post = [round(summary["post_score"] * summary["post_count"]), summary["post_count"]]
    return pre, post


_PCT_METRIC_KEYS = {"misconception_recall", "adaptation_quality", "escalation_precision", "hint_quality"}
_DELTA_METRIC_KEYS = {"learning_gain", "memory_usefulness"}

_METRIC_LABELS = {
    "learning_gain": "Learning gain",
    "misconception_recall": "Misconception recall",
    "adaptation_quality": "Adaptation quality",
    "escalation_precision": "Escalation precision",
    "memory_usefulness": "Memory usefulness",
    "hint_quality": "Hint quality",
}
_METRIC_DESCRIPTIONS = {
    "learning_gain": "Average pre-test to post-test score change across completed rounds.",
    "misconception_recall": "How often a recurring misconception is correctly recognized as one already seen.",
    "adaptation_quality": "How often the hint ladder escalates (never regresses) as misses build up.",
    "escalation_precision": "Of turns escalated to a teacher, the fraction that were actually warranted.",
    "memory_usefulness": "Mastery gain on turns with prior history, vs. a first attempt.",
    "hint_quality": "Of hint-ladder turns, the fraction whose very next attempt succeeded.",
}


def _fmt_metric(key: str, metric: dict[str, Any]) -> dict[str, Any]:
    value = metric["value"]
    if key in _PCT_METRIC_KEYS:
        display = f"{value * 100:.0f}%" if value is not None else "—"
        status = "unknown" if value is None else ("good" if value >= 0.7 else "warn" if value >= 0.4 else "critical")
    else:
        display = (f"{'+' if value >= 0 else ''}{value * 100:.0f} pts") if value is not None else "—"
        status = "unknown" if value is None else ("good" if value >= 0.1 else "warn" if value >= 0 else "critical")
    return {
        "label": _METRIC_LABELS[key],
        "display": display,
        "status": status,
        "description": _METRIC_DESCRIPTIONS[key],
    }


def _escalation_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("cheating_risk_detected"):
        trigger_type = "Cheating risk"
    elif row.get("distress_detected"):
        trigger_type = "Distress"
    else:
        trigger_type = "Repeated failure"
    trigger = row.get("teacher_summary") or f"{trigger_type} on {row['concept']}"
    return {
        "id": row["id"],
        "time": row["created_at"].strftime("%H:%M"),
        "topic": row["concept"],
        "trigger": trigger[:200],
        "misses": row.get("consecutive_misses_before", 0),
        "trigger_type": trigger_type,
        "warranted": row.get("warranted"),
    }


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

@app.post("/api/login")
def login(body: LoginBody) -> dict[str, Any]:
    email = body.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return {"student_id": email, "role": infer_role(email)}


# --------------------------------------------------------------------------- #
# Student: state + chat
# --------------------------------------------------------------------------- #

@app.get("/api/state")
def get_state(student_id: str, concept: Optional[str] = None) -> dict[str, Any]:
    session = get_session(student_id)
    history = get_recent_messages(student_id)
    messages = (
        [{"role": m["role"], "content": m["content"]} for m in history]
        if history
        else [{"role": "assistant", "content": "Hi! What would you like to learn today?"}]
    )
    profile = load_student_profile(student_id)
    due_for_review = [row["concept"] for row in get_due_reviews(student_id)]

    pre_test_scores: dict[str, Any] = {}
    post_test_scores: dict[str, Any] = {}
    if concept is not None:
        round_number = session.quiz_round.get(concept)
        pre, post = _round_scores(student_id, concept, round_number)
        if pre is not None:
            pre_test_scores[concept] = pre
        if post is not None:
            post_test_scores[concept] = post

    return {
        "messages": messages,
        "profile": profile,
        "active_question": session.active_question,
        "pre_test_scores": pre_test_scores,
        "post_test_scores": post_test_scores,
        "due_for_review": due_for_review,
        "hint_ladder": session.last_hint_ladder,
    }


@app.post("/api/chat")
def chat(body: ChatBody) -> dict[str, Any]:
    session = get_session(body.student_id)
    response_time_seconds = (
        time.time() - session.last_tutor_message_at if session.last_tutor_message_at is not None else None
    )

    save_message(body.student_id, "user", body.message, concept=body.concept)
    profile = load_student_profile(body.student_id)

    escalated = False
    if session.active_question is None:
        result = run_explain(
            current_concept=body.concept,
            student_question=body.message,
            student_memory_profile=profile,
        )
        reply = result["hint_text"]
        if result.get("practice_question"):
            reply += f"\n\n**Try this:** {result['practice_question']}"
            session.active_question = result["practice_question"]
        hint_ladder: dict[str, Any] = {"kind": "none"}
    else:
        initial_state = {
            "student_id": body.student_id,
            "current_question": session.active_question,
            "current_concept": body.concept,
            "student_response": body.message,
            "student_memory_profile": profile,
            "response_time_seconds": response_time_seconds,
        }
        final_state = tutor_app.invoke(initial_state)

        reply = (
            final_state.get("hint_text")
            or final_state.get("teacher_summary")
            or "I'm not sure how to respond to that — could you rephrase?"
        )
        if final_state.get("practice_question"):
            reply += f"\n\n**Try this:** {final_state['practice_question']}"
            session.active_question = final_state["practice_question"]
        # else: keep active_question as-is so the next reply is graded
        # against the same problem and the hint ladder can escalate.

        updated_profile = final_state.get("student_memory_profile")
        if updated_profile:
            profile = updated_profile
        escalated = "escalation_result" in final_state
        hint_ladder = _hint_ladder_payload(final_state)

    save_message(body.student_id, "assistant", reply, concept=body.concept)
    session.last_tutor_message_at = time.time()
    session.last_hint_ladder = hint_ladder

    return {
        "reply": reply,
        "active_question": session.active_question,
        "hint_ladder": hint_ladder,
        "profile": profile,
        "escalated": escalated,
    }


# --------------------------------------------------------------------------- #
# Student: pre/post-test quizzes (free-text, EvaluatorAgent-graded)
# --------------------------------------------------------------------------- #

@app.post("/api/quiz/start")
def quiz_start(body: QuizStartBody) -> dict[str, Any]:
    session = get_session(body.student_id)
    if body.concept not in session.quiz_round:
        session.quiz_round[body.concept] = next_round_number(body.student_id, body.concept)
    questions = get_or_generate_questions(body.concept, n=5)
    return {
        "questions": [
            {"question_id": q["id"], "question_text": q["question_text"]} for q in questions
        ]
    }


@app.post("/api/quiz/skip")
def quiz_skip(body: QuizSkipBody) -> dict[str, Any]:
    session = get_session(body.student_id)
    if body.concept not in session.quiz_round:
        session.quiz_round[body.concept] = next_round_number(body.student_id, body.concept)
    return {"ok": True}


@app.post("/api/quiz/submit")
def quiz_submit(body: QuizSubmitBody) -> dict[str, Any]:
    session = get_session(body.student_id)
    round_number = session.quiz_round.get(body.concept)
    if round_number is None:
        round_number = next_round_number(body.student_id, body.concept)
        session.quiz_round[body.concept] = round_number

    phase = "pre" if body.variant == "pre" else "post"
    correct_count = 0
    for answer in body.answers:
        is_correct = grade_quiz_answer(
            question_text=answer.question_text,
            current_concept=body.concept,
            student_answer=answer.answer_text,
        )
        record_attempt(
            body.student_id,
            body.concept,
            round_number,
            phase,
            answer.question_id,
            answer.answer_text,
            is_correct,
        )
        if is_correct:
            correct_count += 1

    total = len(body.answers)
    score = correct_count / total if total else 0.0

    if body.variant == "pre":
        profile = load_student_profile(body.student_id)
        profile.setdefault("concept_mastery", {})
        profile["concept_mastery"][body.concept] = {"mastery": score, "consecutive_misses": 0}
        save_student_profile(body.student_id, profile)
        schedule_next_review(body.student_id, body.concept, score)
    else:
        # Post-test is a measurement, not a mastery seed — mirrors the
        # existing Streamlit app, which leaves mastery updates to chat
        # turns (run_memory_update) rather than the quiz view itself.
        session.quiz_round.pop(body.concept, None)

    return {"correct": correct_count, "total": total}


# --------------------------------------------------------------------------- #
# Teacher (and profile/self-view): lookup, metrics, graph, escalations
# --------------------------------------------------------------------------- #

@app.get("/api/profile")
def get_profile(student_id: str) -> dict[str, Any]:
    return {
        "profile": load_student_profile(student_id),
        "due_for_review": [row["concept"] for row in get_due_reviews(student_id)],
    }


@app.get("/api/eval-metrics")
def eval_metrics(student_id: Optional[str] = None) -> dict[str, Any]:
    metrics = compute_all_metrics(student_id)
    order = ["learning_gain", "misconception_recall", "adaptation_quality", "escalation_precision", "memory_usefulness", "hint_quality"]
    return {"metrics": [_fmt_metric(key, metrics[key]) for key in order]}


@app.get("/api/misconception-graph")
def misconception_graph(concept: str, student_id: Optional[str] = None) -> dict[str, Any]:
    profile = load_student_profile(student_id) if student_id else {}
    return build_prerequisite_graph_json(concept, profile)


@app.get("/api/schedule")
def schedule(student_id: str) -> dict[str, Any]:
    return {"schedule": get_schedule(student_id)}


@app.get("/api/escalations")
def escalations(student_id: str) -> dict[str, Any]:
    rows = [r for r in get_interactions(student_id) if r.get("escalated")]
    return {"escalations": [_escalation_row(r) for r in rows]}


@app.post("/api/escalations/review")
def escalations_review(body: EscalationReviewBody) -> dict[str, Any]:
    ok = set_escalation_warranted(body.id, body.warranted)
    if not ok:
        raise HTTPException(status_code=404, detail="Escalation not found.")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Static frontends (mounted last so /api/* routes above take precedence)
# --------------------------------------------------------------------------- #

app.mount("/gamified", StaticFiles(directory=_REPO_ROOT / "frontend_web" / "gamified", html=True), name="gamified")
app.mount("/", StaticFiles(directory=_REPO_ROOT / "frontend_web" / "simple", html=True), name="simple")
