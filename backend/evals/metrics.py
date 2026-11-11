"""
metrics.py
----------
Computes the six judging metrics named in the hackathon problem statement —
learning gain, misconception recall, adaptation quality, escalation
precision, memory usefulness, hint quality — purely from data the system
already logs (backend/memory/eval_store.py's interaction_log,
backend/memory/quiz_store.py's quiz_attempts). No LLM calls: every metric
here is deterministic math over structured fields already written by
backend/memory/student_profile.py's run_memory_update, so results are
reproducible and cheap to recompute live during a demo.

Each compute_* function returns {"value": float | None, "sample_size": int,
"label": str} — value is None when sample_size is 0, so callers (the
Teacher Dashboard in app.py) can show "not enough data yet" instead of a
misleading number.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from backend.agents.orchestrator import ESCALATION_MISS_THRESHOLD
from backend.memory.eval_store import get_interactions
from backend.memory.quiz_store import get_all_rounds

# Mirrors the hint ladder's severity ordering (diagnostic_agent.py's
# InstructionalStrategy). "none" (praise/correct turns) ranks lowest since
# it means no intervention was needed at all.
_STRATEGY_SEVERITY = {
    "none": -1,
    "retry_prompt": 0,
    "small_clue": 1,
    "stronger_hint": 2,
    "worked_example": 3,
    "prerequisite_review": 3,
    "escalate_to_teacher": 4,
}


def _metric(value: Optional[float], sample_size: int, label: str) -> dict[str, Any]:
    return {"value": value, "sample_size": sample_size, "label": label}


def learning_gain(student_id: Optional[str] = None) -> dict[str, Any]:
    """Mean(post_score - pre_score) across every completed pre/post-test round."""
    rounds = get_all_rounds(student_id)
    gains = [
        r["post_score"] - r["pre_score"]
        for r in rounds
        if r["pre_score"] is not None and r["post_score"] is not None
    ]
    value = sum(gains) / len(gains) if gains else None
    return _metric(value, len(gains), "Learning gain (pre-test -> post-test)")


def misconception_recall(student_id: Optional[str] = None) -> dict[str, Any]:
    """
    Fraction of repeated misconceptions (same student+concept) that the
    diagnostic agent correctly re-identifies as one already seen, rather
    than treating every recurrence as brand new — the core "misconception
    memory" claim of this project.
    """
    rows = get_interactions(student_id)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    hits = 0
    total = 0
    for row in rows:
        misconception = row.get("identified_misconception")
        if not misconception:
            continue
        key = (row["student_id"], row["concept"])
        normalized = misconception.strip().lower()
        if seen[key]:
            total += 1
            if normalized in seen[key]:
                hits += 1
        seen[key].add(normalized)
    value = hits / total if total else None
    return _metric(value, total, "Misconception recall")


def adaptation_quality(student_id: Optional[str] = None) -> dict[str, Any]:
    """
    Fraction of consecutive same-concept turns, during a building miss
    streak, where the hint ladder's strategy severity did not regress vs.
    the previous turn — does the tutor escalate its intervention as misses
    pile up, rather than getting stuck or backsliding.
    """
    rows = get_interactions(student_id)
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["student_id"], row["concept"])].append(row)

    held_or_improved = 0
    total = 0
    for turns in by_key.values():
        for prev, curr in zip(turns, turns[1:]):
            if curr["consecutive_misses_before"] <= prev["consecutive_misses_before"]:
                continue  # miss streak reset or not building - not an escalation moment
            prev_sev = _STRATEGY_SEVERITY.get(prev.get("applied_strategy"), 0)
            curr_sev = _STRATEGY_SEVERITY.get(curr.get("applied_strategy"), 0)
            total += 1
            if curr_sev >= prev_sev:
                held_or_improved += 1
    value = held_or_improved / total if total else None
    return _metric(value, total, "Adaptation quality (hint ladder escalates with misses)")


def escalation_precision(student_id: Optional[str] = None) -> dict[str, Any]:
    """
    Of every turn that escalated to a teacher, the fraction that were
    actually warranted per the deterministic gate (consecutive_misses at or
    above threshold, distress detected, or cheating risk detected) —
    validates orchestrator.route_after_diagnosis against real logged turns
    instead of just asserting it by construction.
    """
    rows = [r for r in get_interactions(student_id) if r.get("escalated")]
    justified = sum(
        1
        for r in rows
        if r["consecutive_misses_before"] >= ESCALATION_MISS_THRESHOLD
        or r.get("distress_detected")
        or r.get("cheating_risk_detected")
    )
    value = justified / len(rows) if rows else None
    return _metric(value, len(rows), "Escalation precision")


def memory_usefulness(student_id: Optional[str] = None) -> dict[str, Any]:
    """
    Mean mastery gain per turn for turns where the student already had
    history on that concept (consecutive_misses_before > 0, i.e. the memory
    profile had something to work with), reported alongside the same
    average for a student's very first attempt on a concept — shows
    whether the tutor's memory of past struggles correlates with faster
    progress compared to a cold start.
    """
    rows = get_interactions(student_id)
    with_history = [
        r["mastery_after"] - r["mastery_before"] for r in rows if r["consecutive_misses_before"] > 0
    ]
    first_attempt = [
        r["mastery_after"] - r["mastery_before"] for r in rows if r["consecutive_misses_before"] == 0
    ]

    value = sum(with_history) / len(with_history) if with_history else None
    baseline = sum(first_attempt) / len(first_attempt) if first_attempt else None
    metric = _metric(value, len(with_history), "Memory usefulness (mastery gain with prior history)")
    metric["baseline_value"] = baseline
    metric["baseline_sample_size"] = len(first_attempt)
    return metric


def hint_quality(student_id: Optional[str] = None) -> dict[str, Any]:
    """
    Fraction of hint-ladder turns whose chronologically next turn on the
    same student+concept produced a positive mastery signal — did the hint
    actually help the student succeed on their next try.
    """
    rows = get_interactions(student_id)
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["student_id"], row["concept"])].append(row)

    worked = 0
    total = 0
    for turns in by_key.values():
        for curr, nxt in zip(turns, turns[1:]):
            if curr["path"] != "hint":
                continue
            total += 1
            if nxt.get("mastery_signal"):
                worked += 1
    value = worked / total if total else None
    return _metric(value, total, "Hint quality (next attempt succeeds)")


def compute_all_metrics(student_id: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Returns all six metrics keyed by name, ready for the Teacher Dashboard."""
    return {
        "learning_gain": learning_gain(student_id),
        "misconception_recall": misconception_recall(student_id),
        "adaptation_quality": adaptation_quality(student_id),
        "escalation_precision": escalation_precision(student_id),
        "memory_usefulness": memory_usefulness(student_id),
        "hint_quality": hint_quality(student_id),
    }


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python -m backend.evals.metrics

    Reads whatever is already in the local database. Run
    `python -m backend.agents.orchestrator` first (its 5-scenario
    __main__ block exercises praise/hint/escalation paths through
    run_memory_update) to populate real interaction_log rows, or use the
    live Streamlit app.
    """
    import json

    print("=" * 80)
    print(json.dumps(compute_all_metrics(), indent=2))
    print("=" * 80)
