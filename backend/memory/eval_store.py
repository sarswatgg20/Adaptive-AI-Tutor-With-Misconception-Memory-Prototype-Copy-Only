"""
eval_store.py
--------------
Persists one row per tutoring turn — the structured detail (strategy
applied, misconception identified, escalation firing, mastery before/after)
that `backend/evals/metrics.py` needs to compute the problem statement's six
judging metrics. Before this, that detail only ever lived in Streamlit
session state (`last_final_state`, `escalation_log` in app.py) and was lost
on every page reload — metrics couldn't be computed across sessions.

Written from backend/memory/student_profile.py's run_memory_update, the one
place in the graph that already sees the full turn outcome, right next to
its existing save_student_profile(...) call.

Backed by the shared SQLAlchemy engine (backend/memory/db.py) — same
database as conversation_store.py and quiz_store.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    inspect,
    select,
    text,
)

from backend.memory.db import get_engine

logger = logging.getLogger("eval_store")
logging.basicConfig(level=logging.INFO)

_engine = get_engine()
_metadata = MetaData()

interaction_log = Table(
    "interaction_log",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_id", String(128), nullable=False, index=True),
    Column("concept", String(256), nullable=False, index=True),
    Column("created_at", DateTime, nullable=False),
    Column("path", String(16), nullable=False),  # "praise" | "hint" | "escalation"
    Column("answer_quality", String(32), nullable=True),
    Column("confidence_score", Float, nullable=True),
    Column("mastery_signal", Boolean, nullable=False),
    Column("distress_detected", Boolean, nullable=False),
    Column("cheating_risk_detected", Boolean, nullable=False),
    Column("error_type", String(32), nullable=True),
    Column("identified_misconception", Text, nullable=True),
    Column("applied_strategy", String(32), nullable=True),
    Column("escalated", Boolean, nullable=False),
    Column("consecutive_misses_before", Integer, nullable=False),
    Column("mastery_before", Float, nullable=False),
    Column("mastery_after", Float, nullable=False),
    Column("teacher_summary", Text, nullable=True),
    Column("warranted", Boolean, nullable=True),
)

# Columns added after this table's original creation — each one needs an
# ad-hoc ALTER TABLE for any database created before it existed, since
# create_all() only creates missing tables, it doesn't alter existing
# ones. No formal migration tool in this project; this is intentionally a
# narrow, ordered list rather than a general migration framework.
_COLUMN_MIGRATIONS = [
    ("cheating_risk_detected", "BOOLEAN NOT NULL DEFAULT 0"),
    ("teacher_summary", "TEXT"),
    ("warranted", "BOOLEAN"),
]


def init_db() -> None:
    """Creates the interaction_log table if it doesn't already exist, and
    applies any pending column migrations from _COLUMN_MIGRATIONS. Idempotent."""
    _metadata.create_all(_engine)

    inspector = inspect(_engine)
    if "interaction_log" in inspector.get_table_names():
        existing_columns = {c["name"] for c in inspector.get_columns("interaction_log")}
        for column_name, ddl_type in _COLUMN_MIGRATIONS:
            if column_name not in existing_columns:
                with _engine.begin() as conn:
                    conn.execute(
                        text(f"ALTER TABLE interaction_log ADD COLUMN {column_name} {ddl_type}")
                    )
                logger.info("Migrated interaction_log: added %s column.", column_name)


# Table must exist before any save/query call.
init_db()


def record_interaction(
    student_id: str,
    concept: str,
    path: str,
    mastery_signal: bool,
    distress_detected: bool,
    cheating_risk_detected: bool,
    escalated: bool,
    consecutive_misses_before: int,
    mastery_before: float,
    mastery_after: float,
    answer_quality: Optional[str] = None,
    confidence_score: Optional[float] = None,
    error_type: Optional[str] = None,
    identified_misconception: Optional[str] = None,
    applied_strategy: Optional[str] = None,
    teacher_summary: Optional[str] = None,
) -> None:
    """Records one tutoring turn. Failures are logged, not raised — this
    must never break the tutoring graph."""
    try:
        with _engine.begin() as conn:
            conn.execute(
                interaction_log.insert().values(
                    student_id=student_id,
                    concept=concept,
                    created_at=datetime.now(timezone.utc),
                    path=path,
                    answer_quality=answer_quality,
                    confidence_score=confidence_score,
                    mastery_signal=mastery_signal,
                    distress_detected=distress_detected,
                    cheating_risk_detected=cheating_risk_detected,
                    error_type=error_type,
                    identified_misconception=identified_misconception,
                    applied_strategy=applied_strategy,
                    escalated=escalated,
                    consecutive_misses_before=consecutive_misses_before,
                    mastery_before=mastery_before,
                    mastery_after=mastery_after,
                    teacher_summary=teacher_summary,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_interaction failed for student_id=%r: %s", student_id, exc)


def get_interactions(
    student_id: Optional[str] = None, concept: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Returns logged turns ordered oldest-first. student_id=None returns every
    student's rows (used later for multi-student/class-wide metrics).
    Returns [] on failure rather than raising.
    """
    try:
        with _engine.connect() as conn:
            stmt = select(interaction_log)
            if student_id is not None:
                stmt = stmt.where(interaction_log.c.student_id == student_id)
            if concept is not None:
                stmt = stmt.where(interaction_log.c.concept == concept)
            stmt = stmt.order_by(interaction_log.c.created_at.asc())
            return [dict(row) for row in conn.execute(stmt).mappings().all()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_interactions failed for student_id=%r: %s", student_id, exc)
        return []


def set_escalation_warranted(interaction_id: int, warranted: bool) -> bool:
    """
    Records a teacher's real verdict on one escalated turn (the web UI's
    Warranted/Not-warranted buttons). Returns whether a row was updated.
    """
    try:
        with _engine.begin() as conn:
            result = conn.execute(
                interaction_log.update()
                .where(interaction_log.c.id == interaction_id)
                .values(warranted=warranted)
            )
            return result.rowcount > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("set_escalation_warranted failed for id=%r: %s", interaction_id, exc)
        return False


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python -m backend.memory.eval_store
    """
    import json

    test_student = "stu_eval_test"

    print("=" * 80)
    print("Recording a mock hint-ladder turn followed by a mastery win...")
    record_interaction(
        student_id=test_student,
        concept="Fractions",
        path="hint",
        mastery_signal=False,
        distress_detected=False,
        cheating_risk_detected=False,
        escalated=False,
        consecutive_misses_before=1,
        mastery_before=0.3,
        mastery_after=0.3,
        answer_quality="incorrect",
        confidence_score=0.7,
        error_type="Conceptual",
        identified_misconception="Believes a larger denominator means a larger fraction.",
        applied_strategy="small_clue",
    )
    record_interaction(
        student_id=test_student,
        concept="Fractions",
        path="praise",
        mastery_signal=True,
        distress_detected=False,
        cheating_risk_detected=False,
        escalated=False,
        consecutive_misses_before=2,
        mastery_before=0.3,
        mastery_after=0.45,
        answer_quality="correct",
        confidence_score=0.9,
        applied_strategy="none",
    )

    rows = get_interactions(test_student)
    for r in rows:
        r["created_at"] = str(r["created_at"])
    print(json.dumps(rows, indent=2))
    print("=" * 80)
