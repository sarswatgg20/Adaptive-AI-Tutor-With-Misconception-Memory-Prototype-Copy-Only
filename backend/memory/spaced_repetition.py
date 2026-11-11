"""
spaced_repetition.py
---------------------
A simple interval-based spaced-repetition scheduler — the technique named
explicitly in the problem statement's flow (step #10) and previously
represented only by a static suggestion sentence on the Teacher Dashboard.

Algorithm: whenever a concept's mastery is (re)computed, schedule its next
review `INTERVAL_DAYS[tier]` days out, where tier is the same
mastered/developing/struggling split used for display coloring elsewhere
in the app (frontend/theme.py's _mastery_tier) — re-declared here rather
than imported, since backend/ shouldn't depend on frontend/.

Backed by the shared SQLAlchemy engine (backend/memory/db.py) — same
database as conversation_store.py, quiz_store.py, and eval_store.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    select,
)

from backend.memory.db import get_engine

logger = logging.getLogger("spaced_repetition")
logging.basicConfig(level=logging.INFO)

_engine = get_engine()
_metadata = MetaData()

# Mirrors frontend/theme.py's _mastery_tier boundaries — same tiers, used
# here for review interval instead of display color.
MASTERED_THRESHOLD = 0.7
DEVELOPING_THRESHOLD = 0.4
INTERVAL_DAYS = {
    "mastered": 7,
    "developing": 3,
    "struggling": 1,
}

review_schedule = Table(
    "review_schedule",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_id", String(128), nullable=False, index=True),
    Column("concept", String(256), nullable=False, index=True),
    Column("last_reviewed_at", DateTime, nullable=False),
    Column("next_review_at", DateTime, nullable=False, index=True),
    Column("mastery_at_schedule", Float, nullable=False),
)


def init_db() -> None:
    """Creates the review_schedule table if it doesn't already exist. Idempotent."""
    _metadata.create_all(_engine)


# Table must exist before any read/write call.
init_db()


def _interval_days(mastery: float) -> int:
    if mastery >= MASTERED_THRESHOLD:
        return INTERVAL_DAYS["mastered"]
    if mastery >= DEVELOPING_THRESHOLD:
        return INTERVAL_DAYS["developing"]
    return INTERVAL_DAYS["struggling"]


def schedule_next_review(student_id: str, concept: str, mastery: float) -> None:
    """
    Upserts this student+concept's review schedule: 7 days out if mastered,
    3 if developing, 1 if still struggling — called right after mastery is
    (re)computed, so the schedule always reflects the latest evidence.
    Failures are logged, not raised — scheduling must never break a
    tutoring turn or quiz submission.
    """
    try:
        now = datetime.now(timezone.utc)
        next_review_at = now + timedelta(days=_interval_days(mastery))
        with _engine.begin() as conn:
            existing = conn.execute(
                select(review_schedule.c.id).where(
                    review_schedule.c.student_id == student_id,
                    review_schedule.c.concept == concept,
                )
            ).first()
            if existing:
                conn.execute(
                    review_schedule.update()
                    .where(review_schedule.c.id == existing.id)
                    .values(
                        last_reviewed_at=now,
                        next_review_at=next_review_at,
                        mastery_at_schedule=mastery,
                    )
                )
            else:
                conn.execute(
                    review_schedule.insert().values(
                        student_id=student_id,
                        concept=concept,
                        last_reviewed_at=now,
                        next_review_at=next_review_at,
                        mastery_at_schedule=mastery,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "schedule_next_review failed for student_id=%r concept=%r: %s",
            student_id,
            concept,
            exc,
        )


def get_due_reviews(student_id: str, as_of: Optional[datetime] = None) -> list[dict[str, Any]]:
    """
    Returns every concept due for review (next_review_at <= as_of, default
    now), most-overdue first. Returns [] on failure rather than raising.
    """
    as_of = as_of or datetime.now(timezone.utc)
    try:
        with _engine.connect() as conn:
            stmt = (
                select(review_schedule)
                .where(
                    review_schedule.c.student_id == student_id,
                    review_schedule.c.next_review_at <= as_of,
                )
                .order_by(review_schedule.c.next_review_at.asc())
            )
            return [dict(row) for row in conn.execute(stmt).mappings().all()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_due_reviews failed for student_id=%r: %s", student_id, exc)
        return []


def get_schedule(student_id: str) -> list[dict[str, Any]]:
    """
    Returns every scheduled concept for a student (due or not), ordered by
    next_review_at ascending — the raw material for the Teacher Dashboard's
    full schedule table. Returns [] on failure rather than raising.
    """
    try:
        with _engine.connect() as conn:
            stmt = (
                select(review_schedule)
                .where(review_schedule.c.student_id == student_id)
                .order_by(review_schedule.c.next_review_at.asc())
            )
            return [dict(row) for row in conn.execute(stmt).mappings().all()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_schedule failed for student_id=%r: %s", student_id, exc)
        return []


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python -m backend.memory.spaced_repetition
    """
    import json

    test_student = "stu_spaced_rep_test"

    print("=" * 80)
    print("Scheduling three concepts at different mastery levels...")
    schedule_next_review(test_student, "Mastered Concept", 0.8)
    schedule_next_review(test_student, "Developing Concept", 0.5)
    schedule_next_review(test_student, "Struggling Concept", 0.2)

    schedule = get_schedule(test_student)
    for row in schedule:
        row["last_reviewed_at"] = str(row["last_reviewed_at"])
        row["next_review_at"] = str(row["next_review_at"])
    print(json.dumps(schedule, indent=2))

    print("\nExpected intervals: Mastered=7d, Developing=3d, Struggling=1d")
    print("None should be due yet (all scheduled in the future):")
    print(json.dumps(get_due_reviews(test_student), indent=2, default=str))

    print("\nBackdating 'Struggling Concept' to be overdue...")
    from datetime import datetime as _dt, timezone as _tz

    with _engine.begin() as conn:
        conn.execute(
            review_schedule.update()
            .where(
                review_schedule.c.student_id == test_student,
                review_schedule.c.concept == "Struggling Concept",
            )
            .values(next_review_at=_dt.now(_tz.utc) - timedelta(days=1))
        )

    due = get_due_reviews(test_student)
    print(f"Due reviews now: {[row['concept'] for row in due]}")
    assert due and due[0]["concept"] == "Struggling Concept", "Backdated concept should be due!"
    print("OK: overdue concept correctly detected.")
    print("=" * 80)
