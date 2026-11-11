"""
conversation_store.py
----------------------
Persists the raw chat transcript (student <-> tutor messages) to a SQL
database — conversation history is a simple ordered log, not a graph, so it
lives here rather than in Neo4j.

Backed by SQLAlchemy Core against `DATABASE_URL` (see .env). Defaults to a
local SQLite file (`sqlite:///backend/data/conversations.db`) for zero-setup
local development; pointing `DATABASE_URL` at a
`postgresql+psycopg2://user:pass@host:port/dbname` DSN switches to real
Postgres with no code changes.

Scope note: this module only persists and retrieves history — it does not
feed history back into any agent's prompt. Wiring conversation context into
evaluator/diagnostic/tutor_planner prompts is a deliberate follow-up, not
part of this pass.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    select,
)

from backend.memory.db import DATABASE_URL, get_engine

logger = logging.getLogger("conversation_store")
logging.basicConfig(level=logging.INFO)

_engine = get_engine()
_metadata = MetaData()

conversation_messages = Table(
    "conversation_messages",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_id", String(128), nullable=False, index=True),
    Column("role", String(32), nullable=False),  # "user" | "assistant"
    Column("content", Text, nullable=False),
    Column("concept", String(256), nullable=True),
    Column("created_at", DateTime, nullable=False),
)


def init_db() -> None:
    """Creates the conversation_messages table if it doesn't already exist. Idempotent."""
    _metadata.create_all(_engine)


# Table must exist before any save/query call.
init_db()


def save_message(student_id: str, role: str, content: str, concept: Optional[str] = None) -> None:
    """
    Appends one message to the transcript. Failures are logged, not raised —
    a persistence hiccup should never break the chat UI.
    """
    try:
        with _engine.begin() as conn:
            conn.execute(
                conversation_messages.insert().values(
                    student_id=student_id,
                    role=role,
                    content=content,
                    concept=concept,
                    created_at=datetime.now(timezone.utc),
                )
            )
    except Exception as exc:  # noqa: BLE001 - persistence must not break the chat turn
        logger.warning("save_message failed for student_id=%r: %s", student_id, exc)


def get_recent_messages(student_id: str, limit: int = 50) -> list[dict]:
    """
    Returns up to `limit` most recent messages for a student, oldest-first
    (ready to hydrate a chat UI's message list directly). Returns [] on
    failure rather than raising.
    """
    try:
        with _engine.connect() as conn:
            stmt = (
                select(
                    conversation_messages.c.role,
                    conversation_messages.c.content,
                    conversation_messages.c.concept,
                    conversation_messages.c.created_at,
                )
                .where(conversation_messages.c.student_id == student_id)
                .order_by(conversation_messages.c.created_at.desc())
                .limit(limit)
            )
            rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in reversed(rows)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_recent_messages failed for student_id=%r: %s", student_id, exc)
        return []


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python backend/memory/conversation_store.py

    Fully runnable with no external services — defaults to a local SQLite
    file, so this always works.
    """
    import json

    test_student_id = "stu_test_1024"

    print("=" * 80)
    print(f"DATABASE_URL = {DATABASE_URL}")
    print(f"Saving messages for {test_student_id}...")
    save_message(test_student_id, "user", "Why is 1/2 bigger than 1/3?", concept="Fraction Addition")
    save_message(
        test_student_id,
        "assistant",
        "Great question! Think about splitting a pizza into 2 slices vs 3 slices...",
        concept="Fraction Addition",
    )

    print(f"Reading back recent messages for {test_student_id}...")
    messages = get_recent_messages(test_student_id)
    for m in messages:
        m["created_at"] = str(m["created_at"])
    print(json.dumps(messages, indent=2))
    print("=" * 80)
