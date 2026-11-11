"""
db.py
-----
Shared SQLAlchemy engine for every SQL-backed store in backend/memory/
(conversation_store.py, quiz_store.py, ...).

Extracted from conversation_store.py so all SQL tables share one engine
(and therefore one connection pool / one SQLite file) against `DATABASE_URL`
rather than each module opening its own. Defaults to a local SQLite file for
zero-setup local development; pointing `DATABASE_URL` at a
`postgresql+psycopg2://user:pass@host:port/dbname` DSN switches every store
to real Postgres with no code changes.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///backend/data/conversations.db")

# SQLite needs its parent directory to exist before the file can be created.
if DATABASE_URL.startswith("sqlite:///"):
    _sqlite_path = DATABASE_URL.replace("sqlite:///", "", 1)
    _parent_dir = os.path.dirname(_sqlite_path)
    if _parent_dir:
        os.makedirs(_parent_dir, exist_ok=True)

_engine: Engine = create_engine(DATABASE_URL, future=True)


def get_engine() -> Engine:
    """Returns the shared engine every SQL-backed memory module should use."""
    return _engine
