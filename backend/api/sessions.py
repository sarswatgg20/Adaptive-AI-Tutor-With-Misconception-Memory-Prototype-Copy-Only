"""
sessions.py
-----------
The one piece of genuinely new state the web API needs. Everything else a
tutoring turn touches (chat history, mastery, quiz scores, escalations,
review schedule) already lives in SQL/Neo4j. What's left is exactly what
app.py used to keep in st.session_state per browser tab:

  - active_question: is the student's next chat message an answer to grade,
    or a fresh question to explain? None means "fresh question."
  - last_tutor_message_at: wall-clock time of the last assistant reply, used
    to compute response_time_seconds for the cheating-risk speed check.
  - last_hint_ladder: the hint-ladder widget payload from the most recent
    turn, so GET /api/state can redraw it after a page refresh without
    replaying the whole conversation.
  - quiz_round: which pre/post-test round is open, per concept.

In-memory, keyed by student_id -- a single-process demo server, same
lifetime tradeoff Streamlit's session state already had (a restart clears
in-progress-question state; persisted data is untouched).
"""

from __future__ import annotations

from typing import Optional


class StudentSession:
    def __init__(self) -> None:
        self.active_question: Optional[str] = None
        self.last_tutor_message_at: Optional[float] = None
        self.last_hint_ladder: dict = {"kind": "none"}
        self.quiz_round: dict[str, int] = {}


_SESSIONS: dict[str, StudentSession] = {}


def get_session(student_id: str) -> StudentSession:
    session = _SESSIONS.get(student_id)
    if session is None:
        session = StudentSession()
        _SESSIONS[student_id] = session
    return session
