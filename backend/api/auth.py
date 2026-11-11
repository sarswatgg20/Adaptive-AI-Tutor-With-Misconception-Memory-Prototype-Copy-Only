"""
auth.py
-------
Role inference for POST /api/login. There is no password system in this
hackathon build -- student_id is just the email itself, and the domain
decides which dashboard opens. Ported directly from the reference UI's own
mock login logic (same suffix list, same behavior), not a new policy.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TEACHER_DOMAIN_SUFFIXES = (".edu", ".ac.in", ".ac.uk", ".edu.au", ".k12.us")


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


def infer_role(email: str) -> str:
    domain = email.rsplit("@", 1)[-1].lower()
    return "teacher" if any(domain.endswith(s) for s in _TEACHER_DOMAIN_SUFFIXES) else "student"
