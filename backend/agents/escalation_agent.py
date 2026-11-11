"""
escalation_agent.py
--------------------
Escalation Agent for the Adaptive AI Tutor with Misconception Memory.

Role
====
This agent replaces the dummy `escalator_node` in orchestrator.py. It runs
when diagnostic_agent.py's `recommended_strategy` is "escalate_to_teacher"
(typically repeated failures on the same concept and/or distress language in
the student's response). It does two distinct jobs in one call:

    1. Teacher-facing: writes a short, insightful "Teacher Alert Summary"
       for the teacher_dashboard.py view — WHAT the student is stuck on,
       WHY (root misconception / pattern), and how urgent it feels.

    2. Student-facing: writes a warm, honest "holding message" telling the
       student their teacher has been looped in, WITHOUT giving away the
       answer and without sounding clinical or alarming.

This agent always produces both outputs, and `escalation_flag` is always
True by construction (it's only ever invoked once that decision has already
been made upstream by the Diagnostic Agent / orchestrator routing).

This module is designed as a single node inside a LangGraph `StateGraph`,
replacing the placeholder `escalator_node` referenced in orchestrator.py.

Tech stack: Python 3.10+, LangChain, Ollama (local inference via
`langchain_ollama.ChatOllama`), Pydantic v2.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Environment & logging setup
# --------------------------------------------------------------------------- #

load_dotenv()  # Pulls OLLAMA_HOST / model overrides from backend/.env if present

logger = logging.getLogger("escalation_agent")
logging.basicConfig(level=logging.INFO)

# Local inference via Ollama.
DEFAULT_MODEL = os.getenv("ESCALATION_AGENT_MODEL", "gemma4:e4b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("ESCALATION_AGENT_TEMPERATURE", "0.3"))

# Safe fallback text used whenever the LLM leaves a required field blank or
# fails outright — the teacher dashboard and student chat should never show
# an empty string.
_FALLBACK_TEACHER_SUMMARY = (
    "Student is showing repeated difficulty or frustration on this concept. "
    "Automatic summary generation failed — please review the session log directly."
)
_FALLBACK_STUDENT_MESSAGE = (
    "You're working really hard on this, and I want to make sure you get the "
    "best help. I'm looping in your teacher to go over this together. Hang tight!"
)


# --------------------------------------------------------------------------- #
# Pydantic schema
# --------------------------------------------------------------------------- #

class EscalationResult(BaseModel):
    """
    Strict, structured output produced by the Escalation Agent.
    Consumed by both frontend/pages/teacher_dashboard.py (`teacher_summary`)
    and frontend/pages/student_view.py (`student_message`, via `hint_text`
    in the shared graph state).
    """

    teacher_summary: str = Field(
        description=(
            "2-3 sentence summary for the teacher dashboard: the core "
            "misconception/gap, evidence of it (pattern across attempts), "
            "and the student's apparent frustration/confidence level."
        )
    )
    student_message: str = Field(
        description=(
            "Warm, empathetic holding message shown directly to the student. "
            "Must acknowledge their effort, state a teacher is being looped "
            "in, and must NOT reveal the answer to current_question."
        )
    )
    escalation_flag: bool = Field(
        default=True,
        description="Always True — this agent only runs once escalation has already been decided.",
    )

    @field_validator("teacher_summary", "student_message", mode="before")
    @classmethod
    def _fill_blank_strings(cls, v: Any) -> Any:
        """
        Safety net for local LLMs: if Ollama returns None, a missing key
        (coerced to None by the parser), or an empty/whitespace-only string,
        substitute a safe non-empty fallback BEFORE Pydantic's normal
        validation runs, so we never raise a ValidationError over this and
        never show a blank message in the UI.
        """
        if v is None or (isinstance(v, str) and not v.strip()):
            # We can't know here which field we're on from `v` alone, so we
            # return a generic placeholder; the two fields have very
            # different lengths/tones, but an empty string is a worse
            # failure mode than a slightly generic one. The wrapping
            # EscalationAgent.escalate() below overwrites these with the
            # more specific _FALLBACK_* constants after validation anyway
            # if it detects this happened (see `_used_fallback` check).
            return "GENERATION_FAILED_PLACEHOLDER"
        return v

    @field_validator("escalation_flag", mode="before")
    @classmethod
    def _force_true(cls, v: Any) -> bool:
        # This agent is only ever invoked on the escalation path — hard-pin
        # the value regardless of what (if anything) the LLM produced, so a
        # local model can never accidentally suppress an alert.
        return True


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Escalation Agent inside an Adaptive AI Tutor. You are activated \
ONLY when another agent has already decided a human teacher needs to step \
in — because the student has repeatedly missed the same concept, their \
response shows frustration/distress/disengagement, OR the response was \
flagged as suspicious (cheating risk). You do not decide whether to \
escalate; that decision is already final and is given to you as \
`escalation_trigger` below. Your job is to write two short pieces of text.

1. `teacher_summary` — written FOR THE TEACHER, not the student. In 2-3 \
   sentences, explain:
   - The `escalation_trigger` given below — state the REAL reason plainly \
     (e.g., "flagged for a suspiciously fast, polished answer inconsistent \
     with the student's prior responses" for cheating risk; "expressed \
     frustration and disengagement" for distress; "repeated incorrect \
     attempts" for a miss streak). Don't blur trigger types together.
   - What concept/question the student is working on.
   - The core misconception or gap driving the errors, if the trigger is a \
     miss streak or distress (use diagnostic_result's \
     identified_misconception / missing_prerequisite and the pattern in \
     student_memory_profile). If the trigger is cheating risk on an \
     otherwise-correct answer, skip this — there's no error to explain.
   Be concrete and specific — avoid generic phrases like "needs more practice". \
   Write like a colleague briefing another teacher before they step in.

2. `student_message` — written FOR THE STUDENT, shown directly in their chat. \
   It must:
   - For a miss-streak or distress trigger: acknowledge their effort without \
     being condescending, warm and reassuring.
   - For a cheating-risk trigger: stay neutral and matter-of-fact — don't \
     accuse them, but don't praise "effort" on an answer that may not be \
     their own either. Simply say a teacher will review this together.
   - Clearly but gently say their teacher is being looped in to help.
   - NEVER reveal or imply the answer to current_question.
   - Never sound clinical, alarming, or like it's describing them as a \
     "problem" — keep it calm, 1-3 sentences.

Respond ONLY with the structured JSON schema below. No prose outside JSON.

{format_instructions}
"""

_HUMAN_PROMPT = """\
Escalation trigger (the real reason this turn was escalated — reflect this \
accurately in teacher_summary): {escalation_trigger}

Current question the student is working on:
\"\"\"{current_question}\"\"\"

Student's most recent response:
\"\"\"{student_response}\"\"\"

Diagnostic result (from the Diagnostic Agent):
{diagnostic_result}

Student memory profile (mastery, consecutive misses, weak prerequisites):
{student_memory_profile}

Write the teacher_summary and student_message now.
"""


def _build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    """Builds the chat prompt template with format instructions baked in."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ]
    ).partial(format_instructions=parser.get_format_instructions())


# --------------------------------------------------------------------------- #
# Core agent logic
# --------------------------------------------------------------------------- #

class EscalationAgent:
    """
    Thin wrapper around a local Ollama LLM + prompt + parser that produces
    one teacher summary + student holding message per escalation event.
    Instantiate once (e.g., in orchestrator.py) and reuse across turns.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.parser = PydanticOutputParser(pydantic_object=EscalationResult)
        self.prompt = _build_prompt(self.parser)

        # format="json" constrains local Ollama generation to valid JSON,
        # which we then validate against EscalationResult via the parser —
        # avoids relying on tool-calling support that varies across local
        # models.
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json",
        )
        self.chain = self.prompt | self.llm | self.parser

    def escalate(
        self,
        current_question: str,
        student_response: str,
        diagnostic_result: dict[str, Any],
        student_memory_profile: dict[str, Any],
        escalation_trigger: str = "repeated incorrect attempts",
    ) -> EscalationResult:
        """
        Run escalation summary generation for a single student turn.

        Args:
            current_question: The problem the student was working on.
            student_response: Raw text of the student's latest answer.
            diagnostic_result: Output dict from diagnostic_agent.run_diagnostic.
            student_memory_profile: Dict of student mastery / history.
            escalation_trigger: Plain-language reason this turn escalated
                (cheating risk / distress / repeated misses) — computed
                deterministically by run_escalation below, not guessed by
                the LLM, so teacher_summary names the real cause.

        Returns:
            A validated EscalationResult. escalation_flag is always True.
            Falls back to safe canned text if the LLM call or parsing fails,
            so a teacher alert and student message are ALWAYS produced —
            this path must never silently fail.
        """
        try:
            result: EscalationResult = self.chain.invoke(
                {
                    "current_question": current_question,
                    "student_response": student_response,
                    "diagnostic_result": json.dumps(diagnostic_result, indent=2),
                    "student_memory_profile": json.dumps(
                        student_memory_profile, indent=2
                    ),
                    "escalation_trigger": escalation_trigger,
                }
            )
        except Exception as exc:  # noqa: BLE001 - escalation must never crash the graph
            logger.exception(
                "Escalation agent failed to produce structured output: %s", exc
            )
            return EscalationResult(
                teacher_summary=(
                    f"{_FALLBACK_TEACHER_SUMMARY} (concept: "
                    f"{diagnostic_result.get('identified_misconception') or 'unknown'})"
                ),
                student_message=_FALLBACK_STUDENT_MESSAGE,
                escalation_flag=True,
            )

        # Swap out the generic before-validator placeholder for our more
        # specific, context-aware fallback text if the LLM left a field
        # blank/None but generation otherwise succeeded.
        if result.teacher_summary == "GENERATION_FAILED_PLACEHOLDER":
            logger.warning("LLM returned an empty teacher_summary; using fallback text.")
            result.teacher_summary = _FALLBACK_TEACHER_SUMMARY
        if result.student_message == "GENERATION_FAILED_PLACEHOLDER":
            logger.warning("LLM returned an empty student_message; using fallback text.")
            result.student_message = _FALLBACK_STUDENT_MESSAGE

        return result


# Module-level singleton so LangGraph nodes don't re-instantiate the Ollama
# client on every invocation. orchestrator.py can still create its own
# EscalationAgent(...) instance if it needs a different model/temperature.
_default_agent: Optional[EscalationAgent] = None


def _get_default_agent() -> EscalationAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = EscalationAgent()
    return _default_agent


# --------------------------------------------------------------------------- #
# LangGraph node entry point
# --------------------------------------------------------------------------- #

def run_escalation(state: dict) -> dict:
    """
    LangGraph node function. Replaces the dummy `escalator_node` referenced
    in orchestrator.py. Runs when `route_after_diagnosis` sends the graph
    down the "escalate_to_teacher" branch.

    Expected keys read from `state`:
        - "current_question" (str): required
        - "student_response" (str): required
        - "diagnostic_result" (dict): required, produced by run_diagnostic
        - "student_memory_profile" (dict): optional, defaults to {}
        - "cheating_risk_detected" / "distress_detected" (bool): optional,
          from run_evaluator — used to name the real escalation trigger
        - "current_concept" (str): optional, used to look up consecutive
          misses for the trigger description

    Keys written back into `state`:
        - "escalation_result" (dict): the full EscalationResult, JSON-serializable,
          for teacher_dashboard.py to render `teacher_summary` from.
        - "teacher_summary" (str): convenience top-level copy.
        - "hint_text" (str): set to `student_message` — IMPORTANT: this keeps
          the key the frontend/student_view.py already reads from consistent
          across both the hint-generator path and the escalation path, so
          the chat UI doesn't need branch-specific rendering logic.
        - "escalation_flag" (bool): always True.

    Returns:
        The updated state dict (LangGraph merges this back into the graph's
        shared state).
    """
    current_question = state.get("current_question")
    student_response = state.get("student_response")
    diagnostic_result = state.get("diagnostic_result")
    student_memory_profile = state.get("student_memory_profile", {}) or {}

    if not current_question or not student_response or not diagnostic_result:
        raise ValueError(
            "run_escalation requires 'current_question', 'student_response', "
            "and 'diagnostic_result' to be present in state."
        )

    if state.get("cheating_risk_detected"):
        escalation_trigger = "cheating risk — suspicious answer pattern (speed/style/formatting)"
    elif state.get("distress_detected"):
        escalation_trigger = "student distress or disengagement language"
    else:
        current_concept = state.get("current_concept")
        misses = (
            student_memory_profile.get("concept_mastery", {})
            .get(current_concept, {})
            .get("consecutive_misses", 0)
        )
        escalation_trigger = f"repeated incorrect attempts ({misses} consecutive misses)"

    agent = _get_default_agent()
    result = agent.escalate(
        current_question=current_question,
        student_response=student_response,
        diagnostic_result=diagnostic_result,
        student_memory_profile=student_memory_profile,
        escalation_trigger=escalation_trigger,
    )

    logger.warning(
        "ESCALATION TRIGGERED | teacher_summary=%r",
        result.teacher_summary,
    )

    state["escalation_result"] = result.model_dump()
    state["teacher_summary"] = result.teacher_summary
    # Deliberately reuse "hint_text" (rather than a new key) so the student
    # chat UI has one consistent field to render regardless of which branch
    # of the graph produced it.
    state["hint_text"] = result.student_message
    state["escalation_flag"] = True
    return state


# --------------------------------------------------------------------------- #
# Mock execution block for standalone testing
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    Run this file directly to sanity-check the agent before wiring it into
    orchestrator.py in place of the dummy escalator_node:

        $ ollama pull gemma4:12b        # one-time, if not already pulled
        $ ollama serve                  # if not already running
        $ python backend/agents/escalation_agent.py
    """

    mock_memory_profile = {
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

    mock_diagnostic_result = {
        "is_correct": False,
        "error_type": "Conceptual",
        "identified_misconception": (
            "Adds numerators and denominators directly without finding a "
            "common denominator; has not internalized the rule after four "
            "consecutive attempts."
        ),
        "missing_prerequisite": None,
        "confidence": 0.65,
        "reasoning": (
            "consecutive_misses=4 on Fraction Addition and the student's "
            "latest response shows explicit disengagement/distress language."
        ),
        "recommended_strategy": "escalate_to_teacher",
    }

    fake_state = {
        "current_question": "What is 1/2 + 1/3?",
        "student_response": "I give up, this makes no sense.",
        "diagnostic_result": mock_diagnostic_result,
        "student_memory_profile": mock_memory_profile,
    }

    print("=" * 80)
    print("TEST CASE: Persistent failure + distress -> escalation")
    print(f"Student response: {fake_state['student_response']!r}")

    try:
        updated_state = run_escalation(fake_state)
        print("\n--- Full escalation_result ---")
        print(json.dumps(updated_state["escalation_result"], indent=2))
        print("\n--- state['hint_text'] (shown to student) ---")
        print(updated_state["hint_text"])
        print("\n--- state['teacher_summary'] (shown on teacher dashboard) ---")
        print(updated_state["teacher_summary"])
        print("\n--- state['escalation_flag'] ---")
        print(updated_state["escalation_flag"])
    except Exception as e:  # noqa: BLE001
        print(f"ERROR running escalation node: {e}")

    print("=" * 80)
    print("Done. If Ollama was not running / model not pulled, the call above")
    print("should have fallen back to the safe canned text rather than crashing.")