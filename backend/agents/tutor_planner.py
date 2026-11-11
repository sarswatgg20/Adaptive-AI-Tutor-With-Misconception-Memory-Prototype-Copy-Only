"""
tutor_planner.py
-----------------
Tutor Planner Agent for the Adaptive AI Tutor with Misconception Memory.

Role
====
This agent has three distinct entry points:

  1. PRAISE MODE (plan_praise) — reached directly from route_after_evaluation
     when the Evaluator scored the response correct + confident (>= 0.6). No
     Diagnostic Agent runs on this path, so there's no diagnostic_result to
     read. This mode writes a short positive reinforcement AND a required
     next practice_question, so the student keeps progressing.

  2. HINT-LADDER MODE (plan) — reached after the Diagnostic Agent, using
     BOTH its error classification and the Evaluator's quality/confidence
     assessment to calibrate a hint response.

  3. EXPLAIN MODE (plan_explain) — called directly by app.py (NOT part of
     the orchestrator.py graph, no Evaluator/Diagnostic involved) when the
     student has asked an open conceptual question rather than answered a
     posed problem. There's nothing to grade yet, so this mode just
     explains clearly and ends with one comprehension-check question,
     which app.py then tracks as the next "active_question" to evaluate an
     answer against.

run_tutor_planner (the LangGraph node) branches between modes 1 and 2 based
on whether "diagnostic_result" is present in state. Mode 3 is invoked via
the separate run_explain(...) module function, outside the graph entirely.

Hard pedagogical rule: NEVER reveal the direct final answer to current_question.

Hint Ladder (driven by diagnostic_result.recommended_strategy, calibrated by
evaluator confidence):
    - none               -> correct + confident: short positive reinforcement
                            (praise mode also uses this strategy value, plus
                            a required next practice_question)
    - retry_prompt       -> careless slip: gentle "double-check" nudge
    - small_clue         -> conceptual, first/second miss: guiding question
    - stronger_hint      -> conceptual, repeated miss OR low-confidence correct
    - worked_example     -> persistent struggle: full worked similar example
    - prerequisite_review-> missing foundational skill: reteach prerequisite
    - escalate_to_teacher-> handled without LLM call (orchestrator's escalation
                            gate routes to escalator_node before this agent runs)

Tech stack: Python 3.10+, LangChain, Ollama (ChatOllama), Pydantic v2.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from backend.agents.diagnostic_agent import ErrorType, InstructionalStrategy

# --------------------------------------------------------------------------- #
# Environment & logging
# --------------------------------------------------------------------------- #

load_dotenv()

logger = logging.getLogger("tutor_planner")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("TUTOR_PLANNER_MODEL", "gemma4:e4b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("TUTOR_PLANNER_TEMPERATURE", "0.4"))


# --------------------------------------------------------------------------- #
# Pydantic schema
# --------------------------------------------------------------------------- #

class TutorPlanResult(BaseModel):
    """
    Structured output from the Tutor Planner Agent.
    Written into the shared LangGraph state and rendered in student_view.py.
    """

    hint_text: str = Field(
        description=(
            "Markdown-formatted text shown to the student. Must NEVER state the "
            "final numeric/symbolic answer to current_question."
        )
    )
    applied_strategy: InstructionalStrategy = Field(
        description="The hint-ladder strategy actually used to produce hint_text."
    )
    practice_question: Optional[str] = Field(
        default=None,
        description=(
            "Optional follow-up practice question for the student, generated "
            "when the strategy is 'prerequisite_review' or 'worked_example'. "
            "Must be simpler than or at the same level as current_question. "
            "Null for other strategies."
        ),
    )
    escalation_flag: bool = Field(
        default=False,
        description=(
            "True if the planner could not produce a safe hint without revealing "
            "the answer, or detects a situation requiring human intervention."
        ),
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short explanation if escalation_flag is True. Null otherwise.",
    )

    @field_validator("applied_strategy", mode="before")
    @classmethod
    def _ensure_valid_strategy(cls, v: Any) -> Any:
        valid = [s.value for s in InstructionalStrategy]
        if not v or v not in valid:
            logger.warning("Planner output invalid strategy %r; defaulting to 'small_clue'.", v)
            return InstructionalStrategy.SMALL_CLUE.value
        return v

    @field_validator("escalation_reason", "practice_question", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Tutor Planner Agent inside an Adaptive AI Tutor. You receive the \
output of a Diagnostic Agent (which classified what is wrong) and an Evaluator \
Agent (which scored answer quality and student confidence). Your job is to turn \
both into a single, well-crafted student-facing tutoring response.

ABSOLUTE RULE — NEVER BROKEN:
You must NEVER reveal, state, or imply the final answer to `current_question`. \
If the requested strategy would force you to reveal it, set escalation_flag=true \
and produce the safest possible non-answer hint_text.

Use `recommended_strategy` from the diagnostic result as your primary guide, \
but CALIBRATE using the evaluator's `confidence_score`:
- If the student answered correctly but confidence_score < 0.6 → use \
  "stronger_hint" even if the diagnostic says "none", to reinforce the shaky \
  understanding before moving on. Override applied_strategy accordingly.
- If the student has low confidence AND a wrong answer → go at most one step \
  heavier than the recommended strategy (e.g., small_clue → stronger_hint).
- High confidence wrong answers → follow the diagnostic strategy faithfully; \
  the student is committed to a flawed model and needs direct confrontation.

Hint Ladder definitions:

- "none": Correct and confident (confidence_score >= 0.6). Write a short, \
  warm, specific positive reinforcement (1-2 sentences). No new hint.

- "retry_prompt": Careless error. Brief, encouraging nudge to re-check — \
  point at the type of slip without doing the arithmetic for them.

- "small_clue": Ask a guiding question or point at the specific part of \
  current_question the student should re-examine. 1-2 sentences. Don't \
  name the rule outright.

- "stronger_hint": Name the rule/concept the student must apply, and/or give \
  a partial setup. Still don't carry it through to the final answer.

- "worked_example": Invent a DIFFERENT but structurally similar problem, solve \
  it fully step-by-step in markdown, then ask the student to apply the same \
  steps to their question. Never solve current_question itself. \
  Also set practice_question to a new follow-up problem at the same level.

- "prerequisite_review": The gap is in a foundational skill. Briefly reteach \
  that concept with a small separate example, then connect back to why it \
  matters for current_question. Set practice_question to a simpler prerequisite \
  exercise.

Also use `student_memory_profile` for tone only (more misses → gentler, more \
patient tone). Never mention the profile explicitly to the student.

Formatting: hint_text should be concise markdown, friendly tone, ≤120 words \
(except "worked_example", which may be longer due to the step-by-step solution).

Respond ONLY with the structured JSON schema. No prose outside JSON.

CRITICAL: Return raw JSON only — no markdown code blocks, no preamble.

{format_instructions}
"""

_HUMAN_PROMPT = """\
Current concept being taught: {current_concept}

Current question the student is solving:
\"\"\"{current_question}\"\"\"

Student's most recent response:
\"\"\"{student_response}\"\"\"

Diagnostic Agent result:
{diagnostic_result}

Evaluator Agent result:
{evaluator_result}

Student memory profile (tone calibration only):
{student_memory_profile}

recommended_strategy from diagnosis: {recommended_strategy}
evaluator answer_quality: {answer_quality}
evaluator confidence_score: {confidence_score}
{curriculum_context_block}
Generate the tutoring response now.
"""


def _build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())


# --------------------------------------------------------------------------- #
# Praise-mode prompt template (correct + confident path, no diagnostic_result)
# --------------------------------------------------------------------------- #

_PRAISE_SYSTEM_PROMPT = """\
You are the Tutor Planner Agent inside an Adaptive AI Tutor. The student just \
answered CORRECTLY and CONFIDENTLY (per the Evaluator Agent) — no diagnostic \
classification was needed. Your job here is simple: affirm the student and \
keep them moving forward.

Produce:

- `hint_text`: 1-2 warm, specific sentences of positive reinforcement. Call \
  out what they did right (e.g., the specific step or reasoning), not just \
  generic praise like "Good job!".
- `applied_strategy`: always "none".
- `practice_question`: REQUIRED (never null). A new question on the SAME \
  concept, at the same difficulty or one small step harder than \
  current_question, so the student keeps progressing. Must be a complete, \
  answerable question, not a hint about one.
- `escalation_flag`: always false.

Use `student_memory_profile` for tone only (never mention it explicitly).

Respond ONLY with the structured JSON schema. No prose outside JSON.

CRITICAL: Return raw JSON only — no markdown code blocks, no preamble.

{format_instructions}
"""

_PRAISE_HUMAN_PROMPT = """\
Current concept being taught: {current_concept}

Current question the student just solved:
\"\"\"{current_question}\"\"\"

Student's response:
\"\"\"{student_response}\"\"\"

Evaluator Agent result:
{evaluator_result}

Student memory profile (tone calibration only):
{student_memory_profile}

Generate the praise + next practice question now.
"""


def _build_praise_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", _PRAISE_SYSTEM_PROMPT), ("human", _PRAISE_HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())


# --------------------------------------------------------------------------- #
# Explain-mode prompt template (open question, no posed problem to grade)
# --------------------------------------------------------------------------- #

_EXPLAIN_SYSTEM_PROMPT = """\
You are the Tutor Planner Agent inside an Adaptive AI Tutor. The student has \
asked an open conceptual question — they are NOT answering a problem you \
posed, so there is nothing to grade here. No Evaluator or Diagnostic Agent \
has run on this turn.

Produce:

- `hint_text`: A clear, simple answer to the student's question, using a \
  concrete example (e.g. splitting objects into pieces) to make the idea \
  intuitive. Unlike hint-ladder mode, you SHOULD explain the concept here — \
  that's the point of this mode. Keep it warm and at most a few sentences; \
  don't over-explain.
- `applied_strategy`: always "none".
- `practice_question`: REQUIRED (never null). One comprehension-check \
  question that tests whether the student now grasps the idea you just \
  explained — a small, concrete, answerable question, not a restatement of \
  their original question. This becomes the next problem they work on.
- `escalation_flag`: always false.

Use `student_memory_profile` for tone only (never mention it explicitly).

Respond ONLY with the structured JSON schema. No prose outside JSON.

CRITICAL: Return raw JSON only — no markdown code blocks, no preamble.

{format_instructions}
"""

_EXPLAIN_HUMAN_PROMPT = """\
Current concept being taught: {current_concept}

Student's question:
\"\"\"{student_question}\"\"\"

Student memory profile (tone calibration only):
{student_memory_profile}

Generate the explanation + comprehension-check question now.
"""


def _build_explain_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", _EXPLAIN_SYSTEM_PROMPT), ("human", _EXPLAIN_HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())


# --------------------------------------------------------------------------- #
# Answer-leak guard
# --------------------------------------------------------------------------- #

def _looks_like_answer_leak(hint_text: str, current_question: str) -> bool:
    """
    Best-effort heuristic: flag hints containing a bare fraction/decimal result
    pattern not present in the original question (likely a disclosed final answer).
    """
    leak_pattern = re.compile(r"=\s*\d+\s*/\s*\d+|=\s*\d+\.\d+")
    return bool(leak_pattern.search(hint_text))


# --------------------------------------------------------------------------- #
# Core agent logic
# --------------------------------------------------------------------------- #

class TutorPlannerAgent:
    """
    Thin wrapper around Ollama LLM + prompt + parser.
    Instantiate once and reuse across turns.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.parser = PydanticOutputParser(pydantic_object=TutorPlanResult)
        self.prompt = _build_prompt(self.parser)
        self.praise_prompt = _build_praise_prompt(self.parser)
        self.explain_prompt = _build_explain_prompt(self.parser)
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json",
        )
        self.chain = self.prompt | self.llm | self.parser
        self.praise_chain = self.praise_prompt | self.llm | self.parser
        self.explain_chain = self.explain_prompt | self.llm | self.parser

    def plan_explain(
        self,
        student_question: str,
        current_concept: str,
        student_memory_profile: dict[str, Any],
    ) -> TutorPlanResult:
        """
        Explain-mode planning: the student asked an open conceptual question,
        not an answer to a posed problem. No Evaluator/Diagnostic input is
        available or needed. Produces a clear explanation plus a required
        comprehension-check practice_question.
        """
        try:
            result: TutorPlanResult = self.explain_chain.invoke(
                {
                    "student_question": student_question,
                    "current_concept": current_concept,
                    "student_memory_profile": json.dumps(student_memory_profile, indent=2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tutor planner (explain mode) failed: %s", exc)
            return TutorPlanResult(
                hint_text="That's a great question — let's think it through together.",
                applied_strategy=InstructionalStrategy.NONE,
                practice_question=f"What do you already know about {current_concept}?",
                escalation_flag=False,
            )

        if not result.practice_question:
            logger.warning("Explain-mode planner did not produce a practice_question.")

        return result

    def plan_praise(
        self,
        student_response: str,
        current_question: str,
        current_concept: str,
        evaluator_result: dict[str, Any],
        student_memory_profile: dict[str, Any],
    ) -> TutorPlanResult:
        """
        Praise-mode planning: correct + confident answer, no diagnostic_result
        available. Produces warm reinforcement plus a required next
        practice_question so the student keeps progressing.
        """
        try:
            result: TutorPlanResult = self.praise_chain.invoke(
                {
                    "student_response": student_response,
                    "current_question": current_question,
                    "current_concept": current_concept,
                    "evaluator_result": json.dumps(evaluator_result, indent=2),
                    "student_memory_profile": json.dumps(student_memory_profile, indent=2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tutor planner (praise mode) failed: %s", exc)
            return TutorPlanResult(
                hint_text="Nice work — that's correct! Ready to try another one?",
                applied_strategy=InstructionalStrategy.NONE,
                practice_question=None,
                escalation_flag=False,
            )

        if not result.practice_question:
            logger.warning("Praise-mode planner did not produce a practice_question.")

        return result

    def plan(
        self,
        student_response: str,
        current_question: str,
        current_concept: str,
        diagnostic_result: dict[str, Any],
        evaluator_result: dict[str, Any],
        student_memory_profile: dict[str, Any],
        curriculum_context: str = "",
    ) -> TutorPlanResult:
        recommended_strategy = diagnostic_result.get(
            "recommended_strategy", InstructionalStrategy.SMALL_CLUE.value
        )

        # Escalation is always handled deterministically — never let the LLM
        # improvise a hint on the escalation path.
        if recommended_strategy == InstructionalStrategy.ESCALATE_TO_TEACHER.value:
            return TutorPlanResult(
                hint_text=(
                    "You're working really hard on this, and I want to make sure "
                    "you get the best help. I'm looping in your teacher to go over "
                    "this one together. Hang tight!"
                ),
                applied_strategy=InstructionalStrategy.ESCALATE_TO_TEACHER,
                escalation_flag=True,
                escalation_reason="Diagnostic agent flagged escalate_to_teacher.",
            )

        curriculum_block = (
            f"\nCURRICULUM REFERENCE (ground your hint in this material — "
            f"do not copy verbatim):\n{curriculum_context}\n"
            if curriculum_context else ""
        )

        try:
            result: TutorPlanResult = self.chain.invoke(
                {
                    "student_response": student_response,
                    "current_question": current_question,
                    "current_concept": current_concept,
                    "diagnostic_result": json.dumps(diagnostic_result, indent=2),
                    "evaluator_result": json.dumps(evaluator_result, indent=2),
                    "student_memory_profile": json.dumps(student_memory_profile, indent=2),
                    "recommended_strategy": recommended_strategy,
                    "answer_quality": evaluator_result.get("answer_quality", "unknown"),
                    "confidence_score": evaluator_result.get("confidence_score", 0.5),
                    "curriculum_context_block": curriculum_block,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tutor planner failed: %s", exc)
            return TutorPlanResult(
                hint_text=(
                    "Let's slow down for a second — can you tell me what step "
                    "you tried first on this problem?"
                ),
                applied_strategy=InstructionalStrategy.SMALL_CLUE,
                escalation_flag=True,
                escalation_reason=f"LLM/parsing error during tutor planning: {exc}",
            )

        if (
            recommended_strategy != InstructionalStrategy.NONE.value
            and _looks_like_answer_leak(result.hint_text, current_question)
        ):
            logger.warning("Answer leak detected in tutor planner output; escalating.")
            return TutorPlanResult(
                hint_text=(
                    "I want to guide you rather than just give this one away — "
                    "let's get your teacher to walk through it with you."
                ),
                applied_strategy=result.applied_strategy,
                escalation_flag=True,
                escalation_reason="Post-generation guard detected a likely answer leak.",
            )

        return result


_default_planner: Optional[TutorPlannerAgent] = None


def _get_default_planner() -> TutorPlannerAgent:
    global _default_planner
    if _default_planner is None:
        _default_planner = TutorPlannerAgent()
    return _default_planner


# --------------------------------------------------------------------------- #
# LangGraph node entry point
# --------------------------------------------------------------------------- #

def run_tutor_planner(state: dict) -> dict:
    """
    LangGraph node function, reached from two different edges:
      - Directly from evaluator_node (route_after_evaluation: correct +
        confident) -> no "diagnostic_result" in state -> praise mode.
      - From diagnostic_node (route_after_diagnosis: not escalated) ->
        "diagnostic_result" present in state -> hint-ladder mode.

    Reads from state:
        - "student_response" (str): required
        - "current_question" (str): required
        - "current_concept" (str): required
        - "diagnostic_result" (dict): optional — presence selects the mode
        - "evaluator_result" (dict): required (from run_evaluator)
        - "student_memory_profile" (dict): optional

    Writes to state:
        - "tutor_plan_result" (dict): full TutorPlanResult
        - "hint_text" (str): the student-facing tutoring response
        - "practice_question" (str | None): follow-up practice question if any
        - "escalation_flag" (bool)
    """
    student_response = state.get("student_response")
    current_question = state.get("current_question")
    current_concept = state.get("current_concept")
    diagnostic_result = state.get("diagnostic_result")
    evaluator_result = state.get("evaluator_result", {}) or {}
    student_memory_profile = state.get("student_memory_profile", {}) or {}

    if not student_response or not current_question or not current_concept:
        raise ValueError(
            "run_tutor_planner requires 'student_response', 'current_question', "
            "and 'current_concept' in state."
        )

    planner = _get_default_planner()

    if diagnostic_result is None:
        result = planner.plan_praise(
            student_response=student_response,
            current_question=current_question,
            current_concept=current_concept,
            evaluator_result=evaluator_result,
            student_memory_profile=student_memory_profile,
        )
    else:
        # Retrieve curriculum context only for hint-ladder mode — praise mode
        # doesn't benefit from it and retrieval adds latency.
        from backend.rag.retriever import get_curriculum_context
        curriculum_context = get_curriculum_context(
            concept=current_concept,
            error_type=state.get("error_type", ""),
            misconception=(diagnostic_result or {}).get("identified_misconception", ""),
        )

        result = planner.plan(
            student_response=student_response,
            current_question=current_question,
            current_concept=current_concept,
            diagnostic_result=diagnostic_result,
            evaluator_result=evaluator_result,
            student_memory_profile=student_memory_profile,
            curriculum_context=curriculum_context,
        )

    logger.info(
        "Tutor plan -> strategy=%s, escalation_flag=%s, has_practice_q=%s",
        result.applied_strategy.value,
        result.escalation_flag,
        result.practice_question is not None,
    )

    state["tutor_plan_result"] = result.model_dump()
    state["hint_text"] = result.hint_text
    state["practice_question"] = result.practice_question
    state["escalation_flag"] = result.escalation_flag
    return state


# --------------------------------------------------------------------------- #
# Explain-mode entry point (called directly by app.py, NOT part of the
# orchestrator.py graph — there's no problem posed yet to run
# evaluator/diagnostic against)
# --------------------------------------------------------------------------- #

def run_explain(current_concept: str, student_question: str, student_memory_profile: dict) -> dict:
    """
    Standalone entry point for explain mode. Mirrors the shape of the other
    run_*(state) functions for consistency, but takes plain arguments and
    returns a plain dict (not a LangGraph state) since it isn't a graph node.

    Returns a dict with:
        - "tutor_plan_result" (dict): full TutorPlanResult
        - "hint_text" (str): the student-facing explanation
        - "practice_question" (str): the required comprehension-check question
    """
    planner = _get_default_planner()
    result = planner.plan_explain(
        student_question=student_question,
        current_concept=current_concept,
        student_memory_profile=student_memory_profile or {},
    )

    logger.info(
        "Explain mode -> has_practice_q=%s",
        result.practice_question is not None,
    )

    return {
        "tutor_plan_result": result.model_dump(),
        "hint_text": result.hint_text,
        "practice_question": result.practice_question,
    }


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ ollama serve
    $ python backend/agents/tutor_planner.py
    """
    from backend.agents.diagnostic_agent import ErrorType, InstructionalStrategy

    mock_memory = {
        "student_id": "stu_1024",
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.42, "consecutive_misses": 1},
        },
        "weak_prerequisites": ["Common Denominators"],
    }

    test_cases = [
        {
            "label": "Prerequisite gap, low confidence",
            "diagnostic_result": {
                "is_correct": False,
                "error_type": ErrorType.PREREQUISITE.value,
                "identified_misconception": "Adds numerators and denominators directly.",
                "missing_prerequisite": "Finding a common denominator",
                "confidence": 0.88,
                "reasoning": "Applied whole-number addition to fractions.",
                "recommended_strategy": InstructionalStrategy.PREREQUISITE_REVIEW.value,
            },
            "evaluator_result": {
                "answer_quality": "incorrect",
                "quality_reasoning": "Added numerators and denominators separately.",
                "confidence_score": 0.3,
                "confidence_reasoning": "Student said 'I think' and used a question mark.",
                "mastery_signal": False,
            },
            "student_response": "I think 1/2 + 1/3 = 2/5?",
        },
        {
            "label": "Correct but shaky confidence (should trigger stronger_hint)",
            "diagnostic_result": {
                "is_correct": True,
                "error_type": ErrorType.NONE.value,
                "identified_misconception": None,
                "missing_prerequisite": None,
                "confidence": 0.95,
                "reasoning": "Correct method and answer.",
                "recommended_strategy": InstructionalStrategy.NONE.value,
            },
            "evaluator_result": {
                "answer_quality": "correct",
                "quality_reasoning": "Correctly converted to common denominator.",
                "confidence_score": 0.35,
                "confidence_reasoning": "Student used 'maybe' and 'I think'.",
                "mastery_signal": False,
            },
            "student_response": "Maybe 5/6? I think I did it right but I'm not sure.",
        },
        {
            "label": "Praise mode: correct + confident, no diagnostic_result",
            "diagnostic_result": None,
            "evaluator_result": {
                "answer_quality": "correct",
                "quality_reasoning": "Correctly converted to a common denominator and added.",
                "confidence_score": 0.9,
                "confidence_reasoning": "Direct, confident phrasing with no hedging.",
                "mastery_signal": True,
                "distress_detected": False,
                "distress_reasoning": "No frustration language.",
            },
            "student_response": "1/2 + 1/3 = 3/6 + 2/6 = 5/6.",
        },
    ]

    for case in test_cases:
        print("=" * 80)
        print(f"TEST: {case['label']}")
        print(f"Response: {case['student_response']!r}")

        fake_state = {
            "student_response": case["student_response"],
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
            "evaluator_result": case["evaluator_result"],
            "student_memory_profile": mock_memory,
        }
        if case["diagnostic_result"] is not None:
            fake_state["diagnostic_result"] = case["diagnostic_result"]

        try:
            updated = run_tutor_planner(fake_state)
            print(json.dumps(updated["tutor_plan_result"], indent=2))
            print(f"\nhint_text:\n{updated['hint_text']}")
            if updated.get("practice_question"):
                print(f"\npractice_question: {updated['practice_question']}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")

    print("=" * 80)
    print("TEST: Explain mode (open question, no posed problem)")
    print("Question: 'Why is 1/2 bigger than 1/3?'")
    try:
        explained = run_explain(
            current_concept="Fraction Addition",
            student_question="Why is 1/2 bigger than 1/3?",
            student_memory_profile=mock_memory,
        )
        print(json.dumps(explained["tutor_plan_result"], indent=2))
        print(f"\nhint_text:\n{explained['hint_text']}")
        print(f"\npractice_question: {explained['practice_question']}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")

    print("=" * 80)
