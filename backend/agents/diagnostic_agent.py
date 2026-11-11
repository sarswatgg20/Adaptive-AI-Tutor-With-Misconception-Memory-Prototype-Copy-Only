"""
diagnostic_agent.py
--------------------
Diagnostic Agent for the Adaptive AI Tutor with Misconception Memory.

Role
====
This agent runs ONLY when the Evaluator Agent has already determined the
student's response is not a confident-correct answer (route_after_evaluation
in orchestrator.py sends anything that isn't "correct + confidence >= 0.6"
here). Given the student's response, the concept currently being taught, the
Evaluator's quality/confidence verdict, and the student's long-term memory
profile, it must:

    1. Decide whether the answer is CORRECT (but shaky) or an ERROR.
    2. If it's an error, classify it as one of:
         - CARELESS      (slip / typo / arithmetic mistake, concept is fine)
         - CONCEPTUAL     (student misunderstands the current concept itself)
         - PREREQUISITE   (the gap traces back to an earlier, foundational skill)
    3. Name the specific misconception (if any), in student-facing language.
    4. Recommend an instructional strategy for the downstream Tutor Planner
       agent to execute (e.g., "small_clue", "worked_example",
       "prerequisite_review").

Note: this agent does NOT decide teacher escalation. That decision is made
deterministically by the orchestrator's escalation gate (route_after_diagnosis),
which reads student_memory_profile's consecutive_misses and the Evaluator's
distress_detected flag — no LLM call needed for that routing step.

This module is designed to be used as a single node inside a LangGraph
`StateGraph`. It reads from and writes to a shared `state: dict`, so it can
be dropped into `orchestrator.py` with a single `graph.add_node(...)` call.

Tech stack: Python 3.10+, LangChain (ChatOpenAI + structured output),
Pydantic v2.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any, Optional

from backend.memory.misconception_graph import get_prerequisites

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Environment & logging setup
# --------------------------------------------------------------------------- #

load_dotenv()  # Pulls OPENAI_API_KEY, etc. from backend/.env

logger = logging.getLogger("diagnostic_agent")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("DIAGNOSTIC_AGENT_MODEL", "gemma4:e4b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = 0.0  # Diagnosis should be deterministic, not creative


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #

class ErrorType(str, Enum):
    """Classification of a student's error."""

    NONE = "None"                # Answer is correct / no error detected
    CARELESS = "Careless"        # Slip, typo, arithmetic mistake
    CONCEPTUAL = "Conceptual"    # Misunderstands the current concept
    PREREQUISITE = "Prerequisite"  # Gap traces to an earlier foundational skill


class InstructionalStrategy(str, Enum):
    """
    Strategies the Hint Generator agent knows how to execute.
    Keeping this as a closed enum (rather than free text) makes the
    hand-off between agents in the LangGraph state machine reliable.
    """

    NONE = "none"                              # No intervention needed, answer is correct
    RETRY_PROMPT = "retry_prompt"               # "Double check your work" - for careless errors
    SMALL_CLUE = "small_clue"                   # Level 1 of hint ladder
    STRONGER_HINT = "stronger_hint"              # Level 2 of hint ladder
    WORKED_EXAMPLE = "worked_example"            # Level 3 of hint ladder
    PREREQUISITE_REVIEW = "prerequisite_review"  # Detour to reteach a foundational skill
    ESCALATE_TO_TEACHER = "escalate_to_teacher"  # Repeated/severe gap -> human handoff.
                                                  # Never set by the Diagnostic Agent's LLM call —
                                                  # only ever assigned by orchestrator.py's
                                                  # deterministic escalation gate.


class DiagnosticResult(BaseModel):
    """
    Strict, structured output produced by the Diagnostic Agent.
    This is what gets written into the shared LangGraph state and consumed
    by escalation_agent.py and tutor_planner.py.
    """

    is_correct: bool = Field(
        description="True if the student's response demonstrates the correct answer/understanding."
    )
    error_type: ErrorType = Field(
        description="Category of the error: None, Careless, Conceptual, or Prerequisite."
    )
    identified_misconception: Optional[str] = Field(
        default=None,
        description=(
            "A short, specific, student-facing description of the misconception "
            "(e.g., 'Believes a larger denominator always means a larger fraction'). "
            "Null if is_correct is True."
        ),
    )
    missing_prerequisite: Optional[str] = Field(
        default=None,
        description=(
            "If error_type is Prerequisite, the name of the earlier concept that "
            "needs to be reviewed (e.g., 'Comparing unit fractions'). Otherwise null."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's confidence in this diagnosis, from 0.0 to 1.0.",
    )
    reasoning: str = Field(
        description="One or two sentence explanation of why this classification was chosen."
    )
    recommended_strategy: InstructionalStrategy = Field(
        description="The instructional strategy the Hint Generator agent should execute next."
    )

    @field_validator("identified_misconception")
    @classmethod
    def _blank_to_none(cls, v: Optional[str]) -> Optional[str]:
        # Normalize empty strings coming back from the LLM to None
        if v is not None and not v.strip():
            return None
        return v


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Diagnostic Agent inside an Adaptive AI Tutor. You run AFTER the \
Evaluator Agent, which has already scored the response's quality and \
confidence — you only run at all because the Evaluator found the response was \
NOT a confident-correct answer. Your job is to analyze WHY, and produce a \
precise diagnostic classification. You do NOT teach, hint, or explain the \
answer to the student — a separate Tutor Planner agent handles that using \
your output. You also do NOT decide whether to escalate to a teacher — that \
is a separate, deterministic decision made outside your scope; never pick \
"escalate_to_teacher" as recommended_strategy.

Classification rules (apply in this order):

1. CORRECT (but shaky) — The Evaluator already flagged this response as \
   correct with low confidence, or as ambiguous. If the response is genuinely \
   mathematically/conceptually correct, set is_correct=true, error_type="None", \
   recommended_strategy="none" — the Tutor Planner will calibrate tone/strength \
   using the Evaluator's confidence_score.

2. CARELESS — The student clearly knows the underlying concept (their \
   reasoning, setup, or method is sound) but made a slip: a typo, an \
   arithmetic slip, mis-copied a number, or a one-off attention error. \
   There is no evidence of misunderstanding. \
   recommended_strategy="retry_prompt".

3. CONCEPTUAL — The student's answer reveals a flawed mental model of the \
   CURRENT concept itself (e.g., for fraction comparison: "1/3 is bigger than \
   1/2 because 3 is bigger than 2"). The error is systematic, not a slip. \
   recommended_strategy should follow the hint ladder: use "small_clue" if \
   this is the student's first or second miss on this concept (check \
   student_memory_profile), "stronger_hint" if they have missed it before, \
   or "worked_example" if they have persistently struggled (3+ prior misses \
   on this concept per the memory profile).

4. PREREQUISITE — The error cannot be fixed by re-explaining the current \
   concept alone, because it depends on a more foundational skill the \
   student has not mastered (check student_memory_profile's weak/low-mastery \
   prerequisites, and infer new ones from the response if needed). \
   If `known_prerequisites` below lists any candidates, PREFER naming one of \
   them verbatim in missing_prerequisite when it genuinely fits the gap — \
   this keeps prerequisite names consistent across turns. Only name \
   something not on that list if none of the known ones explain the error. \
   recommended_strategy="prerequisite_review".

Be conservative: only claim CONCEPTUAL or PREREQUISITE if the response gives \
clear evidence. If you are unsure between Careless and Conceptual, prefer \
Careless and lower your confidence score.

You MUST respond using the exact structured schema you are given. Do not add \
any extra commentary outside the schema fields.

{format_instructions}
"""

_HUMAN_PROMPT = """\
Current concept being taught: {current_concept}

Known prerequisites for this concept, from the curriculum graph (prefer \
naming one of these verbatim as missing_prerequisite when it fits — see \
rule 4): {known_prerequisites}

Student's memory profile (mastery levels, known weak prerequisites, and \
recent history for this concept):
{student_memory_profile}

Evaluator Agent result (already scored this response's quality/confidence — \
use as context, but form your own error classification):
{evaluator_result}

Student's most recent response:
\"\"\"{student_response}\"\"\"

Diagnose this response now.
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

class DiagnosticAgent:
    """
    Thin wrapper around an LLM + prompt + parser that performs one diagnosis.
    Instantiate once (e.g., in orchestrator.py) and reuse across turns to
    avoid rebuilding the chain every call.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.parser = PydanticOutputParser(pydantic_object=DiagnosticResult)
        self.prompt = _build_prompt(self.parser)

        # `with_structured_output` gives us native tool-calling based
        # structured output when available (more reliable than pure text
        # parsing), while the PydanticOutputParser above still supplies
        # human-readable format instructions inside the prompt as a
        # belt-and-suspenders fallback.
        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json"  # Forces the local model to output valid JSON
        )
        
        self.structured_llm = llm.with_structured_output(DiagnosticResult)
        self.chain = self.prompt | self.structured_llm

    def diagnose(
        self,
        student_response: str,
        current_concept: str,
        student_memory_profile: dict[str, Any],
        evaluator_result: dict[str, Any],
    ) -> DiagnosticResult:
        """
        Run the diagnostic classification for a single student turn.

        Args:
            student_response: Raw text of the student's latest answer.
            current_concept: The topic currently being taught, e.g. "Fraction Addition".
            student_memory_profile: Dict describing concept mastery / known
                weak prerequisites / recent attempt history for this student.
            evaluator_result: The Evaluator Agent's quality/confidence verdict
                for this same response (already computed upstream).

        Internally fetches known_prerequisites for current_concept from the
        Neo4j-backed curriculum graph (backend/memory/misconception_graph.py)
        to ground missing_prerequisite in a consistent vocabulary. Returns []
        (not an error) if Neo4j isn't reachable or the concept isn't mapped.

        Returns:
            A validated DiagnosticResult.
        """
        known_prerequisites = get_prerequisites(current_concept)

        try:
            result: DiagnosticResult = self.chain.invoke(
                {
                    "student_response": student_response,
                    "current_concept": current_concept,
                    "known_prerequisites": json.dumps(known_prerequisites),
                    "student_memory_profile": json.dumps(
                        student_memory_profile, indent=2
                    ),
                    "evaluator_result": json.dumps(evaluator_result, indent=2),
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001 - we want a safe fallback for any LLM/parse failure
            logger.exception("Diagnostic agent failed to produce structured output: %s", exc)
            # Fail-safe fallback: never crash the orchestrator graph. Default
            # to a low-confidence conceptual diagnosis and let a human-ish
            # downstream default (small_clue) handle it gently.
            return DiagnosticResult(
                is_correct=False,
                error_type=ErrorType.CONCEPTUAL,
                identified_misconception=None,
                missing_prerequisite=None,
                confidence=0.0,
                reasoning=(
                    "Diagnostic agent encountered an internal error and could not "
                    "confidently classify this response; defaulting to a cautious hint."
                ),
                recommended_strategy=InstructionalStrategy.SMALL_CLUE,
            )


# Module-level singleton so LangGraph nodes don't re-instantiate the LLM
# client on every invocation. orchestrator.py can still create its own
# DiagnosticAgent(...) instance if it needs a different model/temperature.
_default_agent: Optional[DiagnosticAgent] = None


def _get_default_agent() -> DiagnosticAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = DiagnosticAgent()
    return _default_agent


# --------------------------------------------------------------------------- #
# LangGraph node entry point
# --------------------------------------------------------------------------- #

def run_diagnostic(state: dict) -> dict:
    """
    LangGraph node function. Runs after evaluator_node, only on the branch
    where route_after_evaluation decided the response wasn't confident-correct.

    Expected keys read from `state`:
        - "student_response" (str): required
        - "current_concept" (str): required
        - "student_memory_profile" (dict): optional, defaults to {}
        - "evaluator_result" (dict): required, produced by run_evaluator

    Keys written back into `state`:
        - "diagnostic_result" (dict): the DiagnosticResult, JSON-serializable,
          for easy consumption by downstream nodes / the Streamlit frontend.
        - "error_type" (str): convenience top-level copy.
        - "recommended_strategy" (str): convenience top-level copy, consumed
          directly by tutor_planner.py.

    Returns:
        The updated state dict (LangGraph merges this back into the graph's
        shared state).
    """
    student_response = state.get("student_response")
    current_concept = state.get("current_concept")
    student_memory_profile = state.get("student_memory_profile", {}) or {}
    evaluator_result = state.get("evaluator_result", {}) or {}

    if not student_response or not current_concept:
        raise ValueError(
            "run_diagnostic requires 'student_response' and 'current_concept' "
            "to be present in state."
        )

    agent = _get_default_agent()
    result = agent.diagnose(
        student_response=student_response,
        current_concept=current_concept,
        student_memory_profile=student_memory_profile,
        evaluator_result=evaluator_result,
    )

    logger.info(
        "Diagnosis for concept=%r -> error_type=%s, strategy=%s, confidence=%.2f",
        current_concept,
        result.error_type.value,
        result.recommended_strategy.value,
        result.confidence,
    )

    state["diagnostic_result"] = result.model_dump()
    state["error_type"] = result.error_type.value
    state["recommended_strategy"] = result.recommended_strategy.value
    return state


# --------------------------------------------------------------------------- #
# Mock execution block for standalone testing
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    Run this file directly to sanity-check the agent before wiring it into
    orchestrator.py:

        $ python backend/agents/diagnostic_agent.py

    Requires OPENAI_API_KEY to be set (via .env or environment).
    """

    mock_memory_profile = {
        "student_id": "stu_1024",
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.42, "consecutive_misses": 1},
            "Comparing Unit Fractions": {"mastery": 0.31, "consecutive_misses": 2},
        },
        "weak_prerequisites": ["Comparing Unit Fractions", "Common Denominators"],
        "recent_attempts": [
            {"concept": "Fraction Addition", "correct": False},
        ],
    }

    test_cases = [
        {
            "label": "Prerequisite gap example",
            "student_response": "1/2 + 1/3 = 2/5 because you just add the tops and add the bottoms.",
            "current_concept": "Fraction Addition",
            "evaluator_result": {
                "answer_quality": "incorrect",
                "quality_reasoning": "Added numerators and denominators separately.",
                "confidence_score": 0.8,
                "confidence_reasoning": "Stated the answer directly with no hedging.",
                "mastery_signal": False,
                "distress_detected": False,
                "distress_reasoning": "Neutral tone, no frustration language.",
            },
        },
        {
            "label": "Careless slip example",
            "student_response": (
                "To add 1/4 + 1/4 I need a common denominator, which is 4. "
                "So it's 1/4 + 1/4 = 2/8."
            ),
            "current_concept": "Fraction Addition",
            "evaluator_result": {
                "answer_quality": "incorrect",
                "quality_reasoning": "Correct method, arithmetic slip on the final step.",
                "confidence_score": 0.7,
                "confidence_reasoning": "Direct, confident phrasing.",
                "mastery_signal": False,
                "distress_detected": False,
                "distress_reasoning": "No frustration language.",
            },
        },
        {
            "label": "Correct but shaky confidence example",
            "student_response": "Maybe 1/2 + 1/4 = 2/4 + 1/4 = 3/4? I think that's right.",
            "current_concept": "Fraction Addition",
            "evaluator_result": {
                "answer_quality": "correct",
                "quality_reasoning": "Correctly converted to a common denominator and added.",
                "confidence_score": 0.3,
                "confidence_reasoning": "Hedged with 'maybe' and 'I think'.",
                "mastery_signal": False,
                "distress_detected": False,
                "distress_reasoning": "No frustration language.",
            },
        },
    ]

    diagnostic_agent = DiagnosticAgent()

    for case in test_cases:
        print("=" * 80)
        print(f"TEST CASE: {case['label']}")
        print(f"Student response: {case['student_response']!r}")

        fake_state = {
            "student_response": case["student_response"],
            "current_concept": case["current_concept"],
            "student_memory_profile": mock_memory_profile,
            "evaluator_result": case["evaluator_result"],
        }

        try:
            updated_state = run_diagnostic(fake_state)
            print(json.dumps(updated_state["diagnostic_result"], indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"ERROR running diagnostic node: {e}")

    print("=" * 80)
    print("Done. If OPENAI_API_KEY was not set, the calls above should have")
    print("raised an authentication error rather than hanging silently.")
    #test