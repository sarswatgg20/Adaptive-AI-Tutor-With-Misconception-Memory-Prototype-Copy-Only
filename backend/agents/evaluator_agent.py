"""
evaluator_agent.py
------------------
Evaluator Agent for the Adaptive AI Tutor with Misconception Memory.

Role
====
Runs FIRST in the tutoring graph, before the Diagnostic Agent. It produces
the signals the orchestrator's routers need to decide whether the student
even needs diagnosis:

    1. answer_quality  — a clean "correct / partially_correct / incorrect"
       verdict with a short reasoning note.

    2. confidence_score — 0.0–1.0, inferred from the student's linguistic
       hedges ("I think", "maybe", "not sure") vs confident assertions.
       A correct answer said with low confidence is still a gap; a wrong
       answer said with high confidence signals a deep misconception.

    3. mastery_signal — True if this response is positive evidence toward
       updating long-term mastery (e.g., correct + high confidence). The
       memory-update node reads this to decide whether to increment mastery.

    4. distress_detected — True if the response shows frustration,
       disengagement, or "giving up" language. Combined with the student's
       consecutive-miss streak, this drives the orchestrator's escalation
       gate (route_after_diagnosis) without needing an extra LLM call.

route_after_evaluation (in orchestrator.py) reads answer_quality +
confidence_score to decide: correct + confident -> straight to the Tutor
Planner's praise mode; otherwise -> the Diagnostic Agent.

Tech stack: Python 3.10+, LangChain, Ollama (ChatOllama), Pydantic v2.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Environment & logging
# --------------------------------------------------------------------------- #

load_dotenv()

logger = logging.getLogger("evaluator_agent")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("EVALUATOR_AGENT_MODEL", "gemma4:e4b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("EVALUATOR_AGENT_TEMPERATURE", "0.1"))

# Cheating-risk speed check: local LLMs are unreliable at multi-step
# arithmetic buried inside a structured-output call, so this one signal is
# computed deterministically in Python (same "don't make the LLM guess at
# something code can compute" philosophy as the orchestrator's escalation
# gate) rather than asked of the model. A human composing new written
# reasoning — not just typing a memorized short answer — rarely sustains
# more than ~3 words/second; short answers are exempted since a memorized
# number could plausibly be typed instantly either way.
MAX_PLAUSIBLE_WORDS_PER_SECOND = 3.0
MIN_WORDS_FOR_SPEED_CHECK = 8


# --------------------------------------------------------------------------- #
# Pydantic schema
# --------------------------------------------------------------------------- #

class AnswerQuality(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class EvaluatorResult(BaseModel):
    """
    Structured output from the Evaluator Agent.
    Written into the shared LangGraph state; consumed by tutor_planner.py
    and (eventually) the memory-update node.
    """

    answer_quality: AnswerQuality = Field(
        description=(
            "Overall verdict on the student's response: "
            "'correct' (right answer, right reasoning), "
            "'partially_correct' (right direction but incomplete or minor error), "
            "'incorrect' (wrong answer or fundamentally flawed reasoning)."
        )
    )
    quality_reasoning: str = Field(
        description=(
            "One sentence explaining the quality verdict — what specifically "
            "was right or wrong about the student's answer."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated student confidence in their own answer, 0.0–1.0. "
            "Infer from hedging language ('I think', 'maybe', 'not sure' → low) "
            "vs direct assertions ('The answer is', 'I know that' → high). "
            "Neutral phrasing with no hedges → ~0.6."
        ),
    )
    confidence_reasoning: str = Field(
        description="One sentence explaining the confidence estimate."
    )
    mastery_signal: bool = Field(
        description=(
            "True if this response is positive evidence toward mastery of the "
            "current concept — i.e., answer_quality is 'correct' AND "
            "confidence_score >= 0.6. False otherwise."
        )
    )
    distress_detected: bool = Field(
        default=False,
        description=(
            "True if the response shows frustration, disengagement, or "
            "'giving up' language (e.g., 'I give up', 'I hate this', "
            "'this is impossible'). False for normal/neutral responses, "
            "even if incorrect."
        ),
    )
    distress_reasoning: str = Field(
        default="",
        description="One sentence explaining the distress_detected verdict.",
    )
    cheating_risk_detected: bool = Field(
        default=False,
        description=(
            "True only when multiple suspicious signals combine: a very fast "
            "response time for a complex problem, an answer noticeably more "
            "sophisticated/complete than the concept calls for or than the "
            "student's own recent responses, a sharp style mismatch vs. "
            "recent responses, or copy-paste-shaped formatting. False for a "
            "merely correct or confident answer on its own."
        ),
    )
    cheating_risk_reasoning: str = Field(
        default="",
        description="One sentence explaining the cheating_risk_detected verdict.",
    )

    @field_validator("answer_quality", mode="before")
    @classmethod
    def _normalise_quality(cls, v: Any) -> Any:
        valid = [q.value for q in AnswerQuality]
        if not v or v not in valid:
            logger.warning("Evaluator returned invalid answer_quality %r; defaulting to 'incorrect'.", v)
            return AnswerQuality.INCORRECT.value
        return v

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("mastery_signal", "distress_detected", "cheating_risk_detected", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Evaluator Agent inside an Adaptive AI Tutor. You run FIRST, before \
any other agent — your job is to assess the QUALITY, CONFIDENCE, and emotional \
state of the student's response. You do NOT teach, hint, classify error types, \
or decide what to do next — separate Diagnostic and Tutor Planner agents handle \
that, using your output as input.

You receive:
- `current_question`: the problem the student is solving.
- `current_concept`: the concept being taught.
- `student_response`: what the student wrote.

Your outputs:

1. `answer_quality` — one of:
   - "correct": the answer and reasoning are both right (minor phrasing ok).
   - "partially_correct": the right direction but incomplete, or correct answer
     with wrong/missing reasoning, or a small arithmetic slip on otherwise
     sound method.
   - "incorrect": wrong answer OR fundamentally flawed reasoning.

2. `quality_reasoning` — 1 sentence, specific. Say WHAT was right or wrong.

3. `confidence_score` — 0.0 to 1.0, inferred from linguistic cues:
   - Hedges ("I think", "maybe", "I'm not sure", "is it?", "?") → 0.1–0.4
   - Neutral, factual phrasing with no hedges → 0.5–0.7
   - Strong, direct assertions ("The answer is", "I know") → 0.8–1.0
   A correct answer with lots of hedging scores LOW confidence.

4. `confidence_reasoning` — 1 sentence explaining the confidence estimate.

5. `mastery_signal` — true ONLY when answer_quality == "correct" AND \
   confidence_score >= 0.6. False otherwise.

6. `distress_detected` — true if the response shows frustration, disengagement, \
   or "giving up" language (e.g., "I give up", "I hate this", "this is \
   impossible", "I don't get it and never will"). False for a normal wrong \
   answer, hedging, or confusion that doesn't rise to distress — being unsure \
   is NOT distress.

7. `distress_reasoning` — 1 sentence explaining the distress_detected verdict.

8. `cheating_risk_detected` — a separate deterministic typing-speed check \
   already ran in code before this prompt (`Precomputed speed check` \
   below) — trust it completely, don't re-derive it yourself. Your job is \
   ONLY the style judgment:
     Whenever `recent_student_responses` is non-empty, compare this \
     response's vocabulary/formality/structure against them. If 2+ of the \
     recent responses are casual/short/hesitant (things like "idk", \
     "maybe", lowercase-only, no punctuation, under ~10 words, guesses) AND \
     the CURRENT response uses formal textbook vocabulary the casual ones \
     never use (e.g., "least common multiple", "equivalent fraction", \
     numbered steps, "Step 1:") — that is a style jump. Set true.
     Example: recent=["idk maybe 2/5?", "is it 5/6 i think", "wait how do \
     u even do this"], current="To add fractions with unlike denominators, \
     we determine the least common multiple..." -> style jump -> true.
   Set true if EITHER the precomputed speed check says IMPLAUSIBLE, OR your \
   own style judgment finds a jump as described above. If the precomputed \
   check says "not applicable" AND `recent_student_responses` is "(none \
   available)", you have no signal at all — default to false regardless of \
   how polished the answer reads. Do not infer cheating from correctness, \
   confidence, or politeness alone.

9. `cheating_risk_reasoning` — 1 sentence explaining the cheating_risk_detected verdict.

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

Time taken to respond: {response_time_seconds}

Precomputed speed check (already computed in code — trust this, don't \
recompute it): {speed_check}

Student's recent past responses this session, for style comparison:
{recent_student_responses}

Evaluate the response now.
"""


def _build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())


def _compute_speed_check(
    student_response: str, response_time_seconds: Optional[float]
) -> tuple[str, bool]:
    """
    Deterministically checks whether student_response's word count is
    humanly implausible to have been composed and typed in
    response_time_seconds. Returns (description for the prompt, implausible).
    """
    if response_time_seconds is None or response_time_seconds <= 0:
        return "not applicable (no timing data)", False

    word_count = len(student_response.split())
    if word_count < MIN_WORDS_FOR_SPEED_CHECK:
        return (
            f"not applicable ({word_count} words is short enough a memorized "
            "answer could be typed instantly either way)",
            False,
        )

    words_per_second = word_count / response_time_seconds
    implausible = words_per_second > MAX_PLAUSIBLE_WORDS_PER_SECOND
    verdict = "IMPLAUSIBLE" if implausible else "plausible"
    description = (
        f"{verdict} — {word_count} words in {response_time_seconds:.1f}s "
        f"(~{words_per_second:.1f} words/sec; humans rarely sustain "
        f">{MAX_PLAUSIBLE_WORDS_PER_SECOND:.0f} words/sec composing new reasoning)"
    )
    return description, implausible


# --------------------------------------------------------------------------- #
# Core agent logic
# --------------------------------------------------------------------------- #

class EvaluatorAgent:
    """
    Thin wrapper around Ollama LLM + prompt + parser for answer evaluation.
    Instantiate once and reuse across turns.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.parser = PydanticOutputParser(pydantic_object=EvaluatorResult)
        self.prompt = _build_prompt(self.parser)
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json",
        )
        self.chain = self.prompt | self.llm | self.parser

    def evaluate(
        self,
        student_response: str,
        current_question: str,
        current_concept: str,
        response_time_seconds: Optional[float] = None,
        recent_student_responses: Optional[list[str]] = None,
    ) -> EvaluatorResult:
        speed_description, implausible_speed = _compute_speed_check(
            student_response, response_time_seconds
        )
        try:
            result: EvaluatorResult = self.chain.invoke(
                {
                    "student_response": student_response,
                    "current_question": current_question,
                    "current_concept": current_concept,
                    "response_time_seconds": (
                        f"{response_time_seconds:.1f} seconds"
                        if response_time_seconds is not None
                        else "unknown (not tracked for this turn)"
                    ),
                    "speed_check": speed_description,
                    "recent_student_responses": (
                        json.dumps(recent_student_responses)
                        if recent_student_responses
                        else "(none available)"
                    ),
                }
            )
            if implausible_speed and not result.cheating_risk_detected:
                result.cheating_risk_detected = True
                result.cheating_risk_reasoning = (
                    f"Deterministic speed check: {speed_description}."
                )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluator agent failed: %s", exc)
            return EvaluatorResult(
                answer_quality=AnswerQuality.INCORRECT,
                quality_reasoning="Evaluation failed due to an internal error; defaulting to incorrect.",
                confidence_score=0.5,
                confidence_reasoning="Could not assess confidence due to evaluation error.",
                mastery_signal=False,
                distress_detected=False,
                distress_reasoning="Could not assess distress due to evaluation error.",
            )


_default_evaluator: Optional[EvaluatorAgent] = None


def _get_default_evaluator() -> EvaluatorAgent:
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = EvaluatorAgent()
    return _default_evaluator


# --------------------------------------------------------------------------- #
# LangGraph node entry point
# --------------------------------------------------------------------------- #

def run_evaluator(state: dict) -> dict:
    """
    LangGraph node function. Runs FIRST in the graph, before diagnostic_node.

    Reads from state:
        - "student_response" (str): required
        - "current_question" (str): required
        - "current_concept" (str): required

    Writes to state:
        - "evaluator_result" (dict): full EvaluatorResult as JSON-serializable dict
        - "answer_quality" (str): "correct" | "partially_correct" | "incorrect"
        - "confidence_score" (float): 0.0–1.0
        - "mastery_signal" (bool): positive mastery evidence flag
        - "distress_detected" (bool): frustration/disengagement flag, read by
          the orchestrator's escalation gate (route_after_diagnosis)
        - "cheating_risk_detected" (bool): suspicious-answer flag, read by the
          same escalation gate

    Also reads (both optional):
        - "response_time_seconds" (float): wall-clock time app.py measured
          between the tutor's last message and this student response
        - "student_id" (str): used to fetch recent past responses (via
          backend.memory.conversation_store) for style-mismatch comparison
    """
    student_response = state.get("student_response")
    current_question = state.get("current_question")
    current_concept = state.get("current_concept")

    if not student_response or not current_question or not current_concept:
        raise ValueError(
            "run_evaluator requires 'student_response', 'current_question', "
            "and 'current_concept' in state."
        )

    recent_student_responses = None
    student_id = state.get("student_id")
    if student_id:
        from backend.memory.conversation_store import get_recent_messages

        history = get_recent_messages(student_id, limit=10)
        recent_student_responses = [
            m["content"] for m in history if m.get("role") == "user"
        ][-5:]

    evaluator = _get_default_evaluator()
    result = evaluator.evaluate(
        student_response=student_response,
        current_question=current_question,
        current_concept=current_concept,
        response_time_seconds=state.get("response_time_seconds"),
        recent_student_responses=recent_student_responses,
    )

    logger.info(
        "Evaluation -> quality=%s, confidence=%.2f, mastery_signal=%s, distress=%s, cheating_risk=%s",
        result.answer_quality.value,
        result.confidence_score,
        result.mastery_signal,
        result.distress_detected,
        result.cheating_risk_detected,
    )

    state["evaluator_result"] = result.model_dump()
    state["answer_quality"] = result.answer_quality.value
    state["distress_detected"] = result.distress_detected
    state["cheating_risk_detected"] = result.cheating_risk_detected
    state["confidence_score"] = result.confidence_score
    state["mastery_signal"] = result.mastery_signal
    return state


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ ollama serve
    $ python backend/agents/evaluator_agent.py
    """

    test_cases = [
        {
            "label": "Incorrect + low confidence",
            "student_response": "I think it's 2/5? I'm not really sure.",
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
        },
        {
            "label": "Correct + high confidence",
            "student_response": "1/2 + 1/3 = 3/6 + 2/6 = 5/6.",
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
        },
        {
            "label": "Partially correct + medium confidence",
            "student_response": "You need a common denominator. I think it's 5/6 but I'm not positive.",
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
        },
        {
            "label": "Correct + low confidence (mastery_signal should be False)",
            "student_response": "Maybe 5/6? I kind of guessed.",
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
        },
        {
            "label": "Distress language (distress_detected should be True)",
            "student_response": "I give up, I hate fractions, this is impossible.",
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
        },
        {
            "label": "Suspiciously fast + polished vs. casual history (cheating_risk_detected should be True)",
            "student_response": (
                "To add fractions with unlike denominators, we determine the least common "
                "multiple of the denominators, which is 6. Converting each fraction to an "
                "equivalent fraction with denominator 6 yields 3/6 and 2/6. Summing the "
                "numerators gives 5/6, already in lowest terms since gcd(5, 6) = 1."
            ),
            "current_question": "What is 1/2 + 1/3?",
            "current_concept": "Fraction Addition",
            "response_time_seconds": 3.0,
            "recent_student_responses": ["idk maybe 2/5?", "is it 5/6 i think", "wait how do u even do this"],
        },
    ]

    agent = EvaluatorAgent()

    for case in test_cases:
        print("=" * 80)
        print(f"TEST: {case['label']}")
        print(f"Response: {case['student_response']!r}")

        try:
            if "response_time_seconds" in case or "recent_student_responses" in case:
                # These two are normally fetched by run_evaluator() from
                # state["response_time_seconds"] (set by app.py) and the
                # conversation_store (via student_id) respectively — call
                # evaluate() directly here to test them without needing a
                # live DB-backed student history.
                result = agent.evaluate(
                    student_response=case["student_response"],
                    current_question=case["current_question"],
                    current_concept=case["current_concept"],
                    response_time_seconds=case.get("response_time_seconds"),
                    recent_student_responses=case.get("recent_student_responses"),
                )
                print(json.dumps(result.model_dump(), indent=2))
            else:
                fake_state = {
                    "student_response": case["student_response"],
                    "current_question": case["current_question"],
                    "current_concept": case["current_concept"],
                }
                updated = run_evaluator(fake_state)
                print(json.dumps(updated["evaluator_result"], indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")

    print("=" * 80)
