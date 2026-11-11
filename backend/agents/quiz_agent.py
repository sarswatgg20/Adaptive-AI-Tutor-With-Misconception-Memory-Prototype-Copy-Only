"""
quiz_agent.py
-------------
Generates pre/post-test quiz questions for a concept, and grades free-text
quiz answers — the mechanism behind the "pre-test to post-test learning
gain" measurement the problem statement names as its #1 judging metric.

Role
====
Two responsibilities, deliberately asymmetric in how much is genuinely new:

  1. Question generation (generate_quiz_questions) — a new LLM call, grounded
     in the concept's curriculum content via the existing RAG retriever
     (backend/rag/retriever.py, unchanged), including that curriculum's
     documented common misconceptions as natural sources of good questions.

  2. Grading (grade_quiz_answer) — NOT a new agent. It reuses the existing
     EvaluatorAgent (evaluator_agent.py) directly: it already judges whether
     a free-text response is correct without ever being given an answer key,
     which is exactly what quiz grading needs — the same judgement call
     already made on every tutoring turn, just interpreted as pass/fail here.

Tech stack: Python 3.10+, LangChain, Ollama (ChatOllama), Pydantic v2.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from backend.agents.evaluator_agent import _get_default_evaluator
from backend.rag.retriever import get_curriculum_context

load_dotenv()

logger = logging.getLogger("quiz_agent")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("QUIZ_AGENT_MODEL", os.getenv("DIAGNOSTIC_AGENT_MODEL", "gemma4:e4b"))
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("QUIZ_AGENT_TEMPERATURE", "0.6"))


# --------------------------------------------------------------------------- #
# Pydantic schema
# --------------------------------------------------------------------------- #

class QuizQuestion(BaseModel):
    question_text: str = Field(
        description="A free-response question (not multiple choice) testing understanding of the concept."
    )
    reference_answer: str = Field(
        description=(
            "A brief correct answer/explanation, for internal reference only — "
            "never shown to the student."
        )
    )


class QuizQuestionSet(BaseModel):
    questions: list[QuizQuestion] = Field(
        description="Exactly the requested number of quiz questions."
    )


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Quiz Agent inside an Adaptive AI Tutor. Generate free-response \
quiz questions for a pre-test/post-test that measures a student's \
understanding of a concept BEFORE and AFTER a tutoring session.

Use the curriculum context below to ground every question in the actual \
material being taught, including its documented common misconceptions — \
write questions that would surface those misconceptions if the student \
holds them, rather than testing rote recall of definitions.

Each question needs:
- `question_text`: a free-response question a student answers in their own \
  words or with a short calculation. Never multiple choice.
- `reference_answer`: a brief correct answer, for internal grading \
  reference only — never shown to the student.

Vary phrasing and specific numbers across questions so a pre-test and a \
later post-test don't feel identical even when drawn from the same concept.

Respond ONLY with the structured JSON schema. No prose outside JSON.

CRITICAL: Return raw JSON only — no markdown code blocks, no preamble.

{format_instructions}
"""

_HUMAN_PROMPT = """\
Concept: {concept}

Curriculum context:
{curriculum_context}

Generate {n} quiz questions now.
"""


def _build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())


# --------------------------------------------------------------------------- #
# Core agent logic
# --------------------------------------------------------------------------- #

class QuizAgent:
    """
    Thin wrapper around Ollama LLM + prompt + parser for quiz question
    generation. Instantiate once and reuse across calls.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.parser = PydanticOutputParser(pydantic_object=QuizQuestionSet)
        self.prompt = _build_prompt(self.parser)
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature, format="json")
        self.chain = self.prompt | self.llm | self.parser

    def generate(self, concept: str, n: int) -> list[dict[str, Any]]:
        """
        Generates up to n quiz questions grounded in the concept's curriculum
        content. Returns [] (logged, not raised) on any failure — question
        generation must never crash the caller (quiz_store.py falls back to
        whatever's already cached).
        """
        curriculum_context = (
            get_curriculum_context(concept=concept, k=4)
            or f"(No curriculum context found for {concept}; use general knowledge.)"
        )
        try:
            result: QuizQuestionSet = self.chain.invoke(
                {
                    "concept": concept,
                    "curriculum_context": curriculum_context,
                    "n": n,
                }
            )
            return [q.model_dump() for q in result.questions[:n]]
        except Exception as exc:  # noqa: BLE001 - generation failure must not crash the caller
            logger.exception("Quiz generation failed for concept=%r: %s", concept, exc)
            return []


_default_agent: QuizAgent | None = None


def _get_default_agent() -> QuizAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = QuizAgent()
    return _default_agent


# --------------------------------------------------------------------------- #
# Module-level entry points
# --------------------------------------------------------------------------- #

def generate_quiz_questions(concept: str, n: int = 5) -> list[dict[str, Any]]:
    """Entry point used by backend/memory/quiz_store.py to top up its cache."""
    return _get_default_agent().generate(concept, n)


def grade_quiz_answer(question_text: str, current_concept: str, student_answer: str) -> bool:
    """
    Grades a single quiz answer by reusing the existing EvaluatorAgent — no
    new grading prompt. `is_correct` is True for both "correct" and
    "partially_correct" evaluator verdicts (a quiz is pass/fail per
    question; the tutoring chat is where partial credit nuance matters).
    """
    evaluator = _get_default_evaluator()
    result = evaluator.evaluate(
        student_response=student_answer,
        current_question=question_text,
        current_concept=current_concept,
    )
    return result.answer_quality.value in ("correct", "partially_correct")


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ ollama serve
    $ python -m backend.agents.quiz_agent
    """
    import json

    concept = "Fractions"

    print("=" * 80)
    print(f"Generating 3 quiz questions for {concept!r}...")
    questions = generate_quiz_questions(concept, n=3)
    print(json.dumps(questions, indent=2))

    if questions:
        print("\nGrading a correct-ish answer against question 1...")
        q = questions[0]
        is_correct = grade_quiz_answer(
            question_text=q["question_text"],
            current_concept=concept,
            student_answer="You need a common denominator before comparing or adding fractions.",
        )
        print(f"is_correct={is_correct}")

        print("\nGrading a clearly wrong answer against question 1...")
        is_correct = grade_quiz_answer(
            question_text=q["question_text"],
            current_concept=concept,
            student_answer="I don't know, maybe just add the numbers together?",
        )
        print(f"is_correct={is_correct}")

    print("=" * 80)
