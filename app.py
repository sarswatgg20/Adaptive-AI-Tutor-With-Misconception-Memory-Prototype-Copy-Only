import streamlit as st
import time

from backend.agents.orchestrator import tutor_app
from backend.agents.tutor_planner import run_explain
from backend.agents.quiz_agent import grade_quiz_answer
from backend.memory.student_graph_store import load_student_profile
from backend.memory.conversation_store import save_message, get_recent_messages
from backend.memory.quiz_store import (
    get_or_generate_questions,
    record_attempt,
    next_round_number,
    get_round_summary,
)
from backend.evals.metrics import compute_all_metrics
from backend.memory.graph_viz import build_prerequisite_graph_dot
from backend.memory.spaced_repetition import (
    schedule_next_review,
    get_due_reviews,
    get_schedule,
)
from backend.memory.students import list_students
from backend.memory.eval_store import get_interactions
from frontend.theme import (
    MENTORA_CSS,
    render_topbar,
    render_hint_ladder,
    render_concept_mastery_bars,
    render_escalation_banner,
    render_reveal_card,
    render_escalation_table,
    render_due_reviews_banner,
    render_review_schedule_table,
    render_student_roster_table,
    strategy_to_ladder_level,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Adaptive AI Tutor",
    page_icon="🧠",
    layout="wide",
)

# --- Mentora theme (palette/typography lifted from frontend/adaptive_tutor.html).
# The .pill/.pill-ok/etc. classes used by the chat internals panel below live
# inside MENTORA_CSS itself now — a previous version concatenated a second
# <style> block onto this string, which broke: a blank line + indented
# content mid-string gets parsed as a literal Markdown code block instead of
# raw HTML (CommonMark's 4-space-indent rule), so it rendered as visible text
# instead of applying as CSS. Keeping everything in one flush-left constant
# avoids that footgun. ---
st.markdown(MENTORA_CSS, unsafe_allow_html=True)

st.markdown(render_topbar(), unsafe_allow_html=True)

DEFAULT_GREETING = [{"role": "assistant", "content": "Hi! What would you like to learn today?"}]

# Demo student registry (backend/memory/students.py) — a real login system
# is out of scope for this hackathon demo; a small fixed roster is enough
# to make the Teacher Dashboard read as a real classroom instead of a
# single-student toy. Run `python -m backend.memory.students --seed` once
# to populate the two non-interactive demo profiles.
_STUDENT_REGISTRY = list_students()
_STUDENT_IDS = list(_STUDENT_REGISTRY.keys())
_DEFAULT_STUDENT_ID = _STUDENT_IDS[0]


def _load_student_session(student_id: str) -> None:
    """
    (Re)loads every piece of per-student session state for the given
    student — chat history (SQL), memory profile (Neo4j), and all
    in-progress turn/quiz state reset to neutral. Called once on first page
    load and again whenever the sidebar's student switcher changes
    selection, which is what actually makes switching between demo
    students possible (a single set of `if "x" not in st.session_state`
    one-time-init blocks, as this used to be, can't support that).
    """
    history = get_recent_messages(student_id)
    st.session_state.messages = (
        [{"role": m["role"], "content": m["content"]} for m in history]
        if history
        else DEFAULT_GREETING
    )
    st.session_state.memory_profile = load_student_profile(student_id)
    # The specific problem currently being worked on. None means the
    # student's next message is a fresh question, not an answer to
    # anything — see the explain-mode branch below.
    st.session_state.active_question = None
    # Wall-clock time of the last tutor message, used to measure
    # response_time_seconds for the cheating-risk signal.
    st.session_state.last_tutor_message_at = None
    # Last turn's raw result (for internals + the inline escalation banner).
    st.session_state.last_final_state = {}
    # Pre/post-test quiz flow: quiz_phase is None (normal tutoring) |
    # "pretest" | "posttest"; quiz_round is the current round number once a
    # pre-test has started, else None; quiz_reveal is set after a post-test
    # is graded, shown once then cleared.
    st.session_state.quiz_phase = None
    st.session_state.quiz_questions = []
    st.session_state.quiz_round = None
    st.session_state.quiz_reveal = None


# --- Session State: which demo student is currently active in the Student
# Workspace tab. First load defaults to the registry's first entry. ---
if "active_student_id" not in st.session_state:
    st.session_state.active_student_id = _DEFAULT_STUDENT_ID
    _load_student_session(_DEFAULT_STUDENT_ID)


# =====================================================================
# SIDEBAR — session controls + internals toggle
# =====================================================================
_CURRICULUM_TOPICS = [
    "Fractions",
    "Probability",
    "Algebra",
    "Geometry",
    "Decimals and Percentages",
    "Other (custom)",
]

with st.sidebar:
    st.subheader("👤 Student")
    _chosen_student_id = st.selectbox(
        "Viewing as",
        _STUDENT_IDS,
        index=_STUDENT_IDS.index(st.session_state.active_student_id),
        format_func=lambda sid: _STUDENT_REGISTRY[sid],
        help="Switch between demo students — each has their own chat history, mastery, and schedule.",
    )
    if _chosen_student_id != st.session_state.active_student_id:
        st.session_state.active_student_id = _chosen_student_id
        _load_student_session(_chosen_student_id)
        st.rerun()

    # Used throughout the rest of the script — plain variable, not a
    # hardcoded constant, so it always reflects whichever student is active.
    STUDENT_ID = st.session_state.active_student_id

    st.divider()
    st.subheader("⚙️ Session")
    topic_choice = st.selectbox(
        "Current Topic",
        _CURRICULUM_TOPICS,
        help="Select the concept being studied. RAG retrieves curriculum context for the first 5 topics.",
    )
    if topic_choice == "Other (custom)":
        current_concept = st.text_input(
            "Enter topic name",
            placeholder="e.g. Trigonometry, Statistics…",
        )
    else:
        current_concept = topic_choice

    # Spaced-repetition "due for review" banner — only shown when something
    # is actually due, computed fresh each run (backend/memory/spaced_repetition.py).
    _due = get_due_reviews(STUDENT_ID)
    if _due:
        st.markdown(render_due_reviews_banner(_due), unsafe_allow_html=True)

    # Hint-ladder widget: always visible, derived from last_final_state
    # (already populated by the graph — no new data needed). Shows a
    # neutral level-0 ladder before any turn has run yet.
    _last = st.session_state.last_final_state
    if _last and not _last.get("explain_mode"):
        _escalated = "escalation_result" in _last
        _strategy = (_last.get("tutor_plan_result") or {}).get("applied_strategy") or _last.get(
            "recommended_strategy"
        )
        st.markdown(
            render_hint_ladder(strategy_to_ladder_level(_strategy), escalated=_escalated),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(render_hint_ladder(0), unsafe_allow_html=True)

    show_internals = st.toggle(
        "🔬 Show tutor internals",
        value=False,
        help="Reveal memory profile, diagnostics, and which agent handled the last turn.",
    )

    # Only offered once a round is underway (pre-test already taken) and
    # we're not already mid-quiz — always visible during tutoring so the
    # student/demo-driver can trigger it whenever they're ready.
    if st.session_state.quiz_round is not None and st.session_state.quiz_phase is None:
        st.divider()
        if st.button("🎯 Take Post-Test"):
            st.session_state.quiz_questions = get_or_generate_questions(current_concept, n=5)
            st.session_state.quiz_phase = "posttest"
            st.rerun()

    if st.button("🔄 Reset session"):
        # Resets this browser session's view only — does not delete the
        # student's persisted mastery graph (Neo4j) or conversation history
        # (SQL); reloading the page will bring both back.
        st.session_state.messages = DEFAULT_GREETING
        st.session_state.memory_profile = {
            "student_id": STUDENT_ID,
            "concept_mastery": {},
            "weak_prerequisites": [],
            "recent_attempts": [],
        }
        st.session_state.last_final_state = {}
        st.session_state.active_question = None
        st.session_state.last_tutor_message_at = None
        st.session_state.quiz_phase = None
        st.session_state.quiz_questions = []
        st.session_state.quiz_round = None
        st.session_state.quiz_reveal = None
        st.rerun()

    if show_internals:
        st.divider()
        st.markdown(
            render_concept_mastery_bars(
                st.session_state.memory_profile.get("concept_mastery", {}),
                title="Live Memory Profile",
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Raw JSON"):
            st.json(st.session_state.memory_profile)

        last = st.session_state.last_final_state
        if last:
            if last.get("explain_mode"):
                st.caption("💬 Last explanation")
                st.json(last.get("tutor_plan_result", {}))
                path = "explain mode (open question, no grading)"
            else:
                st.caption("🩺 Last evaluation")
                st.json(last.get("evaluator_result", {}))
                if "diagnostic_result" in last:
                    st.caption("🔍 Last diagnostic")
                    st.json(last.get("diagnostic_result", {}))

                # Which path ran last turn?
                if "escalation_result" in last:
                    path = "evaluator → diagnostic → escalator"
                elif "diagnostic_result" in last:
                    path = "evaluator → diagnostic → tutor_planner (hint)"
                else:
                    path = "evaluator → tutor_planner (praise)"
            st.caption("🧭 Agent path (last turn)")
            st.code(path, language=None)


# =====================================================================
# TABS
# =====================================================================
tab_student, tab_teacher = st.tabs(["🎓 Student Workspace", "📊 Teacher Dashboard"])


# ---------------------------------------------------------------------
# Helper: pull the graph's memory-update result into session state
# ---------------------------------------------------------------------
def update_memory_from_turn(final_state: dict, concept: str, student_msg: str) -> None:
    # The graph's memory_update node (backend/memory/student_profile.py) has
    # already folded this turn's outcome into student_memory_profile (Neo4j)
    # AND logged the full turn detail to eval_store.interaction_log — just
    # adopt the updated profile here. The Teacher Dashboard's escalation
    # table reads directly from eval_store (backend/memory/eval_store.py's
    # get_interactions), not from session state, so it works for any
    # student regardless of what's active in this tab.
    updated_profile = final_state.get("student_memory_profile")
    if updated_profile:
        st.session_state.memory_profile = updated_profile


# ---------------------------------------------------------------------
# Helper: render the pre-test / post-test quiz view
# ---------------------------------------------------------------------
def _render_quiz_view(phase: str, concept: str) -> None:
    """
    Renders the active quiz (pre-test or post-test) as a form. On submit,
    grades every answer via the existing EvaluatorAgent (quiz_agent.py's
    grade_quiz_answer — no new grading prompt), records each attempt, and
    either seeds initial concept_mastery (pre-test) or computes the
    learning-gain reveal (post-test).
    """
    title = "📝 Pre-Test — let's see where you're starting" if phase == "pretest" else "📝 Post-Test — let's see how far you've come"
    st.subheader(title)

    questions = st.session_state.quiz_questions
    if not questions:
        st.warning("Couldn't generate quiz questions right now (is Ollama running?) — skipping this step.")
        if st.button("Continue without testing"):
            st.session_state.quiz_phase = None
            if phase == "pretest":
                st.session_state.quiz_round = None
            st.rerun()
        return

    answers: dict[int, str] = {}
    with st.form(key=f"quiz_form_{phase}"):
        for i, q in enumerate(questions):
            answers[i] = st.text_area(f"**Q{i + 1}.** {q['question_text']}", key=f"{phase}_{q['id']}")
        submitted = st.form_submit_button(
            "Submit Pre-Test" if phase == "pretest" else "Submit Post-Test"
        )

    if submitted:
        correct_count = 0
        with st.spinner("Grading..."):
            for i, q in enumerate(questions):
                is_correct = grade_quiz_answer(
                    question_text=q["question_text"],
                    current_concept=concept,
                    student_answer=answers[i],
                )
                record_attempt(
                    STUDENT_ID,
                    concept,
                    st.session_state.quiz_round,
                    phase,
                    q["id"],
                    answers[i],
                    is_correct,
                )
                if is_correct:
                    correct_count += 1
        score = correct_count / len(questions)

        if phase == "pretest":
            # Seed initial mastery from the pre-test score instead of the
            # flat default — this is the "probe current level before
            # teaching" connection (flow step #2).
            profile = st.session_state.memory_profile
            profile.setdefault("concept_mastery", {})
            profile["concept_mastery"][concept] = {"mastery": score, "consecutive_misses": 0}
            schedule_next_review(STUDENT_ID, concept, score)
            st.session_state.quiz_phase = None
        else:
            summary = get_round_summary(STUDENT_ID, concept, st.session_state.quiz_round)
            st.session_state.quiz_reveal = {
                "concept": concept,
                "round": st.session_state.quiz_round,
                "pre_score": summary["pre_score"] if summary["pre_score"] is not None else score,
                "post_score": score,
            }
            st.session_state.quiz_phase = None
            st.session_state.quiz_round = None  # allows a new round to be started later

        st.rerun()


# =====================================================================
# TAB 1 — STUDENT WORKSPACE
# =====================================================================
with tab_student:
    st.header("Active Learning Session")

    if st.session_state.quiz_phase in ("pretest", "posttest"):
        _render_quiz_view(st.session_state.quiz_phase, current_concept)

    else:
        if st.session_state.quiz_round is None:
            st.info(f"Start a learning session on **{current_concept}** to measure your progress.")
            if st.button("▶ Start Learning Session (Pre-Test)"):
                st.session_state.quiz_questions = get_or_generate_questions(current_concept, n=5)
                st.session_state.quiz_round = next_round_number(STUDENT_ID, current_concept)
                st.session_state.quiz_phase = "pretest"
                st.rerun()

        if st.session_state.quiz_reveal:
            reveal = st.session_state.quiz_reveal
            st.markdown(
                render_reveal_card(
                    reveal["pre_score"], reveal["post_score"], reveal["concept"], reveal["round"]
                ),
                unsafe_allow_html=True,
            )
            if st.button("Continue"):
                st.session_state.quiz_reveal = None
                st.rerun()

        chat_container = st.container(height=440)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question (e.g., Why is 1/2 bigger than 1/3?)"):
            response_time_seconds = (
                time.time() - st.session_state.last_tutor_message_at
                if st.session_state.last_tutor_message_at is not None
                else None
            )
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            save_message(STUDENT_ID, "user", prompt, concept=current_concept)

            with st.spinner("Thinking..."):
                try:
                    if st.session_state.active_question is None:
                        # Nothing posed yet — this is an open question, not an
                        # answer to grade. Explain mode skips Evaluator/Diagnostic
                        # entirely and poses the next problem itself.
                        result = run_explain(
                            current_concept=current_concept,
                            student_question=prompt,
                            student_memory_profile=st.session_state.memory_profile,
                        )
                        final_state = {"explain_mode": True, **result}
                        st.session_state.last_final_state = final_state

                        tutor_response = result["hint_text"]
                        if result.get("practice_question"):
                            tutor_response += f"\n\n**Try this:** {result['practice_question']}"
                            st.session_state.active_question = result["practice_question"]

                    else:
                        # Answering a posed problem — run the full
                        # evaluator -> diagnostic -> hint-ladder pipeline.
                        initial_state = {
                            "student_id": STUDENT_ID,
                            "current_question": st.session_state.active_question,
                            "current_concept": current_concept,
                            "student_response": prompt,
                            "student_memory_profile": st.session_state.memory_profile,
                            "response_time_seconds": response_time_seconds,
                        }
                        final_state = tutor_app.invoke(initial_state)
                        st.session_state.last_final_state = final_state

                        tutor_response = (
                            final_state.get("hint_text")
                            or final_state.get("teacher_summary")
                            or "I'm not sure how to respond to that — could you rephrase?"
                        )
                        if final_state.get("practice_question"):
                            tutor_response += f"\n\n**Try this:** {final_state['practice_question']}"
                            st.session_state.active_question = final_state["practice_question"]
                        # else: no new question was generated (retry_prompt / small_clue /
                        # stronger_hint / escalation) — keep active_question as-is, so the
                        # student's next reply is still evaluated against the SAME
                        # problem and the hint ladder can escalate.

                        update_memory_from_turn(final_state, current_concept, prompt)

                except Exception as exc:
                    tutor_response = f"⚠️ Tutor error: {exc}"

            save_message(STUDENT_ID, "assistant", tutor_response, concept=current_concept)

            st.session_state.messages.append({"role": "assistant", "content": tutor_response})
            st.session_state.last_tutor_message_at = time.time()
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(tutor_response)

            # Escalation banner: always visible (not gated behind the internals
            # toggle) — a real student/demo viewer needs to see this happened,
            # not just infer it from the chat message tone.
            if "escalation_result" in st.session_state.last_final_state:
                _final = st.session_state.last_final_state
                _signals = []
                if _final.get("cheating_risk_detected"):
                    _signals.append("Cheating risk")
                if _final.get("distress_detected"):
                    _signals.append("Distress")
                if not _signals:
                    _signals.append("Repeated misses")
                st.markdown(
                    render_escalation_banner(
                        "Your teacher has been notified and will follow up with you on this together.",
                        risk_signals=_signals,
                    ),
                    unsafe_allow_html=True,
                )

            # If internals are on, surface the agent path inline too
            if show_internals and st.session_state.last_final_state:
                last = st.session_state.last_final_state
                if last.get("explain_mode"):
                    st.markdown('<span class="pill pill-info">Explained · new question posed</span>', unsafe_allow_html=True)
                elif "escalation_result" in last:
                    reason = "cheating risk" if last.get("cheating_risk_detected") else (
                        "distress" if last.get("distress_detected") else "repeated misses"
                    )
                    st.markdown(f'<span class="pill pill-warn">Escalated to teacher · {reason}</span>', unsafe_allow_html=True)
                elif "diagnostic_result" in last:
                    q = last.get("evaluator_result", {}).get("answer_quality", "?")
                    st.markdown(f'<span class="pill pill-info">Tutored · answer: {q}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="pill pill-ok">Correct · praised + next question</span>', unsafe_allow_html=True)


# =====================================================================
# TAB 2 — TEACHER DASHBOARD (live session data)
# =====================================================================
with tab_teacher:
    st.header("Classroom Monitoring & Escalations")

    # --- Class roster: every demo student, summary stats pulled fresh from
    # persisted stores (not session state) so this always reflects reality
    # regardless of which student is active in the Student Workspace tab. ---
    st.subheader("🏫 Class Roster")
    _roster_rows = []
    for _sid, _name in _STUDENT_REGISTRY.items():
        _roster_profile = load_student_profile(_sid)
        _masteries = [
            d.get("mastery", 0.0) for d in _roster_profile.get("concept_mastery", {}).values()
        ]
        _roster_rows.append(
            {
                "name": _name,
                "avg_mastery": (sum(_masteries) / len(_masteries)) if _masteries else None,
                "attempts": len(_roster_profile.get("recent_attempts", [])),
                "escalations": sum(1 for row in get_interactions(_sid) if row.get("escalated")),
            }
        )
    st.markdown(render_student_roster_table(_roster_rows), unsafe_allow_html=True)

    st.divider()

    # --- Per-student detail view: defaults to whichever student is active
    # in the Student Workspace tab, but independently changeable — a
    # teacher can review one student's history while another is mid-session. ---
    _teacher_view_student_id = st.selectbox(
        "Viewing details for",
        _STUDENT_IDS,
        index=_STUDENT_IDS.index(STUDENT_ID),
        format_func=lambda sid: _STUDENT_REGISTRY[sid],
        key="teacher_view_student_id",
    )
    profile = load_student_profile(_teacher_view_student_id)
    concept_mastery = profile.get("concept_mastery", {})
    attempts = profile.get("recent_attempts", [])
    _interactions = get_interactions(_teacher_view_student_id)
    _escalation_rows_raw = [row for row in _interactions if row.get("escalated")]

    # --- Top metrics ---
    total_attempts = len(attempts)
    correct_attempts = sum(1 for a in attempts if a.get("correct"))
    accuracy = f"{(correct_attempts / total_attempts * 100):.0f}%" if total_attempts else "—"
    num_escalations = len(_escalation_rows_raw)

    col1, col2, col3 = st.columns(3)
    col1.metric("Attempts logged", total_attempts)
    col2.metric("Answer accuracy", accuracy)
    col3.metric("Escalations", num_escalations)

    st.divider()

    # --- Escalations, from persisted interaction_log
    # (backend/memory/eval_store.py) — works for any student regardless of
    # the current chat session, unlike the old session-only log. ---
    st.subheader("🚨 Active Escalations (Requires Intervention)")
    if _escalation_rows_raw:
        _escalation_table_rows = []
        for row in _escalation_rows_raw:
            _risk_signals = []
            if row.get("cheating_risk_detected"):
                _risk_signals.append("Cheating risk")
            if row.get("distress_detected"):
                _risk_signals.append("Distress")
            if row.get("consecutive_misses_before", 0) > 0:
                _risk_signals.append("Repeated misses")
            # teacher_summary is the LLM's narrative explanation, persisted
            # since this feature; older rows recorded before that column
            # existed fall back to a synthesized description.
            _trigger = row.get("teacher_summary") or (
                f"{' & '.join(_risk_signals) or 'Escalation triggered'} on {row['concept']}"
            )
            _escalation_table_rows.append(
                {
                    "Time": row["created_at"].strftime("%b %d, %H:%M"),
                    "Topic": row["concept"],
                    "Trigger": _trigger[:100],
                    "Risk Signals": ", ".join(_risk_signals) or "—",
                    "Misses": row.get("consecutive_misses_before", 0),
                }
            )
        st.markdown(render_escalation_table(_escalation_table_rows), unsafe_allow_html=True)
    else:
        st.success("No escalations yet. ✅")

    colA, colB = st.columns(2)

    with colA:
        st.markdown(
            render_concept_mastery_bars(concept_mastery, title="🧠 Concept Mastery & Misconceptions"),
            unsafe_allow_html=True,
        )

    with colB:
        weak = profile.get("weak_prerequisites", [])
        weak_items = "".join(f"<li>{w}</li>" for w in weak) if weak else ""
        st.markdown(
            '<div class="mnt-card"><h4><span class="mnt-dot"></span>🧩 Weak Prerequisites Detected</h4>'
            + (
                f'<ul style="margin:0 0 10px 18px;padding:0;font-size:12.5px;color:var(--ink);">{weak_items}</ul>'
                if weak
                else '<div style="font-size:12.5px;color:var(--ink-soft);">No missing prerequisites detected yet.</div>'
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        if weak:
            st.info(
                "Recommended: schedule targeted revision on the prerequisites above "
                "before advancing the current topic."
            )

    st.divider()

    # --- Spaced repetition schedule: a simple 1/3/7-day interval scheduler
    # (backend/memory/spaced_repetition.py), updated automatically whenever
    # mastery is recomputed (chat turns and pre-tests). ---
    st.subheader("📅 Spaced Repetition Schedule")
    schedule_rows = get_schedule(_teacher_view_student_id)
    if schedule_rows:
        st.markdown(render_review_schedule_table(schedule_rows), unsafe_allow_html=True)
    else:
        st.info("No review schedule yet — it fills in as concepts are studied.")

    st.divider()

    # --- Misconception & prerequisite graph: the curriculum's REQUIRES
    # structure (Neo4j, backend/memory/misconception_graph.py) rendered via
    # backend/memory/graph_viz.py, with this student's weak/mastered nodes
    # highlighted. No new dependency — st.graphviz_chart renders a raw DOT
    # string client-side. ---
    st.subheader("🕸️ Misconception & Prerequisite Graph")
    graph_dot = build_prerequisite_graph_dot(current_concept, profile)
    if graph_dot:
        st.caption(
            "🔴 weak prerequisite &nbsp;·&nbsp; 🟢 mastered &nbsp;·&nbsp; 🟡 developing "
            "&nbsp;·&nbsp; 🟠 struggling &nbsp;·&nbsp; ⚪ not yet studied &nbsp;·&nbsp; "
            "**bold border** = current topic",
            unsafe_allow_html=True,
        )
        st.graphviz_chart(graph_dot, width="stretch")
    else:
        st.info(
            "No curriculum graph data yet. Seed it with "
            "`python -m backend.memory.misconception_graph --seed` "
            "(requires Neo4j to be running)."
        )

    st.divider()

    # --- Evaluation metrics: the six judging metrics named in the problem
    # statement, computed purely from logged data (backend/evals/metrics.py)
    # — no LLM calls, deterministic and cheap to recompute live. ---
    st.subheader("📈 Evaluation Metrics")

    def _fmt_pct(value):
        return f"{value * 100:.0f}%" if value is not None else "—"

    def _fmt_delta(value):
        if value is None:
            return "—"
        sign = "+" if value >= 0 else ""
        return f"{sign}{value * 100:.0f} pts"

    metrics = compute_all_metrics(_teacher_view_student_id)

    row1 = st.columns(3)
    row1[0].metric(
        "Learning Gain",
        _fmt_delta(metrics["learning_gain"]["value"]),
        help="Average pre-test -> post-test score change across completed rounds.",
    )
    row1[1].metric(
        "Misconception Recall",
        _fmt_pct(metrics["misconception_recall"]["value"]),
        help="How often a recurring misconception is correctly recognized as one already seen.",
    )
    row1[2].metric(
        "Adaptation Quality",
        _fmt_pct(metrics["adaptation_quality"]["value"]),
        help="How often the hint ladder escalates (never regresses) as misses build up.",
    )

    row2 = st.columns(3)
    row2[0].metric(
        "Escalation Precision",
        _fmt_pct(metrics["escalation_precision"]["value"]),
        help="Of turns escalated to a teacher, the fraction that were actually warranted.",
    )
    row2[1].metric(
        "Memory Usefulness",
        _fmt_delta(metrics["memory_usefulness"]["value"]),
        help="Mastery gain on turns with prior history, vs. a first attempt (below).",
    )
    row2[2].metric(
        "Hint Quality",
        _fmt_pct(metrics["hint_quality"]["value"]),
        help="Of hint-ladder turns, the fraction whose very next attempt succeeded.",
    )

    total_samples = sum(m["sample_size"] for m in metrics.values())
    if total_samples == 0:
        st.info(
            "No metrics yet — these fill in as students take quizzes and chat with the tutor. "
            "Run `python -m backend.agents.orchestrator` for a scripted example, or use the Student tab."
        )
    else:
        baseline = metrics["memory_usefulness"]["baseline_value"]
        if baseline is not None:
            st.caption(
                f"Memory Usefulness baseline (first attempt, no prior history): {_fmt_delta(baseline)}"
            )
