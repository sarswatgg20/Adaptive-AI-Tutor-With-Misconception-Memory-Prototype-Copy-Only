"""
theme.py
--------
Presentation layer for the Streamlit app — the "Mentora" visual language
(warm paper background, gold/moss/rust/indigo/teal/pink accents, Fraunces
serif headings) lifted from the teammate's static mockup
(frontend/adaptive_tutor.html) and adapted into real Streamlit CSS +
HTML-snippet builders.

Pure presentation, no backend imports. Every function here returns a
string meant for st.markdown(..., unsafe_allow_html=True) — app.py owns
all data flow and session state; this module only turns already-computed
values into markup.

CSS selectors below target `data-testid` attributes confirmed by reading
Streamlit 1.59's compiled JS bundle directly (not guessed) — these are
stable, intentional hooks Streamlit ships specifically for custom styling
via unsafe_allow_html, independent of its internal hashed CSS classes.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Hint-ladder strategy -> visual level mapping
# --------------------------------------------------------------------------- #

_LADDER_LEVELS = {
    "none": 0,
    "retry_prompt": 0,
    "small_clue": 1,
    "stronger_hint": 2,
    "worked_example": 3,
    "prerequisite_review": 3,
}


def strategy_to_ladder_level(strategy: Optional[str]) -> int:
    """Maps an InstructionalStrategy value to a 0-3 hint-ladder display level."""
    if not strategy:
        return 0
    return _LADDER_LEVELS.get(strategy, 0)


def _mastery_tier(mastery: float) -> tuple[str, str, str]:
    """Returns (css_color_var, label, pill_bg_var) for a mastery fraction."""
    if mastery >= 0.7:
        return "var(--moss)", "mastered", "var(--moss-soft)"
    if mastery >= 0.4:
        return "var(--gold)", "developing", "var(--gold-soft)"
    return "var(--rust)", "struggling", "var(--rust-soft)"


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #

MENTORA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #f5f2eb;
    --bg-2: #fcfbfa;
    --paper: #faf8f5;
    --paper-2: #efebe2;
    --ink: #2c2520;
    --ink-soft: #6e6557;
    --muted: #8a7f70;
    --line: rgba(44, 37, 32, 0.08);

    --gold: #d4a35c;
    --gold-soft: rgba(212, 163, 92, 0.14);
    --moss: #507c5e;
    --moss-soft: rgba(80, 124, 94, 0.14);
    --rust: #ba5848;
    --rust-soft: rgba(186, 88, 72, 0.14);
    --indigo: #5c677d;
    --indigo-soft: rgba(92, 103, 125, 0.14);
    --teal: #4b8b94;
    --teal-soft: rgba(75, 139, 148, 0.14);
    --pink: #b86082;
    --pink-soft: rgba(184, 96, 130, 0.14);

    --radius: 14px;
    --shadow-sm: 0 2px 8px rgba(44, 37, 32, 0.04);
    --shadow-md: 0 10px 26px rgba(44, 37, 32, 0.06);
    --shadow-lg: 0 18px 44px rgba(44, 37, 32, 0.09);
}

/* ---------- App shell / background ---------- */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(ellipse 800px 500px at 8% -8%, rgba(184, 96, 130, 0.06), transparent 55%),
        radial-gradient(ellipse 900px 550px at 50% -10%, rgba(212, 163, 92, 0.10), transparent 60%),
        radial-gradient(ellipse 700px 500px at 95% 5%, rgba(92, 103, 125, 0.06), transparent 60%),
        var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1.2rem; max-width: 1280px; }
/* Deliberately NOT a universal `*` selector — that would also override
   Streamlit's own icon-font elements (Material Symbols Rounded), which
   render icons as ligature text and break into visible words like
   "keyboard_double_arrow_left" if their font-family gets clobbered.
   Plain inheritance from html/body is enough for normal text elements,
   since icon elements set their own explicit font-family already. */
html, body { font-family: 'Inter', sans-serif; }

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    letter-spacing: 0.1px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--bg-2);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif !important;
    font-size: 15px !important;
}

/* ---------- Tabs restyled as a pill toggle ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--paper-2);
    border: 1px solid var(--line);
    border-radius: 100px;
    padding: 4px;
    gap: 2px;
    width: fit-content;
}
[data-testid="stTab"] {
    border-radius: 100px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    color: var(--ink-soft) !important;
    padding: 8px 20px !important;
}
[data-testid="stTab"][aria-selected="true"] {
    background: var(--gold) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(212, 163, 92, 0.3);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* ---------- Buttons ---------- */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"],
button[kind="primary"], button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all .15s ease !important;
}
[data-testid="stBaseButton-primary"], button[kind="primary"] {
    background: linear-gradient(120deg, var(--gold), var(--pink)) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(212, 163, 92, 0.28) !important;
}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(212, 163, 92, 0.4) !important;
}
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: var(--bg-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--paper) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-sm);
    padding: 4px 6px;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(140deg, var(--gold), var(--pink)) !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(140deg, var(--indigo), var(--teal)) !important;
}
[data-testid="stChatMessageContent"] { color: var(--ink); }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-left: 4px solid var(--gold);
    padding: 14px 18px;
    border-radius: 10px;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; font-size: 11px !important; color: var(--ink-soft) !important; text-transform: uppercase; letter-spacing: .3px; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; color: var(--ink) !important; }

/* ---------- Alerts (st.info/success/warning) ---------- */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--line) !important;
    box-shadow: var(--shadow-sm);
}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-2);
}

/* ---------- Forms / inputs ---------- */
[data-testid="stForm"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-2);
    padding: 18px 20px !important;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 8px !important;
    border-color: var(--line) !important;
}

/* ---------- Custom components (see render_* helpers below) ---------- */
.mnt-card {
    background: var(--bg-2); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 16px 18px; box-shadow: var(--shadow-md); margin-bottom: 14px;
}
.mnt-card h4 {
    font-family: 'Fraunces', serif; font-size: 14px; font-weight: 600; color: var(--ink);
    margin: 0 0 12px 0; display: flex; align-items: center; gap: 7px;
}
.mnt-dot { width: 8px; height: 8px; border-radius: 50%; background: conic-gradient(var(--pink), var(--gold), var(--teal), var(--pink)); display: inline-block; }

.mnt-topbar {
    display: flex; align-items: center; gap: 10px; padding: 4px 0 18px 0;
    border-bottom: 1px solid var(--line); margin-bottom: 20px;
}
.mnt-brand-mark {
    width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(145deg, var(--pink), var(--gold) 55%, var(--teal));
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-weight: 700; color: #fff; font-size: 19px;
    box-shadow: 0 4px 18px rgba(212, 163, 92, 0.28);
}
.mnt-brand-name {
    font-family: 'Fraunces', serif; font-size: 24px; font-weight: 600;
    background: linear-gradient(100deg, var(--ink) 20%, var(--gold) 65%, var(--pink) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    line-height: 1.1;
}
.mnt-brand-sub { font-size: 12px; color: var(--ink-soft); margin-top: 1px; }

.mnt-ladder { display: flex; flex-direction: column-reverse; gap: 7px; }
.mnt-step {
    display: flex; align-items: center; gap: 9px; padding: 9px 11px; border-radius: 8px;
    border: 1px solid var(--line); font-size: 12.5px; color: var(--ink-soft); background: var(--bg-2);
}
.mnt-step .mnt-n {
    width: 21px; height: 21px; border-radius: 6px; background: var(--bg); border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace; color: var(--ink-soft); flex-shrink: 0;
}
.mnt-step.active { color: var(--ink); font-weight: 600; }
.mnt-step.l1.active { background: var(--moss-soft); border-color: rgba(80,124,94,.3); }
.mnt-step.l1.active .mnt-n { background: var(--moss); color: #fff; border-color: var(--moss); }
.mnt-step.l2.active { background: var(--gold-soft); border-color: rgba(212,163,92,.3); }
.mnt-step.l2.active .mnt-n { background: var(--gold); color: #fff; border-color: var(--gold); }
.mnt-step.l3.active { background: var(--rust-soft); border-color: rgba(186,88,72,.3); }
.mnt-step.l3.active .mnt-n { background: var(--rust); color: #fff; border-color: var(--rust); }
.mnt-step.done { opacity: .5; }
.mnt-ladder-escalated {
    text-align: center; font-size: 12.5px; color: var(--rust); font-weight: 600; padding: 8px;
    background: var(--rust-soft); border-radius: 8px;
}

.mnt-concept { margin-bottom: 13px; }
.mnt-concept:last-child { margin-bottom: 0; }
.mnt-concept-top { display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 5px; }
.mnt-concept-top .mnt-name { color: var(--ink); font-weight: 500; }
.mnt-concept-top .mnt-pct { font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; }
.mnt-bar { height: 7px; border-radius: 100px; background: var(--bg); overflow: hidden; border: 1px solid rgba(44,37,32,.05); }
.mnt-bar-fill { height: 100%; border-radius: 100px; }
.mnt-concept-meta { font-size: 10.5px; color: var(--ink-soft); margin-top: 5px; display: flex; gap: 7px; align-items: center; }
.mnt-pill-tiny { padding: 2px 7px; border-radius: 100px; font-size: 9.5px; font-weight: 700; letter-spacing: .2px; }

.mnt-escalation {
    border: 1px solid rgba(186, 88, 72, 0.4);
    background: linear-gradient(135deg, rgba(186, 88, 72, 0.08), rgba(184, 96, 130, 0.03));
    border-radius: var(--radius); padding: 14px 16px; margin-bottom: 14px; box-shadow: var(--shadow-md);
}
.mnt-escalation h4 { color: var(--rust); font-family: 'Fraunces', serif; font-size: 14px; margin: 0 0 6px 0; }
.mnt-escalation p { font-size: 12.5px; color: var(--ink); line-height: 1.5; margin: 0; }

.mnt-reveal {
    background: linear-gradient(120deg, var(--gold-soft), var(--pink-soft));
    border: 1px solid rgba(212, 163, 92, 0.35); border-radius: var(--radius);
    padding: 22px 26px; text-align: center; margin-bottom: 16px; box-shadow: var(--shadow-md);
}
.mnt-reveal .mnt-headline { font-family: 'Fraunces', serif; font-size: 17px; font-weight: 600; color: var(--ink); margin-bottom: 12px; }
.mnt-reveal .mnt-scoreline { display: flex; align-items: center; justify-content: center; gap: 14px; font-family: 'Fraunces', serif; }
.mnt-reveal .mnt-score { font-size: 34px; font-weight: 700; }
.mnt-reveal .mnt-score.pre { color: var(--ink-soft); }
.mnt-reveal .mnt-score.post { color: var(--moss); }
.mnt-reveal .mnt-arrow { font-size: 22px; color: var(--gold); }

.mnt-table-wrap { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow-md); margin-bottom: 14px; }
.mnt-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.mnt-table th { text-align: left; padding: 11px 14px; color: var(--ink-soft); font-weight: 600; font-size: 10.5px; text-transform: uppercase; letter-spacing: .4px; border-bottom: 1px solid var(--line); background: var(--paper); }
.mnt-table td { padding: 11px 14px; border-bottom: 1px solid var(--line); color: var(--ink); }
.mnt-table tr:last-child td { border-bottom: none; }

.mnt-risk { padding: 3px 9px; border-radius: 100px; font-size: 10.5px; font-weight: 700; white-space: nowrap; }
.mnt-risk.low { background: var(--moss-soft); color: var(--moss); }
.mnt-risk.med { background: var(--gold-soft); color: #a3792f; }
.mnt-risk.high { background: var(--rust-soft); color: var(--rust); }

/* Turn-outcome pills used inline in the chat internals panel (app.py) */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; margin-left: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.pill-ok   { background: var(--moss-soft); color: var(--moss); }
.pill-warn { background: var(--rust-soft); color: var(--rust); }
.pill-info { background: var(--indigo-soft); color: var(--indigo); }
</style>
"""
MENTORA_CSS = textwrap.dedent(MENTORA_CSS).strip()


# --------------------------------------------------------------------------- #
# Component builders
# --------------------------------------------------------------------------- #

def render_topbar() -> str:
    return textwrap.dedent(
        """
        <div class="mnt-topbar">
            <div class="mnt-brand-mark">A</div>
            <div>
                <div class="mnt-brand-name">Adaptive AI Tutor</div>
                <div class="mnt-brand-sub">misconception memory &middot; hint ladder &middot; teacher escalation</div>
            </div>
        </div>
        """
    ).strip()


def render_hint_ladder(level: int, escalated: bool = False) -> str:
    """
    Renders the 3-step hint-ladder widget. `level` is 0-3 (0 = no
    intervention needed yet / correct), from strategy_to_ladder_level().
    """
    if escalated:
        inner = '<div class="mnt-ladder-escalated">🚨 Escalated to teacher</div>'
        return f'<div class="mnt-card"><h4><span class="mnt-dot"></span>Hint Ladder</h4><div class="mnt-ladder">{inner}</div></div>'

    steps = [
        (1, "l1", "Small clue"),
        (2, "l2", "Stronger hint"),
        (3, "l3", "Worked example"),
    ]
    rows = []
    for n, cls, label in steps:
        state = "active" if n == level else ("done" if n < level else "")
        rows.append(
            f'<div class="mnt-step {cls} {state}"><div class="mnt-n">{n}</div>{label}</div>'
        )
    return (
        '<div class="mnt-card"><h4><span class="mnt-dot"></span>Hint Ladder</h4>'
        f'<div class="mnt-ladder">{"".join(rows)}</div></div>'
    )


def render_concept_mastery_bars(concept_mastery: dict[str, Any], title: str = "Concept Mastery") -> str:
    if not concept_mastery:
        return (
            f'<div class="mnt-card"><h4><span class="mnt-dot"></span>{_escape(title)}</h4>'
            '<div style="font-size:12.5px;color:var(--ink-soft);">No concepts tracked yet.</div></div>'
        )

    rows = []
    for concept, data in concept_mastery.items():
        mastery = float(data.get("mastery", 0.0))
        pct = round(mastery * 100)
        color, tier_label, tier_bg = _mastery_tier(mastery)
        misses = data.get("consecutive_misses", 0)
        rows.append(
            textwrap.dedent(
                f"""
                <div class="mnt-concept">
                    <div class="mnt-concept-top">
                        <span class="mnt-name">{_escape(concept)}</span>
                        <span class="mnt-pct" style="color:{color}">{pct}%</span>
                    </div>
                    <div class="mnt-bar"><div class="mnt-bar-fill" style="width:{pct}%;background:{color}"></div></div>
                    <div class="mnt-concept-meta">
                        <span class="mnt-pill-tiny" style="background:{tier_bg};color:{color}">{tier_label}</span>
                        <span>{misses} consecutive miss{'es' if misses != 1 else ''}</span>
                    </div>
                </div>
                """
            ).strip()
        )
    return (
        f'<div class="mnt-card"><h4><span class="mnt-dot"></span>{_escape(title)}</h4>{"".join(rows)}</div>'
    )


def render_escalation_banner(reason: str, risk_signals: Optional[list[str]] = None) -> str:
    signals_html = ""
    if risk_signals:
        signals_html = (
            '<div style="margin-top:8px;">'
            + "".join(
                f'<span class="mnt-risk high" style="margin-right:6px;">{_escape(s)}</span>'
                for s in risk_signals
            )
            + "</div>"
        )
    return (
        '<div class="mnt-escalation"><h4>🚨 Escalated to your teacher</h4>'
        f'<p>{_escape(reason)}</p>{signals_html}</div>'
    )


def render_reveal_card(pre_score: float, post_score: float, concept: str, round_number: int) -> str:
    pre_pct = round(pre_score * 100)
    post_pct = round(post_score * 100)
    gained = post_pct - pre_pct
    headline = (
        f"🎉 Round {round_number} on {_escape(concept)} — "
        + ("+" if gained >= 0 else "")
        + f"{gained} points"
    )
    return textwrap.dedent(
        f"""
        <div class="mnt-reveal">
            <div class="mnt-headline">{headline}</div>
            <div class="mnt-scoreline">
                <span class="mnt-score pre">{pre_pct}%</span>
                <span class="mnt-arrow">&rarr;</span>
                <span class="mnt-score post">{post_pct}%</span>
            </div>
        </div>
        """
    ).strip()


def render_risk_pill(signals: str) -> str:
    """`signals` is the comma-joined string already stored in the escalation log row."""
    if "Cheating risk" in signals or "Distress" in signals:
        level = "high"
    elif "Repeated misses" in signals:
        level = "med"
    else:
        level = "low"
    return f'<span class="mnt-risk {level}">{_escape(signals)}</span>'


def _review_status(next_review_at: datetime) -> tuple[str, str]:
    """
    Returns (label, risk_level) for a next_review_at timestamp. SQLite
    doesn't preserve tzinfo on round-trip, so next_review_at usually comes
    back naive even though it was written as UTC — compare against a
    matching naive/aware "now" rather than assuming either.
    """
    now = datetime.now(timezone.utc) if next_review_at.tzinfo else datetime.now()
    delta_days = (next_review_at.date() - now.date()).days

    if next_review_at <= now:
        return "Overdue", "high"
    if delta_days == 0:
        return "Due today", "med"
    return f"Due in {delta_days}d", "low"


def render_due_reviews_banner(due_rows: list[dict[str, Any]]) -> str:
    """Compact "due for review" card for the Student tab sidebar — only
    called when due_rows is non-empty."""
    items = []
    for row in due_rows:
        label, _ = _review_status(row["next_review_at"])
        items.append(
            f'<div class="mnt-concept-meta" style="justify-content:space-between;margin:6px 0;">'
            f'<span class="mnt-name" style="color:var(--ink);">{_escape(row["concept"])}</span>'
            f'<span class="mnt-risk high">{_escape(label)}</span></div>'
        )
    return (
        '<div class="mnt-card"><h4><span class="mnt-dot"></span>📅 Due for Review</h4>'
        + "".join(items)
        + "</div>"
    )


def render_review_schedule_table(schedule_rows: list[dict[str, Any]]) -> str:
    if not schedule_rows:
        return ""
    body_rows = []
    for row in schedule_rows:
        label, risk = _review_status(row["next_review_at"])
        last_reviewed = row["last_reviewed_at"].strftime("%b %d")
        next_due = row["next_review_at"].strftime("%b %d")
        mastery_pct = round(row.get("mastery_at_schedule", 0.0) * 100)
        body_rows.append(
            "<tr>"
            f"<td>{_escape(row['concept'])}</td>"
            f"<td>{mastery_pct}%</td>"
            f"<td>{last_reviewed}</td>"
            f"<td>{next_due}</td>"
            f'<td><span class="mnt-risk {risk}">{_escape(label)}</span></td>'
            "</tr>"
        )
    return (
        '<div class="mnt-table-wrap"><table class="mnt-table">'
        "<thead><tr><th>Concept</th><th>Mastery</th><th>Last Reviewed</th>"
        "<th>Next Due</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_escalation_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header_cells = "".join(f"<th>{_escape(k)}</th>" for k in rows[0].keys() if k != "Risk Signals")
    header_cells += "<th>Risk</th>"
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{_escape(v)}</td>" for k, v in row.items() if k != "Risk Signals"
        )
        cells += f"<td>{render_risk_pill(str(row.get('Risk Signals', '')))}</td>"
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="mnt-table-wrap"><table class="mnt-table">'
        f"<thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def render_student_roster_table(rows: list[dict[str, Any]]) -> str:
    """
    `rows`: list of {"name", "avg_mastery" (float|None), "attempts" (int),
    "escalations" (int)} — one per demo student (backend/memory/students.py).
    Status pill: any escalations -> high, low avg mastery -> med, else low.
    """
    if not rows:
        return ""
    body_rows = []
    for row in rows:
        avg_mastery = row.get("avg_mastery")
        mastery_label = f"{round(avg_mastery * 100)}%" if avg_mastery is not None else "—"
        if row.get("escalations", 0) > 0:
            status, risk = "Needs attention", "high"
        elif avg_mastery is not None and avg_mastery < 0.4:
            status, risk = "Struggling", "med"
        elif avg_mastery is None:
            status, risk = "No data yet", "low"
        else:
            status, risk = "On track", "low"
        body_rows.append(
            "<tr>"
            f"<td>{_escape(row['name'])}</td>"
            f"<td>{mastery_label}</td>"
            f"<td>{row.get('attempts', 0)}</td>"
            f"<td>{row.get('escalations', 0)}</td>"
            f'<td><span class="mnt-risk {risk}">{_escape(status)}</span></td>'
            "</tr>"
        )
    return (
        '<div class="mnt-table-wrap"><table class="mnt-table">'
        "<thead><tr><th>Student</th><th>Avg. Mastery</th><th>Attempts</th>"
        "<th>Escalations</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )
