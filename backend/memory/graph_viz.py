"""
graph_viz.py
------------
Renders the curriculum prerequisite graph (backend/memory/misconception_graph.py)
as a Graphviz DOT string, with a given student's weak and mastered concepts
highlighted — the visual behind the Teacher Dashboard's "Misconception &
Prerequisite Graph" panel.

No new dependency: Streamlit's st.graphviz_chart(dot_string) renders a raw
DOT string client-side (dagre-d3) without needing the `graphviz` pip package
at all — that package is only required if you pass a graphviz.Digraph
*object*, not a plain string. So this module just builds DOT source by hand.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.memory.misconception_graph import get_all_concepts, get_all_edges, get_prerequisites
from backend.memory.neo4j_client import is_available

# Fill colors, in priority order (checked top to bottom per node).
_COLOR_WEAK = "#f4a29e"          # red/orange — flagged weak prerequisite
_COLOR_MASTERED = "#8fd3a0"      # green — mastery >= 0.7
_COLOR_DEVELOPING = "#f5d68a"    # gold — mastery 0.4-0.7
_COLOR_STRUGGLING = "#f0b48a"    # orange — mastery < 0.4
_COLOR_UNKNOWN = "#e6e6e6"       # light gray — no data on this concept yet
_COLOR_CURRENT_BORDER = "#1a56db"  # bold blue border — current topic


def _escape(name: str) -> str:
    return name.replace('"', '\\"')


def _node_color(
    concept: str,
    weak_prerequisites: set[str],
    concept_mastery: dict[str, Any],
) -> str:
    if concept in weak_prerequisites:
        return _COLOR_WEAK
    entry = concept_mastery.get(concept)
    if entry is not None:
        mastery = entry.get("mastery", 0.0)
        if mastery >= 0.7:
            return _COLOR_MASTERED
        if mastery >= 0.4:
            return _COLOR_DEVELOPING
        return _COLOR_STRUGGLING
    return _COLOR_UNKNOWN


def build_prerequisite_graph_dot(
    current_concept: Optional[str], student_memory_profile: dict[str, Any]
) -> Optional[str]:
    """
    Builds a DOT digraph of the curriculum prerequisite structure, colored
    by the given student's mastery/weak-prerequisite data. Returns None if
    there's nothing to draw at all (Neo4j has no seeded graph AND the
    profile has no concept data either) so the caller can show a fallback
    instead of an empty graph.
    """
    concept_mastery: dict[str, Any] = student_memory_profile.get("concept_mastery", {}) or {}
    weak_prerequisites = set(student_memory_profile.get("weak_prerequisites", []) or [])

    edges = get_all_edges()
    nodes = set(get_all_concepts())
    for concept, prereq in edges:
        nodes.add(concept)
        nodes.add(prereq)
    nodes.update(concept_mastery.keys())
    nodes.update(weak_prerequisites)
    if current_concept:
        nodes.add(current_concept)

    if not nodes:
        return None

    lines = [
        "digraph PrerequisiteGraph {",
        "  rankdir=TB;",
        '  bgcolor="transparent";',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, margin=0.15];',
        '  edge [color="#8891a6", arrowsize=0.8];',
    ]

    for concept in sorted(nodes):
        color = _node_color(concept, weak_prerequisites, concept_mastery)
        extra = ""
        if concept == current_concept:
            extra = f', color="{_COLOR_CURRENT_BORDER}", penwidth=3'
        lines.append(
            f'  "{_escape(concept)}" [fillcolor="{color}"{extra}];'
        )

    for concept, prereq in edges:
        # Draw prerequisite -> concept: a top-to-bottom read is "master this
        # first, then this" rather than the REQUIRES edge's own direction.
        lines.append(f'  "{_escape(prereq)}" -> "{_escape(concept)}";')

    lines.append("}")
    return "\n".join(lines)


def build_prerequisite_graph_json(
    concept: str, student_memory_profile: dict[str, Any]
) -> dict[str, Any]:
    """
    JSON counterpart to build_prerequisite_graph_dot for the web frontend
    (which draws its own SVG client-side instead of using st.graphviz_chart).
    Scoped to one concept's *direct* prerequisites only — matches the
    reference UI's single-concept graph explorer, not the whole curriculum.
    Same mastery-lookup logic as the DOT version, just reshaped.
    """
    concept_mastery: dict[str, Any] = student_memory_profile.get("concept_mastery", {}) or {}

    def mastery_of(name: str) -> float:
        entry = concept_mastery.get(name)
        return entry.get("mastery", 0.0) if entry else 0.0

    neo4j_available = is_available()
    prereqs = get_prerequisites(concept) if neo4j_available else []

    nodes = [{"name": concept, "mastery": mastery_of(concept), "is_focus": True}]
    edges = []
    for prereq in prereqs:
        nodes.append({"name": prereq, "mastery": mastery_of(prereq), "is_focus": False})
        edges.append({"from": concept, "to": prereq})

    return {"nodes": nodes, "edges": edges, "neo4j_available": neo4j_available}


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    $ python -m backend.memory.graph_viz
    """
    mock_profile = {
        "concept_mastery": {
            "Fraction Addition": {"mastery": 0.72, "consecutive_misses": 0},
            "Comparing Unit Fractions": {"mastery": 0.35, "consecutive_misses": 2},
        },
        "weak_prerequisites": ["Common Denominators"],
    }

    print("=" * 80)
    print("DOT for a mock profile with mixed mastery + a weak prerequisite:")
    dot = build_prerequisite_graph_dot("Fraction Addition", mock_profile)
    print(dot)

    print("\n" + "=" * 80)
    print("DOT for an empty profile (should still show the curriculum graph if Neo4j has one):")
    print(build_prerequisite_graph_dot(None, {}))
    print("=" * 80)
