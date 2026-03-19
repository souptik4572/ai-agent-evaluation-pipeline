import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_suggestions, update_suggestion_status, generate_suggestions, get_suggestions_summary

st.set_page_config(page_title="Suggestions", layout="wide")
st.title("Improvement Suggestions")

# ── Actions ────────────────────────────────────────────────────────────────────
col_gen, col_filt1, col_filt2 = st.columns([2, 2, 2])
with col_gen:
    if st.button("Generate Suggestions", type="primary"):
        with st.spinner("Detecting patterns and generating suggestions..."):
            result = generate_suggestions()
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Generated {result.get('suggestions_generated', 0)} suggestions from {result.get('patterns_found', 0)} patterns")
            st.rerun()

with col_filt1:
    status_filter = st.selectbox("Filter by status", ["all", "pending", "accepted", "rejected", "implemented"])
with col_filt2:
    type_filter = st.selectbox("Filter by type", ["all", "prompt", "tool"])

status = None if status_filter == "all" else status_filter
stype = None if type_filter == "all" else type_filter

# ── Summary metrics ────────────────────────────────────────────────────────────
sugg_summary = get_suggestions_summary()
by_status = sugg_summary.get("by_status", {})
by_type = sugg_summary.get("by_type", {})

if by_status:
    cols = st.columns(len(by_status) + 1)
    with cols[0]:
        st.metric("Total", sum(by_status.values()))
    for i, (s, cnt) in enumerate(by_status.items(), 1):
        emoji = {"pending": "⏳", "accepted": "✅", "rejected": "❌", "implemented": "🚀"}.get(s, "")
        with cols[i]:
            st.metric(f"{emoji} {s.capitalize()}", cnt)

st.divider()

# ── Suggestion list ────────────────────────────────────────────────────────────
sugg_data = get_suggestions(status=status, type=stype, limit=200)
suggestions = sugg_data.get("data", []) if isinstance(sugg_data, dict) else []

if not suggestions:
    st.info("No suggestions found. Click 'Generate Suggestions' to create some.")
    st.stop()

# Sort by confidence descending
suggestions = sorted(suggestions, key=lambda x: -x.get("confidence", 0))

# ── Group by type (pattern grouping) ──────────────────────────────────────────
prompt_suggs = [s for s in suggestions if s.get("type") == "prompt"]
tool_suggs = [s for s in suggestions if s.get("type") == "tool"]

def _render_suggestion_group(items: list, label: str, color: str) -> None:
    if not items:
        return
    st.subheader(f"{color} {label} Suggestions ({len(items)})")

    for sugg in items:
        conf = sugg.get("confidence", 0)
        status_val = sugg.get("status", "pending")
        status_emoji = {"pending": "⏳", "accepted": "✅", "rejected": "❌", "implemented": "🚀"}.get(status_val, "❓")
        target = sugg.get("target", "n/a")

        with st.expander(
            f"[{sugg.get('type','?').upper()}] {target} — conf={conf:.2f} {status_emoji} {status_val}"
        ):
            # Confidence progress bar
            st.markdown("**Confidence**")
            st.progress(min(int(conf * 100), 100), text=f"{conf:.0%}")

            st.markdown(f"**Suggestion:**\n{sugg.get('suggestion', '')}")
            st.markdown(f"**Rationale:** {sugg.get('rationale', '')}")
            if sugg.get("expected_impact"):
                st.markdown(f"**Expected Impact:** {sugg.get('expected_impact')}")

            # Frequency / pattern info if available
            if sugg.get("pattern_frequency"):
                st.caption(f"Pattern frequency: {sugg['pattern_frequency']} occurrences")
            if sugg.get("affected_conversations"):
                convs = sugg["affected_conversations"]
                st.caption(f"Triggered by {len(convs)} conversation(s): {', '.join(convs[:5])}")

            # Action buttons for pending
            if status_val == "pending":
                col_a, col_r, _ = st.columns([1, 1, 6])
                sugg_id = sugg.get("suggestion_id", "")
                with col_a:
                    if st.button("Accept", key=f"accept_{sugg_id}"):
                        update_suggestion_status(sugg_id, "accepted")
                        st.rerun()
                with col_r:
                    if st.button("Reject", key=f"reject_{sugg_id}"):
                        update_suggestion_status(sugg_id, "rejected")
                        st.rerun()
            elif status_val == "accepted":
                sugg_id = sugg.get("suggestion_id", "")
                if st.button("Mark Implemented", key=f"impl_{sugg_id}"):
                    update_suggestion_status(sugg_id, "implemented")
                    st.rerun()


_render_suggestion_group(prompt_suggs, "Prompt", "🔵")
_render_suggestion_group(tool_suggs, "Tool", "🟠")

# ── Confidence distribution chart ─────────────────────────────────────────────
if suggestions:
    st.divider()
    st.subheader("Confidence Distribution")
    df_conf = pd.DataFrame([
        {"Confidence": s.get("confidence", 0), "Type": s.get("type", "?"), "Status": s.get("status", "?")}
        for s in suggestions
    ])
    fig = px.histogram(df_conf, x="Confidence", color="Type", nbins=20,
                       title="Distribution of Suggestion Confidence Scores", barmode="overlay")
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
