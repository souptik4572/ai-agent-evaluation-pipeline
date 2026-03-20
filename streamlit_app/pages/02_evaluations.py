import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import get_evaluations, get_evaluation

st.set_page_config(page_title="Evaluations", layout="wide")
st.title("Evaluations")

min_score = st.slider("Filter by minimum score", 0.0, 1.0, 0.0, 0.05)
evals_data = get_evaluations(limit=200, min_score=min_score if min_score > 0 else None)
evaluations = evals_data.get("data", []) if isinstance(evals_data, dict) else []

if not evaluations:
    st.info("No evaluations found. Run `python seed_data.py` to generate sample data.")
    st.stop()

df = pd.DataFrame([
    {
        "evaluation_id": e.get("evaluation_id", ""),
        "conversation_id": e.get("conversation_id", ""),
        "agent_version": e.get("agent_version", ""),
        "overall": e.get("scores", {}).get("overall", 0),
        "response_quality": e.get("scores", {}).get("response_quality", 0),
        "tool_accuracy": e.get("scores", {}).get("tool_accuracy", 0),
        "coherence": e.get("scores", {}).get("coherence", 0),
        "issues_count": len(e.get("issues_detected", [])),
        "routing": e.get("routing_decision", {}).get("routing_decision", "n/a") if e.get("routing_decision") else "n/a",
    }
    for e in evaluations if isinstance(e, dict)
])

version_scores = df.groupby("agent_version")["overall"].mean().reset_index()
fig2 = px.bar(version_scores, x="agent_version", y="overall",
              title="Average Score by Agent Version", color="agent_version")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("All Evaluations")
st.dataframe(df, use_container_width=True)

# Drill-down
st.subheader("Drill Down")
eval_id = st.selectbox("Select an evaluation ID", df["evaluation_id"].tolist())
if eval_id:
    ev = get_evaluation(eval_id)
    if "error" in ev:
        st.error(ev["error"])
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.json(ev.get("scores", {}))
        with col2:
            issues = ev.get("issues_detected", [])
            if issues:
                st.warning(f"{len(issues)} issues")
                for i in issues:
                    st.write(f"- **[{i.get('severity')}]** {i.get('type')}: {i.get('description')}")
            else:
                st.success("No issues detected.")

        routing = ev.get("routing_decision") or {}
        if routing:
            st.info(f"Routing: **{routing.get('routing_decision')}** — {routing.get('reason')}")
