import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import get_meta_drift, get_meta_calibration, get_meta_correlation, render_api_docs_button

st.set_page_config(page_title="Meta-Evaluation", layout="wide")
st.title("Meta-Evaluation")
render_api_docs_button()

# ── Eval-to-User-Rating Correlation ───────────────────────────────────────────
st.header("Eval Score ↔ User Rating Correlation")

corr_data = get_meta_correlation()
correlations = corr_data.get("correlations", [])
scatter_data = corr_data.get("scatter_data", [])
best_dim = corr_data.get("best_dimension")

if correlations:
    # Correlation summary metrics
    corr_cols = st.columns(len(correlations))
    for i, c in enumerate(correlations):
        r = c.get("pearson_r", 0)
        with corr_cols[i]:
            st.metric(
                label=c.get("dimension", "").replace("_", " ").title(),
                value=f"r = {r:.2f}",
                help=c.get("interpretation", ""),
                delta=c.get("interpretation", ""),
                delta_color="off",
            )

    # Scatter plots per dimension
    if scatter_data:
        df_scatter = pd.DataFrame(scatter_data)

        dims = ["overall", "response_quality", "tool_accuracy", "coherence"]
        available_dims = [d for d in dims if d in df_scatter.columns]

        best_tab, *other_tabs = st.tabs(
            [f"{'★ ' if d == best_dim else ''}{d.replace('_', ' ').title()}" for d in available_dims]
        )
        all_tabs = [best_tab] + other_tabs

        for tab, dim in zip(all_tabs, available_dims):
            with tab:
                r_val = next((c.get("pearson_r", 0) for c in correlations if c.get("dimension") == dim), 0)
                interp = next((c.get("interpretation", "") for c in correlations if c.get("dimension") == dim), "")

                fig = px.scatter(
                    df_scatter,
                    x=dim,
                    y="user_rating",
                    hover_data=["conversation_id"],
                    labels={dim: f"Auto Score ({dim.replace('_', ' ').title()})", "user_rating": "User Rating (1-5)"},
                    title=f"{dim.replace('_', ' ').title()} vs User Rating  (r = {r_val:.2f}, {interp})",
                    trendline="ols",
                    trendline_color_override="red",
                )
                fig.update_traces(marker_size=7, opacity=0.7)
                fig.update_layout(yaxis=dict(tickvals=[1, 2, 3, 4, 5]))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Pearson r = **{r_val:.3f}** | n = {len(df_scatter)} samples | interpretation: {interp}")
else:
    st.info(
        "No correlation data yet. Evaluations must have associated user_rating feedback. "
        "Run `python seed_data.py` to populate demo data."
    )

st.divider()

# ── Drift analysis ─────────────────────────────────────────────────────────────
st.header("Evaluator Drift")
drift_data = get_meta_drift()

if isinstance(drift_data, list) and drift_data:
    records = []
    for d in drift_data:
        if isinstance(d, dict) and d.get("accuracy") is not None:
            records.append({
                "Evaluator": d.get("evaluator", "?"),
                "Accuracy": d.get("accuracy", 0),
                "Precision": d.get("precision", 0),
                "Recall": d.get("recall", 0),
                "Trend": d.get("trend", "?"),
                "Records": d.get("record_count", 0),
            })
    if records:
        df_drift = pd.DataFrame(records)
        fig = px.bar(
            df_drift.melt(id_vars="Evaluator", value_vars=["Accuracy", "Precision", "Recall"]),
            x="Evaluator", y="value", color="variable", barmode="group",
            title="Evaluator Accuracy / Precision / Recall",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_drift, use_container_width=True)

        st.subheader("Recommendations")
        for d in drift_data:
            if isinstance(d, dict):
                trend_icon = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(d.get("trend", ""), "❓")
                st.write(f"{trend_icon} **{d.get('evaluator')}**: {d.get('recommendation', 'n/a')}")
                blind_spots = d.get("blind_spots", [])
                if blind_spots:
                    st.caption(f"Blind spots: {', '.join(blind_spots)}")
    else:
        st.info("No drift records with sufficient data yet.")
else:
    st.info("No meta-eval records. Submit calibration requests via POST /api/v1/meta/calibrate.")

# ── Calibration summary ────────────────────────────────────────────────────────
st.header("Calibration Summary")
cal_data = get_meta_calibration()

if isinstance(cal_data, dict) and cal_data.get("summary"):
    summary = cal_data["summary"]
    rows = []
    for evaluator, stats in summary.items():
        rows.append({
            "Evaluator": evaluator,
            "Records": stats.get("record_count", 0),
            "Avg Diff": stats.get("avg_diff", 0),
            "Alignment Rate": stats.get("alignment_rate", 0),
        })
    df_cal = pd.DataFrame(rows)
    st.dataframe(df_cal, use_container_width=True)
    st.caption(f"Total calibration records: {cal_data.get('total_records', 0)}")
else:
    st.info("No calibration records yet. Use POST /api/v1/meta/calibrate to add records.")
