import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import (
    get_regression_versions,
    compare_versions,
    auto_check_version,
    get_regression_reports,
    render_api_docs_button,
)

st.set_page_config(page_title="Regression Analysis", layout="wide")
st.title("Version-Over-Version Regression Analysis")
render_api_docs_button()

# ── Version timeline ───────────────────────────────────────────────────────────
versions = get_regression_versions()

if not versions:
    st.info("No version data yet. Run the seed script or evaluate conversations first.")
    st.stop()

st.subheader("Version Timeline")

df_versions = pd.DataFrame([
    {
        "Version": v.get("version", ""),
        "Conversations": v.get("conversation_count", 0),
        "Evaluations": v.get("eval_count", 0),
        "Avg Score": round(v.get("mean_overall_score", 0.0), 3),
    }
    for v in versions
])

fig_timeline = px.line(
    df_versions,
    x="Version",
    y="Avg Score",
    markers=True,
    title="Average Overall Score per Agent Version",
    labels={"Avg Score": "Mean Overall Score"},
)
fig_timeline.update_traces(marker_size=10, line_width=2)
fig_timeline.update_layout(yaxis_range=[0, 1])
st.plotly_chart(fig_timeline, use_container_width=True)
st.dataframe(df_versions, use_container_width=True)

st.divider()

# ── Manual comparison ─────────────────────────────────────────────────────────
st.subheader("Compare Two Versions")

all_version_strings = [v.get("version", "") for v in versions]

col_b, col_t, col_btn = st.columns([2, 2, 1])
with col_b:
    default_b = all_version_strings[-2] if len(all_version_strings) >= 2 else all_version_strings[0]
    baseline = st.selectbox("Baseline version", all_version_strings, index=all_version_strings.index(default_b))
with col_t:
    default_t = all_version_strings[-1]
    target = st.selectbox("Target version", all_version_strings, index=all_version_strings.index(default_t))
with col_btn:
    st.write("")  # vertical alignment
    st.write("")
    run_compare = st.button("Compare", type="primary")

# Auto-check button
col_auto, _ = st.columns([2, 6])
with col_auto:
    auto_version = st.selectbox("Auto-check version", all_version_strings, index=len(all_version_strings) - 1)
    run_auto = st.button("Auto-check vs previous baseline")

# ── Run comparison ─────────────────────────────────────────────────────────────
report = None

if run_compare:
    if baseline == target:
        st.warning("Baseline and target must be different versions.")
    else:
        with st.spinner(f"Comparing {baseline} → {target}..."):
            report = compare_versions(baseline, target)
        if "error" in report:
            st.error(f"Comparison failed: {report['error']}")
            report = None

if run_auto:
    with st.spinner(f"Auto-checking {auto_version}..."):
        report = auto_check_version(auto_version)
    if report is None or (isinstance(report, dict) and "error" in report):
        st.warning("Auto-check returned no result (not enough data, or no previous version).")
        report = None

def _render_report(report: dict):
    is_regression = report.get("is_regression", False)
    severity = report.get("severity", "none")
    summary = report.get("summary", "")
    b_ver = report.get("baseline_version", "")
    t_ver = report.get("target_version", "")
    regressions = report.get("regressions_detected", [])

    # Severity badge
    SEVERITY_COLOR = {"critical": "error", "major": "error", "minor": "warning", "none": "success"}
    badge_fn = getattr(st, SEVERITY_COLOR.get(severity, "info"))
    badge_fn(f"{'⚠️ REGRESSION DETECTED' if is_regression else '✅ No Regression'} — severity: {severity.upper()}")

    st.markdown(f"**{b_ver} (n={report.get('baseline_sample_size', 0)}) → {t_ver} (n={report.get('target_sample_size', 0)})**")
    st.write(summary)

    # Dimension-by-dimension grouped bar chart
    dims = report.get("dimensions", {})
    if dims:
        rows = []
        for dim_name, dim_data in dims.items():
            rows.append({
                "Dimension": dim_name,
                "Baseline": dim_data.get("baseline_mean", 0),
                "Target": dim_data.get("target_mean", 0),
                "Δ%": dim_data.get("delta_pct", 0),
                "Regression": dim_data.get("is_regression", False),
                "Significance": dim_data.get("significance", ""),
            })
        df_dims = pd.DataFrame(rows)

        # Grouped bar chart
        fig_dims = go.Figure()
        fig_dims.add_trace(go.Bar(name="Baseline", x=df_dims["Dimension"], y=df_dims["Baseline"],
                                  marker_color="steelblue"))
        fig_dims.add_trace(go.Bar(name="Target", x=df_dims["Dimension"], y=df_dims["Target"],
                                  marker_color=[
                                      "crimson" if r else "mediumseagreen"
                                      for r in df_dims["Regression"]
                                  ]))
        fig_dims.update_layout(
            barmode="group",
            title=f"Score Comparison: {b_ver} vs {t_ver}",
            yaxis_range=[0, 1],
            legend_title="Version",
        )
        st.plotly_chart(fig_dims, use_container_width=True)

        # Delta table
        df_delta = df_dims[["Dimension", "Baseline", "Target", "Δ%", "Significance", "Regression"]].copy()
        st.dataframe(df_delta.style.format({"Baseline": "{:.3f}", "Target": "{:.3f}", "Δ%": "{:+.1f}"}),
                     use_container_width=True)

    # Issue rate changes
    issue_changes = report.get("issue_rate_changes", {})
    if issue_changes:
        st.subheader("Issue Rate Changes")
        issue_rows = [
            {
                "Issue Type": itype,
                "Baseline Rate": f"{data.get('baseline_rate', 0):.1%}",
                "Target Rate": f"{data.get('target_rate', 0):.1%}",
                "Change": f"{data.get('change_pct', 0):+.1f}%",
                "Elevated": "⚠️" if data.get("is_elevated") else "✅",
            }
            for itype, data in issue_changes.items()
        ]
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)


if report and isinstance(report, dict) and "dimensions" in report:
    _render_report(report)


st.divider()

# ── Recent regression reports ──────────────────────────────────────────────────
st.subheader("Recent Regression Reports")
reports_data = get_regression_reports(limit=20)
reports = reports_data.get("data", [])

if reports:
    df_reports = pd.DataFrame([
        {
            "Report ID": r.get("report_id", ""),
            "Baseline": r.get("baseline_version", ""),
            "Target": r.get("target_version", ""),
            "Regression": "⚠️ Yes" if r.get("is_regression") else "✅ No",
            "Severity": r.get("severity", "").upper(),
            "Created": r.get("created_at", "")[:19] if r.get("created_at") else "",
            "Summary": r.get("summary", "")[:80],
        }
        for r in reports
    ])
    st.dataframe(df_reports, use_container_width=True)
else:
    st.info("No regression reports yet. Run a comparison above.")
