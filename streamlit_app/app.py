import streamlit as st
import plotly.express as px
import pandas as pd

from utils import (
    get_conversations,
    get_evaluations,
    get_suggestions_summary,
    get_meta_drift,
    get_alert_summary,
    get_regression_versions,
    get_meta_correlation,
    get_metrics,
    health_check,
    render_api_docs_button,
)

st.set_page_config(
    page_title="AI Agent Evaluation Pipeline",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Agent Evaluation Pipeline")
render_api_docs_button()

# ── Health check banner ────────────────────────────────────────────────────────
health = health_check()
if health.get("status") == "ok":
    st.success(f"API is online (v{health.get('version', '?')})")
else:
    st.error(f"API unreachable: {health.get('error', 'unknown error')}")

# ── ALERT BANNER ───────────────────────────────────────────────────────────────
alert_summary = get_alert_summary()
total_open = alert_summary.get("total_open", 0)
by_severity = alert_summary.get("by_severity", {})

if total_open > 0:
    critical_count = by_severity.get("critical", 0)
    warning_count = by_severity.get("warning", 0)
    parts = []
    if critical_count:
        parts.append(f"{critical_count} critical")
    if warning_count:
        parts.append(f"{warning_count} warning")
    info_count = by_severity.get("info", 0)
    if info_count:
        parts.append(f"{info_count} info")

    severity_label = ", ".join(parts)
    banner_fn = st.error if critical_count > 0 else st.warning
    banner_fn(
        f"⚠️ {total_open} open alert{'s' if total_open > 1 else ''}: {severity_label}  "
        f"→ [View Alerts](./05_alerts)"
    )

# ── VERSION COMPARISON CARD ────────────────────────────────────────────────────
st.header("Overview")

versions = get_regression_versions()
if len(versions) >= 2:
    st.subheader("Version Comparison")
    # Use last two versions by semver
    baseline_v = versions[-2]
    target_v = versions[-1]

    b_score = baseline_v.get("mean_overall_score", 0.0)
    t_score = target_v.get("mean_overall_score", 0.0)
    delta = t_score - b_score
    delta_pct = (delta / b_score * 100) if b_score > 0 else 0.0

    col_b, col_t, col_d = st.columns(3)
    with col_b:
        st.metric(
            label=f"{baseline_v['version']} (baseline)",
            value=f"{b_score:.3f}",
            help=f"n={baseline_v.get('eval_count', 0)} evaluations",
        )
    with col_t:
        st.metric(
            label=f"{target_v['version']} (latest)",
            value=f"{t_score:.3f}",
            delta=f"{delta_pct:+.1f}%",
            delta_color="normal",
            help=f"n={target_v.get('eval_count', 0)} evaluations",
        )
    with col_d:
        regression_detected = delta_pct < -5
        if regression_detected:
            st.error(f"📉 Regression: {delta_pct:.1f}%")
        elif delta_pct < 0:
            st.warning(f"↓ Minor drop: {delta_pct:.1f}%")
        else:
            st.success(f"✅ Stable / improved: {delta_pct:+.1f}%")

# ── CORRELATION CARD ───────────────────────────────────────────────────────────
corr_data = get_meta_correlation()
correlations = corr_data.get("correlations", [])
best_dim = corr_data.get("best_dimension")
if correlations:
    best = next((c for c in correlations if c.get("dimension") == best_dim), None)
    if best:
        r = best.get("pearson_r", 0)
        interp = best.get("interpretation", "")
        n = best.get("sample_size", 0)
        st.info(
            f"📊 Auto-eval correlation with user ratings: "
            f"**r = {r:.2f}** ({interp}) on {n} samples  "
            f"— best dimension: `{best_dim}`  → [See scatter plot](./04_meta_eval)"
        )

# ── Key Metrics ────────────────────────────────────────────────────────────────
conv_data = get_conversations(limit=200)
evals_data = get_evaluations(limit=200)
sugg_summary = get_suggestions_summary()
metrics = get_metrics()

conversations = conv_data.get("data", []) if isinstance(conv_data, dict) else []
evaluations = evals_data.get("data", []) if isinstance(evals_data, dict) else []

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Conversations", metrics.get("total_conversations", conv_data.get("meta", {}).get("count", len(conversations))))
with col2:
    scores = [e.get("scores", {}).get("overall", 0) for e in evaluations if isinstance(e, dict)]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    st.metric("Avg Evaluation Score", avg_score)
with col3:
    total_issues = sum(len(e.get("issues_detected", [])) for e in evaluations if isinstance(e, dict))
    st.metric("Total Issues Detected", total_issues)
with col4:
    pending_suggestions = sugg_summary.get("by_status", {}).get("pending", 0)
    st.metric("Pending Suggestions", pending_suggestions)
with col5:
    st.metric("Open Alerts", total_open, delta=None if total_open == 0 else f"{total_open} open")

# ── Score Distribution ─────────────────────────────────────────────────────────
if scores:
    st.subheader("Score Distribution")
    df_scores = pd.DataFrame({"Overall Score": scores})
    fig = px.histogram(df_scores, x="Overall Score", nbins=20, title="Distribution of Overall Evaluation Scores")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

# ── Recent Evaluations ─────────────────────────────────────────────────────────
st.subheader("Recent Evaluations")
if evaluations:
    df_evals = pd.DataFrame([
        {
            "Evaluation ID": e.get("evaluation_id", ""),
            "Conversation ID": e.get("conversation_id", ""),
            "Agent Version": e.get("agent_version", ""),
            "Overall Score": e.get("scores", {}).get("overall", 0),
            "Tool Accuracy": e.get("scores", {}).get("tool_accuracy", 0),
            "Coherence": e.get("scores", {}).get("coherence", 0),
            "Issues": len(e.get("issues_detected", [])),
            "Routing": e.get("routing_decision", {}).get("routing_decision", "n/a") if e.get("routing_decision") else "n/a",
        }
        for e in evaluations[:20]
        if isinstance(e, dict)
    ])
    st.dataframe(df_evals, use_container_width=True)
else:
    st.info("No evaluations yet. Run the seed script or evaluate conversations via the API.")

# ── Top Pending Suggestions ────────────────────────────────────────────────────
st.subheader("Top Pending Suggestions")
top_pending = sugg_summary.get("top_pending", [])
if top_pending:
    for sugg in top_pending:
        with st.expander(f"[{sugg.get('type','?').upper()}] {sugg.get('target','n/a')} — confidence={sugg.get('confidence',0):.2f}"):
            st.markdown(f"**Suggestion:** {sugg.get('suggestion','')}")
            st.markdown(f"**Rationale:** {sugg.get('rationale','')}")
            st.markdown(f"**Expected Impact:** {sugg.get('expected_impact','n/a')}")
else:
    st.info("No pending suggestions. Click 'Generate Suggestions' in the Suggestions page.")
