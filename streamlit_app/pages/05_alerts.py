import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from utils import get_alerts, get_alert_summary, update_alert_status, render_api_docs_button

st.set_page_config(page_title="Alerts", layout="wide")
st.title("Alerts")
render_api_docs_button()

# ── Summary metrics ────────────────────────────────────────────────────────────
summary = get_alert_summary()
total_open = summary.get("total_open", 0)
by_severity = summary.get("by_severity", {})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Open Alerts", total_open)
with col2:
    st.metric("Critical", by_severity.get("critical", 0))
with col3:
    st.metric("Warning", by_severity.get("warning", 0))
with col4:
    st.metric("Info", by_severity.get("info", 0))

st.divider()

# ── Filters ────────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    status_filter = st.selectbox("Status", ["open", "acknowledged", "resolved", "all"], index=0)
with col_f2:
    type_filter = st.selectbox("Type", ["all", "regression", "quality_drop", "high_failure_rate", "annotator_conflict"])
with col_f3:
    severity_filter = st.selectbox("Severity", ["all", "critical", "warning", "info"])

status_val = None if status_filter == "all" else status_filter
type_val = None if type_filter == "all" else type_filter
severity_val = None if severity_filter == "all" else severity_filter

alerts = get_alerts(status=status_val, alert_type=type_val, severity=severity_val, limit=100)

if not alerts:
    st.info("No alerts match the current filters.")
    st.stop()

# ── Severity colour coding ──────────────────────────────────────────────────────
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
STATUS_ICON = {"open": "🔓", "acknowledged": "👁️", "resolved": "✅"}
TYPE_LABEL = {
    "regression": "Regression",
    "quality_drop": "Quality Drop",
    "high_failure_rate": "High Failure Rate",
    "annotator_conflict": "Annotator Conflict",
}

for alert in alerts:
    sev = alert.get("severity", "info")
    atype = alert.get("type", "")
    status = alert.get("status", "open")
    icon = SEVERITY_ICON.get(sev, "⚪")
    s_icon = STATUS_ICON.get(status, "❓")

    header = f"{icon} [{TYPE_LABEL.get(atype, atype.upper())}] {alert.get('title', '')}  {s_icon} {status}"

    with st.expander(header, expanded=(sev == "critical" and status == "open")):
        st.markdown(f"**Description:**\n\n{alert.get('description', '')}")

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.caption(f"Alert ID: `{alert.get('alert_id', '')}`")
            if alert.get("related_entity_id"):
                st.caption(f"Related: `{alert.get('related_entity_id')}`")
        with col_meta2:
            st.caption(f"Created: {alert.get('created_at', '')[:19]}")
            if alert.get("acknowledged_at"):
                st.caption(f"Acknowledged: {alert['acknowledged_at'][:19]}")
            if alert.get("resolved_at"):
                st.caption(f"Resolved: {alert['resolved_at'][:19]}")

        if status == "open":
            col_ack, col_res, _ = st.columns([1, 1, 4])
            alert_id = alert.get("alert_id", "")
            with col_ack:
                if st.button("Acknowledge", key=f"ack_{alert_id}"):
                    result = update_alert_status(alert_id, "acknowledged")
                    if "error" not in result:
                        st.success("Acknowledged")
                        st.rerun()
                    else:
                        st.error(result["error"])
            with col_res:
                if st.button("Resolve", key=f"res_{alert_id}"):
                    result = update_alert_status(alert_id, "resolved")
                    if "error" not in result:
                        st.success("Resolved")
                        st.rerun()
                    else:
                        st.error(result["error"])
        elif status == "acknowledged":
            alert_id = alert.get("alert_id", "")
            if st.button("Mark Resolved", key=f"res2_{alert_id}"):
                result = update_alert_status(alert_id, "resolved")
                if "error" not in result:
                    st.success("Resolved")
                    st.rerun()
