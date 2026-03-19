import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from utils import get_conversations, get_conversation, evaluate_conversation

st.set_page_config(page_title="Conversations", layout="wide")
st.title("Conversations")

# List
conv_data = get_conversations(limit=100)
conversations = conv_data.get("data", []) if isinstance(conv_data, dict) else []

if not conversations:
    st.info("No conversations yet. Run `python seed_data.py` to generate sample data.")
    st.stop()

conv_ids = [c.get("conversation_id", "") for c in conversations]
selected_id = st.selectbox("Select a conversation", conv_ids)

if selected_id:
    conv = get_conversation(selected_id)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Conversation: {selected_id}")
        st.caption(f"Agent version: {conv.get('agent_version', 'n/a')} | Created: {conv.get('created_at', 'n/a')}")

        for turn in conv.get("turns", []):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            turn_id = turn.get("turn_id", "?")
            if role == "user":
                with st.chat_message("user"):
                    st.write(f"**Turn {turn_id}:** {content}")
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.write(f"**Turn {turn_id}:** {content}")
                    for tc in (turn.get("tool_calls") or []):
                        with st.expander(f"Tool call: `{tc.get('tool_name')}`"):
                            st.json(tc)
            else:
                st.info(f"[{role.upper()}] {content}")

    with col2:
        st.subheader("Feedback")
        feedback = conv.get("feedback") or {}
        st.metric("User Rating", f"{feedback.get('user_rating', 'n/a')}/5")
        ops = feedback.get("ops_review") or {}
        st.metric("Ops Quality", ops.get("quality", "n/a"))

        anns = feedback.get("annotations") or []
        if anns:
            st.write("**Annotations:**")
            for ann in anns:
                st.write(f"- [{ann.get('type')}] `{ann.get('label')}` by {ann.get('annotator_id')} (conf={ann.get('confidence', '?')})")

        st.subheader("Metadata")
        meta = conv.get("metadata") or {}
        st.metric("Latency (ms)", meta.get("total_latency_ms", "n/a"))
        st.metric("Mission Completed", str(meta.get("mission_completed", "n/a")))

        if st.button("Evaluate this conversation"):
            with st.spinner("Running evaluation..."):
                result = evaluate_conversation(selected_id)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Evaluation complete! ID: {result.get('evaluation_id')}")
                st.metric("Overall Score", result.get("scores", {}).get("overall", "n/a"))
                if result.get("issues_detected"):
                    st.warning(f"{len(result['issues_detected'])} issues detected")
                    for issue in result["issues_detected"]:
                        st.write(f"- [{issue.get('severity','?')}] {issue.get('description','')}")
