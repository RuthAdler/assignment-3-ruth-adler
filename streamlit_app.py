"""
Streamlit chat UI for the Bitext data analyst agent (Bonus A).

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import GRAPH_RECURSION_LIMIT, app
from memory import CHECKPOINT_DB, make_graph_config

REASONING_TRUNCATE = 400


def _graph_config(session_id: str, user_id: str) -> Dict[str, Any]:
    return make_graph_config(session_id.strip() or "default", user_id.strip() or "default", GRAPH_RECURSION_LIMIT)


def reasoning_lines_from_event(event: dict) -> List[str]:
    """Turn one LangGraph update event into display lines."""
    lines: List[str] = []
    router_data = event.get("router", {})
    if "query_type" in router_data:
        lines.append(f"[{router_data['query_type']}]")

    for message in event.get("agent", {}).get("messages", []):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                lines.append(f"> {tc['name']}({tc['args']})")

    for message in event.get("tools", {}).get("messages", []):
        if hasattr(message, "name") and message.name:
            content = str(message.content)
            if len(content) > REASONING_TRUNCATE:
                content = content[:REASONING_TRUNCATE] + "..."
            lines.append(f"< {message.name}: {content}")
    return lines


def chat_messages_from_checkpoint(config: dict) -> List[Tuple[str, str]]:
    """Load user/assistant messages from the checkpoint for this session."""
    try:
        snapshot = app.get_state(config)
    except Exception:
        return []

    rows: List[Tuple[str, str]] = []
    for message in snapshot.values.get("messages", []):
        if getattr(message, "type", None) == "human":
            rows.append(("user", message.content))
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            rows.append(("assistant", message.content))
    return rows


def run_agent_turn(user_input: str, config: dict) -> Tuple[str, str]:
    """Run the graph and return (final_answer, reasoning_text)."""
    lines: List[str] = []
    final_answer = ""

    for event in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="updates",
        config=config,
    ):
        lines.extend(reasoning_lines_from_event(event))
        for node in ("agent", "decline", "profile_recall", "recommend"):
            node_messages = event.get(node, {}).get("messages", [])
            if node_messages:
                last = node_messages[-1]
                if isinstance(last, AIMessage) and last.content and not last.tool_calls:
                    final_answer = last.content

    return final_answer or "No response generated.", "\n".join(lines)


def render_sidebar() -> Tuple[str, str]:
    """Sidebar: session switcher and profile user id."""
    st.sidebar.header("Session")
    session_id = st.sidebar.text_input(
        "Session ID",
        value=st.session_state.get("session_id", "default"),
        help="Same ID resumes this conversation from checkpoints/conversations.db",
    )
    user_id = st.sidebar.text_input(
        "User ID (profile)",
        value=st.session_state.get("user_id", session_id),
        help="Distilled profile stored in profiles/<user_id>.json",
    )

    if st.sidebar.button("New session ID"):
        import uuid

        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.user_id = st.session_state.session_id
        st.session_state.pop("last_reasoning", None)
        st.rerun()

    st.session_state.session_id = session_id
    st.session_state.user_id = user_id

    st.sidebar.caption(f"Checkpoints: `{CHECKPOINT_DB}`")
    return session_id, user_id


def main() -> None:
    st.set_page_config(page_title="Bitext Analyst", page_icon="💬", layout="wide")
    st.title("Bitext customer service data analyst")
    st.caption("Ask questions about the Bitext dataset. Reasoning steps appear under each reply.")

    session_id, user_id = render_sidebar()
    config = _graph_config(session_id, user_id)

    history = chat_messages_from_checkpoint(config)
    for idx, (role, content) in enumerate(history):
        with st.chat_message(role):
            if role == "assistant" and idx == len(history) - 1:
                reasoning = st.session_state.get("last_reasoning")
                if reasoning:
                    with st.expander("Reasoning", expanded=True):
                        st.code(reasoning)
            st.markdown(content)

    if prompt := st.chat_input("Ask about categories, intents, refunds, examples..."):
        with st.spinner("Thinking..."):
            answer, reasoning = run_agent_turn(prompt, config)
        st.session_state.last_reasoning = reasoning
        st.rerun()


if __name__ == "__main__":
    main()
