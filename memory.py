"""Persistent conversation checkpoints via SQLite."""

import sqlite3
from pathlib import Path
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DB = CHECKPOINT_DIR / "conversations.db"


def get_checkpointer() -> SqliteSaver:
    """Return a SQLite-backed checkpointer that survives process restarts."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    return SqliteSaver(conn)


def make_graph_config(session_id: str, user_id: str, recursion_limit: int) -> Dict[str, Any]:
    """Build the RunnableConfig for a session/thread and user profile."""
    return {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        },
        "recursion_limit": recursion_limit,
    }
