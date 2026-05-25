"""End-to-end smoke test for the Bitext data analyst agent.

Exercises the router, the ReAct loop, multi-step reasoning, the out-of-scope
decline path, episodic memory across turns, the profile node, and the FastMCP
server. Requires NEBIUS_API_KEY in the environment.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def header(name: str) -> None:
    print(f"\n=== {name} ===")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"PASS: {msg}")


def collect_final_answer(events: List[dict]) -> str:
    for event in reversed(events):
        for node in ("agent", "decline", "profile_recall", "recommend"):
            messages = event.get(node, {}).get("messages", [])
            for message in reversed(messages):
                if (
                    getattr(message, "content", None)
                    and not getattr(message, "tool_calls", None)
                    and getattr(message, "type", None) == "ai"
                ):
                    return message.content
    return ""


def tool_names_used(events: List[dict]) -> List[str]:
    names: List[str] = []
    for event in events:
        for message in event.get("agent", {}).get("messages", []):
            for tool_call in getattr(message, "tool_calls", None) or []:
                names.append(tool_call["name"])
    return names


def router_types_seen(events: List[dict]) -> List[str]:
    return [
        event["router"]["query_type"]
        for event in events
        if "router" in event and "query_type" in event["router"]
    ]


def run_turn(app, user_input: str, config: dict):
    from langchain_core.messages import HumanMessage

    events = []
    for event in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="updates",
        config=config,
    ):
        events.append(event)
    return events


def main() -> None:
    if not os.getenv("NEBIUS_API_KEY"):
        fail("NEBIUS_API_KEY not set")

    # Use a temp checkpoint DB and profile dir so the CI run is hermetic.
    tmp = Path(tempfile.mkdtemp(prefix="bitext_smoke_"))
    os.environ["BITEXT_CHECKPOINT_DIR"] = str(tmp / "checkpoints")
    os.environ["BITEXT_PROFILES_DIR"] = str(tmp / "profiles")

    # Patch memory + profile module paths before importing the graph.
    import memory
    import profile as profile_mod

    memory.CHECKPOINT_DIR = tmp / "checkpoints"
    memory.CHECKPOINT_DB = memory.CHECKPOINT_DIR / "conversations.db"
    profile_mod.PROFILES_DIR = tmp / "profiles"

    from agent import GRAPH_RECURSION_LIMIT, app
    from memory import make_graph_config

    session = f"smoke_{uuid.uuid4().hex[:6]}"
    user_id = f"smoke_user_{uuid.uuid4().hex[:6]}"
    config = make_graph_config(session, user_id, GRAPH_RECURSION_LIMIT)

    header("Structured: refund count (multi-step)")
    events = run_turn(app, "How many refund requests did we get?", config)
    answer = collect_final_answer(events)
    used = tool_names_used(events)
    print("router:", router_types_seen(events))
    print("tools called:", used)
    print("answer:", answer)
    if "structured" not in router_types_seen(events):
        fail("router did not classify refund query as structured")
    if "filter_data" not in used or "count_rows" not in used:
        fail("refund query did not chain filter_data and count_rows")
    if "997" not in answer:
        fail(f"expected 997 in answer, got: {answer!r}")
    ok("structured refund query: filter_data → count_rows → 997")

    header("Structured: categories list")
    events = run_turn(app, "What categories exist in the dataset?", config)
    answer = collect_final_answer(events)
    used = tool_names_used(events)
    print("tools called:", used)
    print("answer (head):", answer[:200])
    if "get_unique_values" not in used:
        fail("categories query did not call get_unique_values")
    if not any(category in answer.upper() for category in ("REFUND", "ACCOUNT", "SHIPPING")):
        fail(f"categories answer missing expected categories: {answer!r}")
    ok("structured categories query used get_unique_values")

    header("Out-of-scope: declined politely")
    events = run_turn(app, "Who won the 2024 Champions League?", config)
    answer = collect_final_answer(events)
    used = tool_names_used(events)
    types = router_types_seen(events)
    print("router:", types)
    print("tools called:", used)
    print("answer:", answer)
    if "out_of_scope" not in types:
        fail("out-of-scope query was not classified as out_of_scope")
    if used:
        fail(f"out-of-scope query called tools: {used}")
    ok("out-of-scope question routed to decline (no tools)")

    header("Episodic memory: examples + 'show me 3 more'")
    events = run_turn(app, "Show me 3 examples from the REFUND category", config)
    first_used = tool_names_used(events)
    print("first turn tools:", first_used)
    events = run_turn(app, "Show me 3 more", config)
    second_used = tool_names_used(events)
    print("follow-up tools:", second_used)
    if "get_examples" not in second_used:
        fail("'show me 3 more' did not call get_examples")
    used_offset = False
    for event in events:
        for message in event.get("agent", {}).get("messages", []):
            for tool_call in getattr(message, "tool_calls", None) or []:
                if tool_call["name"] == "get_examples" and tool_call["args"].get("offset", 0):
                    used_offset = True
    if not used_offset:
        print("WARN: follow-up did not pass a non-zero offset to get_examples")
    else:
        ok("'show me 3 more' used offset on get_examples")

    header("Profile: name + recall")
    events = run_turn(app, "My name is Smoketest and I care about refunds.", config)
    events = run_turn(app, "What do you remember about me?", config)
    answer = collect_final_answer(events)
    print("recall answer:", answer)
    if "smoketest" not in answer.lower():
        fail(f"profile recall did not include the name: {answer!r}")
    profile_file = profile_mod.PROFILES_DIR / f"{user_id}.json"
    if not profile_file.exists():
        fail(f"profile file not written: {profile_file}")
    ok(f"profile persisted at {profile_file}")

    header("FastMCP: filter_data + count_rows")

    async def mcp_call() -> int:
        from fastmcp import Client

        async with Client(str(REPO_ROOT / "mcp_server.py")) as client:
            tools = await client.list_tools()
            print("MCP tools:", [tool.name for tool in tools])
            if len(tools) < 3:
                fail(f"MCP server exposes <3 tools: {tools}")
            await client.call_tool("filter_data", {"column": "intent", "value": "get_refund"})
            result = await client.call_tool("count_rows", {})
            return int(result.data)

    refund_count = asyncio.run(mcp_call())
    print("MCP count_rows:", refund_count)
    if refund_count != 997:
        fail(f"MCP count_rows expected 997, got {refund_count}")
    ok("MCP server returned 997 refund rows")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
