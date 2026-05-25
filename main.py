# main.py
import argparse

from langchain_core.messages import HumanMessage

from agent import GRAPH_RECURSION_LIMIT, app
from memory import make_graph_config


def print_reasoning(event: dict) -> None:
    """Print tool calls and observations from one graph step."""
    router_data = event.get("router", {})
    if "query_type" in router_data:
        print(f"  [{router_data['query_type']}]")

    for message in event.get("agent", {}).get("messages", []):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"  > {tc['name']}({tc['args']})")

    for message in event.get("tools", {}).get("messages", []):
        if hasattr(message, "name") and message.name:
            content = str(message.content)
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  < {message.name}: {content}")


def run_turn(user_input: str, config: dict) -> None:
    """Run one user message through the graph (checkpoint restores prior turns)."""
    print()
    events = []
    for event in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="updates",
        config=config,
    ):
        events.append(event)
        print_reasoning(event)

    if not events:
        return

    final = events[-1]
    last_messages = (
        final.get("agent", {}).get("messages", [])
        or final.get("recommend", {}).get("messages", [])
        or final.get("decline", {}).get("messages", [])
        or final.get("profile_recall", {}).get("messages", [])
    )
    if last_messages:
        print(f"\n{last_messages[-1].content}\n")


def run_cli(session_id: str, user_id: str) -> None:
    """Run the interactive CLI loop with persistent session memory."""
    config = make_graph_config(session_id, user_id, GRAPH_RECURSION_LIMIT)

    print("Bitext customer service data analyst")
    print(f"Session: {session_id}  |  User profile: {user_id}")
    print("Conversation is saved across restarts for this session.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        run_turn(user_input, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitext customer service data analyst")
    parser.add_argument(
        "--session",
        default="default",
        help="Session/thread ID for conversation checkpoints (default: default)",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User ID for long-term profile (default: same as --session)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(session_id=args.session, user_id=args.user or args.session)
