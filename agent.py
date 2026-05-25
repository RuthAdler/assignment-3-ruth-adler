# agent.py
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from memory import get_checkpointer
from profile import (
    format_profile_for_prompt,
    format_profile_for_user,
    is_profile_question,
    load_profile,
    update_profile_from_exchange,
)
from recommender import (
    build_recommendation,
    is_recommendation_confirm,
    is_recommendation_refinement,
    is_recommendation_request,
)
from router import classify_query, is_follow_up
from tools import TOOLS, reset_df

load_dotenv()

# --- State ---
class AgentState(TypedDict):
    """The state that flows through the graph at every step."""
    messages: Annotated[list, add_messages]
    query_type: str
    iterations: int
    user_id: str
    user_profile: str
    pending_suggested_query: Optional[str]
    awaiting_recommendation_confirm: bool
    confirmed_query: Optional[str]


# --- Constants ---
MAX_ITERATIONS = 12
STRUCTURED_HINT = """
This is a structured query: use tools to compute concrete answers (counts, lists, examples, distributions).
Chain tools as needed (e.g. filter_data → count_rows, or filter_data → get_value_counts).
For follow-ups like "show me 3 more", apply the same filter as before and call get_examples with a higher offset.
When the user refers to "the last two" counts, use earlier tool results in the conversation and/or the user profile facts.
"""

UNSTRUCTURED_HINT = """
This is an unstructured query: use get_data_for_summary (it filters by topics in the question).
For one specific category, you may filter_data first, then get_data_for_summary.
After you have sample data, give a final text answer — do not call the same tool again with the same arguments.
"""

SYSTEM_PROMPT_BASE = """You are a data analyst agent for the Bitext customer service dataset.
Columns: instruction (customer message), response (agent reply), category (e.g. ACCOUNT, SHIPPING),
intent (e.g. get_refund, cancel_order). There are no earnings, revenue, or year columns.

On the first step, call a tool. After you receive tool results, either call another tool OR give
a final text answer — do not call tools forever. Call exactly ONE tool per step.
If a tool reports an error or the data cannot answer the question, explain that in your final
answer and stop. Never repeat the same tool with the same arguments.
For refund counts: filter_data(intent, get_refund) then count_rows.
For category distributions: filter_data(category, VALUE) then get_value_counts(intent).

Use the conversation history for follow-up questions (e.g. "3 more", "what about refunds?").
"""


def _system_prompt(
    query_type: str,
    user_profile: str,
    confirmed_query: Optional[str] = None,
) -> str:
    """Build a system prompt tailored to structured vs unstructured queries."""
    base = SYSTEM_PROMPT_BASE
    if user_profile:
        base += f"\n\nUser profile (distilled from past sessions):\n{user_profile}"
    if query_type == "execute_recommendation" and confirmed_query:
        base += (
            f"\n\nThe user confirmed they want you to execute this query:\n{confirmed_query}\n"
            "Use tools now to answer it. Do not ask for confirmation again."
        )
        return base + STRUCTURED_HINT
    if query_type == "unstructured":
        return base + UNSTRUCTURED_HINT
    return base + STRUCTURED_HINT


def _user_id_from_config(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("user_id", "default")


# --- LLM ---
def get_llm() -> ChatOpenAI:
    """Initialize the Nebius LLM with tools bound."""
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url="https://api.studio.nebius.ai/v1/",
        temperature=0,
    )
    return llm.bind_tools(TOOLS, parallel_tool_calls=False)


def _keep_single_tool_call(message: AIMessage) -> AIMessage:
    """Nebius Llama only allows one tool call per model turn."""
    if message.tool_calls and len(message.tool_calls) > 1:
        return message.model_copy(update={"tool_calls": message.tool_calls[:1]})
    return message


def _last_human_message(messages: list) -> Optional[str]:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return message.content
    return None


def _last_agent_reply(messages: list) -> Optional[str]:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return message.content
    return None


# --- Nodes ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Classify the incoming query and set query_type in state."""
    reset_df()
    user_id = _user_id_from_config(config)
    profile = load_profile(user_id)
    last_message = state["messages"][-1].content

    if is_profile_question(last_message):
        return {
            "query_type": "profile_recall",
            "iterations": 0,
            "user_id": user_id,
            "user_profile": format_profile_for_prompt(profile),
            "pending_suggested_query": None,
            "awaiting_recommendation_confirm": False,
            "confirmed_query": None,
        }

    pending = state.get("pending_suggested_query")
    awaiting = state.get("awaiting_recommendation_confirm", False)

    if pending and is_recommendation_confirm(last_message):
        return {
            "query_type": "execute_recommendation",
            "iterations": 0,
            "user_id": user_id,
            "user_profile": format_profile_for_prompt(profile),
            "pending_suggested_query": None,
            "awaiting_recommendation_confirm": False,
            "confirmed_query": pending,
        }

    if is_recommendation_request(last_message):
        return {
            "query_type": "recommend",
            "iterations": 0,
            "user_id": user_id,
            "user_profile": format_profile_for_prompt(profile),
            "pending_suggested_query": None,
            "awaiting_recommendation_confirm": False,
            "confirmed_query": None,
        }

    if awaiting and pending and is_recommendation_refinement(last_message):
        return {
            "query_type": "recommend_refine",
            "iterations": 0,
            "user_id": user_id,
            "user_profile": format_profile_for_prompt(profile),
            "confirmed_query": None,
        }

    prior_turns = sum(1 for m in state["messages"] if getattr(m, "type", None) == "human") > 1
    if is_follow_up(last_message, prior_turns):
        query_type = "structured"
    else:
        query_type = classify_query(last_message)
    updates = {
        "query_type": query_type,
        "iterations": 0,
        "user_id": user_id,
        "user_profile": format_profile_for_prompt(profile),
        "confirmed_query": None,
    }
    if awaiting and pending and query_type in ("structured", "unstructured"):
        updates["pending_suggested_query"] = None
        updates["awaiting_recommendation_confirm"] = False
    return updates


def recommend_node(state: AgentState) -> AgentState:
    """Suggest a follow-up query from history + profile; do not run tools."""
    last_message = state["messages"][-1].content
    query_type = state.get("query_type", "recommend")
    return build_recommendation(
        user_profile=state.get("user_profile", ""),
        messages=state["messages"],
        user_message=last_message,
        pending_query=state.get("pending_suggested_query"),
        is_refinement=query_type == "recommend_refine",
    )


def profile_recall_node(state: AgentState) -> AgentState:
    """Answer from the distilled user profile without calling dataset tools."""
    profile = load_profile(state.get("user_id", "default"))
    return {
        "messages": [AIMessage(content=format_profile_for_user(profile))],
    }


def _has_tool_results(messages: list) -> bool:
    return any(getattr(m, "type", None) == "tool" for m in messages)


def _tool_call_key(tool_call: dict) -> str:
    return f"{tool_call.get('name')}:{tool_call.get('args')}"


def _is_repeated_tool_call(messages: list, tool_calls: list) -> bool:
    if not tool_calls:
        return False
    new_key = _tool_call_key(tool_calls[0])
    count = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tc in message.tool_calls:
                if _tool_call_key(tc) == new_key:
                    count += 1
    return count >= 2


def agent_node(state: AgentState) -> AgentState:
    """LLM reasoning step — thinks and decides which tool to call next."""
    iterations = state.get("iterations", 0) + 1

    if iterations > MAX_ITERATIONS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I've reached the maximum number of steps. Based on what I've found so far, "
                        "I cannot complete this query. Please try rephrasing your question."
                    )
                )
            ],
            "iterations": iterations,
        }

    query_type = state.get("query_type", "structured")
    profile_text = state.get("user_profile", "")
    confirmed = state.get("confirmed_query")
    messages = [
        SystemMessage(content=_system_prompt(query_type, profile_text, confirmed))
    ] + state["messages"]
    llm = get_llm()

    run_tools = query_type in ("structured", "unstructured", "execute_recommendation")
    if run_tools and not _has_tool_results(messages):
        llm = llm.bind(tool_choice="required")

    response = _keep_single_tool_call(llm.invoke(messages))

    if response.tool_calls and _has_tool_results(messages):
        if _is_repeated_tool_call(state["messages"], response.tool_calls):
            response = AIMessage(
                content=(
                    "I've reached the limit of useful tool steps for this question. "
                    "Please try a narrower question about categories, intents, or customer interactions."
                )
            )
        elif iterations >= MAX_ITERATIONS:
            response = AIMessage(
                content=(
                    "I've reached the maximum number of steps. Based on what I've found so far, "
                    "I cannot complete this query. Please try rephrasing your question."
                )
            )

    return {"messages": [response], "iterations": iterations}


def decline_node(state: AgentState) -> AgentState:
    """Return a polite decline for out-of-scope queries."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "I can only answer questions about the Bitext customer service dataset. "
                    "Your question appears to be outside that scope. Please ask me about the "
                    "dataset's categories, intents, or customer interactions."
                )
            )
        ],
    }


def update_profile_node(state: AgentState) -> AgentState:
    """Distill this turn into the per-user profile file."""
    user_id = state.get("user_id", "default")
    user_msg = _last_human_message(state["messages"])
    agent_reply = _last_agent_reply(state["messages"])
    if (
        user_msg
        and agent_reply
        and not is_profile_question(user_msg)
        and not is_recommendation_request(user_msg)
        and not (
            state.get("awaiting_recommendation_confirm")
            and is_recommendation_refinement(user_msg)
        )
    ):
        profile = update_profile_from_exchange(user_id, user_msg, agent_reply)
        return {"user_profile": format_profile_for_prompt(profile)}
    return {}


# --- Edges ---
def route_after_router(state: AgentState) -> str:
    query_type = state.get("query_type", "structured")
    if query_type == "profile_recall":
        return "profile_recall"
    if query_type in ("recommend", "recommend_refine"):
        return "recommend"
    if query_type == "out_of_scope":
        return "decline"
    return "agent"


def route_after_agent(state: AgentState) -> str:
    if state.get("iterations", 0) > MAX_ITERATIONS:
        return "update_profile"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "update_profile"


# --- Build Graph ---
def build_graph():
    """Build and compile the LangGraph ReAct agent graph with SQLite checkpoints."""
    reset_df()

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("profile_recall", profile_recall_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("decline", decline_node)
    graph.add_node("update_profile", update_profile_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "agent": "agent",
            "decline": "decline",
            "profile_recall": "profile_recall",
            "recommend": "recommend",
        },
    )
    graph.add_edge("recommend", "update_profile")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "update_profile": "update_profile",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("decline", "update_profile")
    graph.add_edge("profile_recall", "update_profile")
    graph.add_edge("update_profile", END)

    return graph.compile(checkpointer=get_checkpointer())


app = build_graph()
GRAPH_RECURSION_LIMIT = MAX_ITERATIONS * 3 + 5
