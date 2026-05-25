"""Bonus B — interactive query recommendations (suggest → refine → confirm → execute)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

_RECOMMEND_PHRASES = (
    "what should i query next",
    "what should i ask next",
    "what should i query",
    "what can i query next",
    "what can i ask next",
    "what to query next",
    "suggest a query",
    "suggest a follow-up",
    "suggest a follow up",
    "recommend a query",
    "query recommendation",
    "what do you suggest i ask",
)

_CONFIRM_EXACT = frozenset(
    {
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "do it",
        "run it",
        "please",
        "execute",
        "yes please",
        "yes, please",
        "yes do it",
        "yes, do it",
    }
)

_REFINE_HINTS = (
    "instead",
    "rather",
    "prefer",
    "change",
    "different",
    "not distribution",
    "examples",
    "summary",
    "count",
    "don't want",
    "do not want",
)


def is_recommendation_request(query: str) -> bool:
    """User wants a suggested next query (not execution yet)."""
    q = query.lower().strip()
    return any(phrase in q for phrase in _RECOMMEND_PHRASES)


def is_recommendation_confirm(query: str) -> bool:
    """User confirmed running the pending suggestion."""
    q = query.lower().strip().strip('"').rstrip(".")
    if q in _CONFIRM_EXACT:
        return True
    prefixes = ("yes,", "yes ", "go ahead", "do it", "please run", "please go", "sure,")
    return any(q.startswith(p) for p in prefixes)


def is_recommendation_refinement(query: str) -> bool:
    """User is adjusting the suggestion before confirming."""
    q = query.lower().strip()
    if is_recommendation_confirm(q) or is_recommendation_request(q):
        return False
    if len(q) > 200:
        return False
    return any(hint in q for hint in _REFINE_HINTS) or (
        len(q) < 100 and ("see " in q or "show " in q or "want " in q)
    )


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url="https://api.studio.nebius.ai/v1/",
        temperature=0,
    )


def _conversation_summary(messages: list, max_turns: int = 6) -> str:
    """Short text summary of recent human/assistant turns for the recommender."""
    lines: List[str] = []
    for message in messages[-max_turns * 2 :]:
        role = getattr(message, "type", None)
        content = (getattr(message, "content", None) or "").strip()
        if not content or role not in ("human", "ai"):
            continue
        if role == "ai" and getattr(message, "tool_calls", None):
            continue
        prefix = "User" if role == "human" else "Assistant"
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines) if lines else "(no prior turns yet)"


def _parse_recommendation_response(text: str) -> Dict[str, str]:
    """Parse LLM JSON {reply, suggested_query}."""
    raw = text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        reply = str(data.get("reply", "")).strip()
        suggested = str(data.get("suggested_query", "")).strip()
        if reply and suggested:
            return {"reply": reply, "suggested_query": suggested}
    except json.JSONDecodeError:
        pass
    return {
        "reply": text.strip() or "I can suggest a follow-up query about the dataset.",
        "suggested_query": "Show 5 examples from the REFUND category.",
    }


def build_recommendation(
    *,
    user_profile: str,
    messages: list,
    user_message: str,
    pending_query: Optional[str] = None,
    is_refinement: bool = False,
) -> Dict[str, Any]:
    """
    Suggest or refine a query using profile + episodic memory.
    Does not call dataset tools.
    """
    history = _conversation_summary(messages)
    mode = "refine" if is_refinement and pending_query else "suggest"

    if mode == "refine":
        task = f"""The user is refining a pending suggestion.
Pending executable query: {pending_query}
User refinement: {user_message}

Update the suggestion to match their preference. End with a clear confirmation question
(e.g. "Should I go ahead?"). Do NOT claim you already ran tools."""
    else:
        task = f"""The user asked for a recommended next query about the Bitext dataset.
User message: {user_message}

Suggest ONE relevant follow-up based on conversation history and profile.
Do NOT execute tools or invent results. End with a confirmation question."""

    prompt = f"""{task}

User profile:
{user_profile}

Recent conversation:
{history}

Return ONLY valid JSON:
{{
  "reply": "natural language for the user (2-4 sentences, ends with Should I go ahead?)",
  "suggested_query": "a single imperative query the agent can run later, e.g. Show 5 examples from the REFUND category."
}}

Rules for suggested_query:
- Must be answerable with dataset tools (filter, count, examples, distribution, summary).
- Use category/intent names from the dataset (REFUND, SHIPPING, get_refund, etc.).
- No financial/year/CRM questions."""

    response = _get_llm().invoke(
        [SystemMessage(content="You output only JSON."), HumanMessage(content=prompt)]
    )
    parsed = _parse_recommendation_response(response.content)
    return {
        "messages": [AIMessage(content=parsed["reply"])],
        "pending_suggested_query": parsed["suggested_query"],
        "awaiting_recommendation_confirm": True,
    }
