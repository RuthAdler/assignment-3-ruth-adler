"""Per-user distilled profile storage (separate from conversation checkpoints)."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

PROFILES_DIR = Path("profiles")

EMPTY_PROFILE: Dict[str, Any] = {
    "name": None,
    "topics": [],
    "facts": [],
}


def _profile_path(user_id: str) -> Path:
    safe_id = re.sub(r"[^\w\-]", "_", user_id)
    return PROFILES_DIR / f"{safe_id}.json"


def load_profile(user_id: str) -> Dict[str, Any]:
    """Load a user profile from disk, or return an empty profile."""
    path = _profile_path(user_id)
    if not path.exists():
        return dict(EMPTY_PROFILE)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "name": data.get("name"),
        "topics": data.get("topics", [])[:10],
        "facts": data.get("facts", [])[:15],
    }


def save_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Persist a user profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def format_profile_for_prompt(profile: Dict[str, Any]) -> str:
    """Format profile as text for the agent system prompt."""
    if profile == EMPTY_PROFILE or (
        not profile.get("name") and not profile.get("topics") and not profile.get("facts")
    ):
        return "No stored profile yet for this user."
    lines: List[str] = []
    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    if profile.get("topics"):
        lines.append(f"Topics they ask about: {', '.join(profile['topics'])}")
    if profile.get("facts"):
        lines.append("Facts from past sessions:")
        for fact in profile["facts"]:
            lines.append(f"- {fact}")
    return "\n".join(lines)


def format_profile_for_user(profile: Dict[str, Any]) -> str:
    """Format profile as a natural answer to 'what do you remember about me?'"""
    if not profile.get("name") and not profile.get("topics") and not profile.get("facts"):
        return (
            "I don't have anything saved about you yet. "
            "As we talk about the Bitext dataset, I'll remember your name and topics you care about."
        )
    parts: List[str] = []
    if profile.get("name"):
        parts.append(f"Your name is {profile['name']}.")
    if profile.get("topics"):
        parts.append(f"You often ask about: {', '.join(profile['topics'])}.")
    if profile.get("facts"):
        parts.append("I also recall:")
        parts.extend(f"- {fact}" for fact in profile["facts"])
    return "\n".join(parts)


def is_profile_question(query: str) -> bool:
    """Return True if the user is asking what the agent remembers about them."""
    q = query.lower()
    phrases = (
        "what do you remember about me",
        "what do you know about me",
        "do you remember me",
        "what have you learned about me",
    )
    return any(p in q for p in phrases)


def _get_profile_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url="https://api.studio.nebius.ai/v1/",
        temperature=0,
    )


def update_profile_from_exchange(
    user_id: str,
    user_message: str,
    agent_reply: str,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Distill new facts from the latest turn and merge into the profile."""
    existing = existing or load_profile(user_id)
    prompt = f"""You maintain a short user profile for a Bitext dataset analyst chatbot.
Update the profile using ONLY the latest exchange. Do not copy full chat messages.

Current profile JSON:
{json.dumps(existing)}

User said: {user_message}
Agent replied: {agent_reply[:1500]}

Return ONLY valid JSON with keys:
- name (string or null)
- topics (list of up to 10 short strings: categories/intents they care about)
- facts (list of up to 15 short strings: e.g. "Asked about 997 refund requests", "Interested in SHIPPING examples")

Keep useful existing entries; add new ones; drop duplicates."""
    response = _get_profile_llm().invoke(
        [SystemMessage(content="You output only JSON."), HumanMessage(content=prompt)]
    )
    text = response.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        updated = json.loads(text)
        profile = {
            "name": updated.get("name") or existing.get("name"),
            "topics": list(updated.get("topics", existing.get("topics", [])))[:10],
            "facts": list(updated.get("facts", existing.get("facts", [])))[:15],
        }
    except json.JSONDecodeError:
        profile = existing
    save_profile(user_id, profile)
    return profile
