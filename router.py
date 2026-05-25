# router.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm() -> ChatOpenAI:
    """Initialize the Nebius LLM via OpenAI-compatible API."""
    return ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url="https://api.studio.nebius.ai/v1/",
        temperature=0
    )

ROUTER_PROMPT = """You are a query classifier for a customer service data analyst agent.
The agent ONLY has the Bitext customer service dataset with columns:
- instruction, response, category, intent

There is NO financial data, dates, years, revenue, or earnings in this dataset.

Classify the user query as exactly one of:
- structured: counts, lists, examples, or distributions of categories/intents/interactions
- unstructured: summarizing or interpreting customer service interactions in the dataset
- out_of_scope: anything else (sports, politics, poems, CRM advice, revenue/earnings/years,
  or questions that need columns not in the dataset)

Examples of out_of_scope:
- "What is the earning from year 2024?"
- "Who won the Champions League?"
- "What's the best CRM software?"

Reply with ONLY the label, nothing else.

Query: {query}
"""

# Fast path so obvious non-dataset questions never reach the agent loop.
_OUT_OF_SCOPE_HINTS = (
    "earning", "earnings", "revenue", "profit", "salary", "stock price",
    "champions league", "president of", "write me a poem", "crm software",
    "best crm", "weather in",
)
_DATASET_HINTS = (
    "category", "categories", "intent", "refund", "shipping", "feedback",
    "account", "order", "cancel", "dataset", "customer", "interaction",
    "bitext", "support", "complaint", "example",
)


def _heuristic_out_of_scope(query: str) -> bool:
    """Keyword guard for questions clearly unrelated to the Bitext dataset."""
    q = query.lower()
    if any(hint in q for hint in _OUT_OF_SCOPE_HINTS) and not any(
        hint in q for hint in _DATASET_HINTS
    ):
        return True
    # Year/financial questions without any dataset vocabulary
    if any(y in q for y in ("2020", "2021", "2022", "2023", "2024", "2025", "year")):
        if not any(hint in q for hint in _DATASET_HINTS):
            return True
    return False


_FOLLOW_UP_PHRASES = (
    "show me more",
    "more examples",
    "3 more",
    "five more",
    "what about",
    "how about",
    "the last two",
    "total count",
    "and refunds",
)


def is_follow_up(query: str, has_prior_turns: bool) -> bool:
    """Short continuations that rely on conversation history."""
    if not has_prior_turns:
        return False
    q = query.lower().strip().strip('"')
    if any(phrase in q for phrase in _FOLLOW_UP_PHRASES):
        return True
    if len(q) < 60 and ("more" in q or "what about" in q):
        return True
    return False


def classify_query(query: str) -> str:
    """Classify a user query as structured, unstructured, or out_of_scope."""
    if _heuristic_out_of_scope(query):
        return "out_of_scope"

    llm = get_llm()
    prompt = ROUTER_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    label = response.content.strip().lower()

    if label not in ["structured", "unstructured", "out_of_scope"]:
        return "out_of_scope"

    return label



