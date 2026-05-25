from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from data import load_data

# Global state to hold the current working DataFrame
_current_df: Optional[pd.DataFrame] = None

def reset_df():
    """Reset the working DataFrame to the full dataset."""
    global _current_df
    _current_df = load_data()

def get_df() -> pd.DataFrame:
    """Get the current working DataFrame, loading it if necessary."""
    global _current_df
    if _current_df is None:
        reset_df()
    return _current_df


VALID_COLUMNS = frozenset({"category", "intent"})

# Map natural-language terms in questions to dataset category values.
CATEGORY_KEYWORDS = {
    "shipping": "SHIPPING",
    "account": "ACCOUNT",
    "feedback": "FEEDBACK",
    "refund": "REFUND",
    "order": "ORDER",
    "cancel": "CANCEL",
    "invoice": "INVOICE",
    "payment": "PAYMENT",
    "contact": "CONTACT",
    "delivery": "DELIVERY",
    "subscription": "SUBSCRIPTION",
    "complaint": "FEEDBACK",
    "complaints": "FEEDBACK",
}


def _validate_column(column: str) -> Optional[str]:
    """Return an error message if column is invalid, else None."""
    if column not in VALID_COLUMNS:
        return (
            f"Invalid column '{column}'. Valid columns are: {', '.join(sorted(VALID_COLUMNS))}. "
            "This dataset has no financial or date fields."
        )
    return None


def _categories_from_question(question: str) -> List[str]:
    """Extract category filters mentioned in a natural-language question."""
    q = question.lower()
    found = []
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in q and category not in found:
            found.append(category)
    return found


def _filter_df_for_question(df: pd.DataFrame, question: str) -> Tuple[pd.DataFrame, str]:
    """Narrow the dataframe using topic words from the question."""
    q = question.lower()
    notes: List[str] = []
    filtered = df

    categories = _categories_from_question(question)
    if categories:
        filtered = filtered[filtered["category"].str.upper().isin(categories)]
        notes.append(f"categories={categories}")

    # Single-topic questions can narrow further by intent.
    single_topic = len(categories) <= 1
    if single_topic and ("refund" in q or "money back" in q):
        filtered = filtered[filtered["intent"] == "get_refund"]
        notes.append("intent=get_refund")
    elif single_topic and "complaint" in q:
        complaint_rows = filtered[filtered["intent"].str.lower() == "complaint"]
        if len(complaint_rows) > 0:
            filtered = complaint_rows
            notes.append("intent=complaint")
        elif not categories:
            filtered = filtered[filtered["category"] == "FEEDBACK"]
            notes.append("category=FEEDBACK")

    if notes:
        context = ", ".join(notes)
    else:
        context = "no topic filter matched; using current dataset view"
    return filtered, context


def _sample_for_summary(df: pd.DataFrame, question: str, max_rows: int = 20) -> pd.DataFrame:
    """Sample rows, spreading across multiple categories when the question mentions several."""
    categories = _categories_from_question(question)
    if len(categories) >= 2:
        per_category = max(3, max_rows // len(categories))
        chunks = []
        for category in categories:
            part = df[df["category"].str.upper() == category.upper()].head(per_category)
            if len(part) > 0:
                chunks.append(part)
        if chunks:
            return pd.concat(chunks).head(max_rows)
    return df.head(max_rows)


# --- Tool 1 ---
class GetUniqueValuesInput(BaseModel):
    column: str

@tool(args_schema=GetUniqueValuesInput)
def get_unique_values(column: str) -> List[str]:
    """Get all distinct values in a column across the current (possibly filtered) dataset.
    Use for questions like 'what categories exist?' or 'what intents are there?'
    Do not use for counts or distributions — use count_rows or get_value_counts instead.
    Valid columns: category, intent.
    """
    err = _validate_column(column)
    if err:
        return err
    df = get_df()
    return df[column].unique().tolist()


# --- Tool 2 ---
class FilterDataInput(BaseModel):
    column: str
    value: str

@tool(args_schema=FilterDataInput)
def filter_data(column: str, value: str) -> str:
    """Filter the working dataset to rows matching a column value (e.g. intent='get_refund').
    Always filter before count_rows, get_examples, or get_value_counts when the question
    is about a specific category or intent (e.g. refund requests → intent get_refund).
    Valid columns: category, intent. Values are case-insensitive.
    """
    err = _validate_column(column)
    if err:
        return err
    global _current_df
    df = get_df()
    _current_df = df[df[column].str.lower() == value.lower()]
    return f"Filtered to {len(_current_df)} rows where {column} = '{value}'"


# --- Tool 3 ---
class CountRowsInput(BaseModel):
    pass

@tool(args_schema=CountRowsInput)
def count_rows() -> int:
    """Use this tool to count the number of rows in the current dataset.
    Use after filter_data to count how many records match a filter.
    """
    return len(get_df())


# --- Tool 4 ---
class GetExamplesInput(BaseModel):
    n: int = 3
    offset: int = Field(
        0,
        description="Number of rows to skip in the current view (use for 'show me more' follow-ups).",
    )

@tool(args_schema=GetExamplesInput)
def get_examples(n: int = 3, offset: int = 0) -> List[dict]:
    """Use this tool to get sample rows from the current dataset.
    Use after filter_data to show example instructions and responses.
    For follow-ups like 'show me 3 more', reuse the same filter and increase offset.
    Returns n rows showing instruction, intent, category, and response.
    """
    df = get_df()
    sample = df[["instruction", "intent", "category", "response"]].iloc[offset : offset + n]
    return sample.to_dict(orient="records")


# --- Tool 5 ---
class SummarizeDataInput(BaseModel):
    question: str

@tool(args_schema=SummarizeDataInput)
def get_data_for_summary(question: str) -> str:
    """Use for open-ended questions that need summarization or interpretation.
    Automatically filters by categories/intents mentioned in the question (e.g. FEEDBACK,
    complaint, refund, shipping). For a single known category you may filter_data first, then call this.
    Returns sample customer/agent exchanges as text to summarize.
    """
    base_df = get_df()
    filtered, context = _filter_df_for_question(base_df, question)
    if len(filtered) == 0:
        return (
            f"No rows matched the topics in '{question}' ({context}). "
            "Try filter_data with a valid category or intent, then call this again."
        )

    sample = _sample_for_summary(filtered, question)
    rows = "\n".join(
        f"Customer: {r['instruction']}\nAgent: {r['response']}"
        for _, r in sample[["instruction", "response"]].iterrows()
    )
    return (
        f"Samples for '{question}' ({len(sample)} of {len(filtered)} rows; {context}):\n\n{rows}"
    )


# --- Tool 6 ---
class GetValueCountsInput(BaseModel):
    column: str = Field(
        description="Column to count values for. Use 'intent' after filtering by category."
    )

@tool(args_schema=GetValueCountsInput)
def get_value_counts(column: str) -> Dict[str, int]:
    """Return how many rows each value appears in for the given column (distribution).
    Use after filter_data for questions like 'distribution of intents in ACCOUNT category'
    (first filter category=ACCOUNT, then call with column='intent').
    Valid columns: intent, category.
    """
    err = _validate_column(column)
    if err:
        return err
    df = get_df()
    return df[column].value_counts().to_dict()


TOOLS = [
    get_unique_values,
    filter_data,
    count_rows,
    get_examples,
    get_data_for_summary,
    get_value_counts,
]