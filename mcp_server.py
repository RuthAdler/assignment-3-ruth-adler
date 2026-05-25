"""
FastMCP server exposing Bitext dataset tools.

Requires Python 3.10+ (FastMCP does not support 3.9).
Run:  python3.10 mcp_server.py
"""

from fastmcp import FastMCP

from tools import (
    count_rows as lc_count_rows,
    filter_data as lc_filter_data,
    get_examples as lc_get_examples,
    get_unique_values as lc_get_unique_values,
    get_value_counts as lc_get_value_counts,
    reset_df,
)

mcp = FastMCP(
    "bitext-analyst",
    instructions=(
        "Tools for the Bitext customer service dataset (categories, intents, "
        "customer/agent messages). Typical flow: filter_data → count_rows or "
        "get_examples. Valid filter columns: category, intent."
    ),
)


@mcp.tool
def get_unique_values(column: str) -> list:
    """
    List distinct values in a column.

    Args:
        column: 'category' or 'intent'

    Use for questions like what categories exist in the dataset.
    """
    return lc_get_unique_values.invoke({"column": column})


@mcp.tool
def filter_data(column: str, value: str) -> str:
    """
    Filter the working dataset to rows matching a column value.

    Args:
        column: 'category' or 'intent'
        value: e.g. REFUND, SHIPPING, get_refund

    Call before count_rows or get_examples for scoped queries.
    """
    return lc_filter_data.invoke({"column": column, "value": value})


@mcp.tool
def count_rows() -> int:
    """Count rows in the current filtered dataset view."""
    return lc_count_rows.invoke({})


@mcp.tool
def get_examples(n: int = 3, offset: int = 0) -> list:
    """
    Sample rows from the current view (instruction, intent, category, response).

    Args:
        n: number of examples (default 3)
        offset: skip rows for pagination (default 0)
    """
    return lc_get_examples.invoke({"n": n, "offset": offset})


@mcp.tool
def get_value_counts(column: str) -> dict:
    """
    Count rows per value in a column (distribution).

    Args:
        column: 'category' or 'intent'

    Use after filter_data (e.g. filter category=ACCOUNT, then column=intent).
    """
    return lc_get_value_counts.invoke({"column": column})


@mcp.tool
def reset_dataset() -> str:
    """Reset to the full unfiltered Bitext dataset (clears any filter_data)."""
    reset_df()
    return "Dataset reset to the full Bitext training set."


def main() -> None:
    """Load data once at startup, then serve tools over stdio (MCP default)."""
    reset_df()
    mcp.run()


if __name__ == "__main__":
    main()
