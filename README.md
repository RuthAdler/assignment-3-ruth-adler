# Bitext Customer Service Data Analyst Agent

LangGraph ReAct agent that answers questions about the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) using Nebius Token Factory models.

## Setup (≈5 minutes)

1. **Clone the repo** and create a virtual environment (Python **3.10+** required):

   ```bash
   python3.10 -m venv venv    # or python3.11+
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **API key** — copy `.env.example` to `.env` and set your Nebius Token Factory key:

   ```bash
   cp .env.example .env
   # Edit .env: NEBIUS_API_KEY=...
   ```

3. **First run** downloads the dataset from Hugging Face (~27k rows). Allow a minute on first launch.

## Run the CLI

```bash
python main.py
python main.py --session my_session
python main.py --session my_session --user ruth
```

- **`--session`**: LangGraph checkpoint thread ID — same session restores conversation after restart (`checkpoints/conversations.db`).
- **`--user`**: Profile file ID (defaults to session) — distilled facts in `profiles/<user>.json`.

The CLI prints **reasoning steps** (router classification, tool calls, observations) and then the **final answer**. Type `exit` to quit.

## Streamlit UI (Bonus A)

```bash
streamlit run streamlit_app.py
```

### Manual test (graders — same steps as other tasks)

1. Start the app: `streamlit run streamlit_app.py`
2. In the sidebar, set **Session ID** to `test_streamlit` (do not use a random name unless testing isolation).
3. Ask: `How many refund requests did we get?`
   - Open **Reasoning** on the reply — should show `filter_data` and `count_rows`.
   - Answer should mention **997**.
4. In the same session, ask: `How many complaints did we get?` then `What about refunds?` then `What is the total count of the last two?`
   - Final answer should use prior turns (e.g. **1997**).
5. Change **Session ID** to `test_streamlit_other` — chat should be empty or unrelated.
6. Set **Session ID** back to `test_streamlit` — earlier messages should reappear (checkpoint resume).
7. Optional: quit Streamlit, run `python main.py --session test_streamlit` — CLI should see the same conversation.

### Memory & follow-ups (Task 2)

| Scenario | Example |
|----------|---------|
| Episodic memory | `Show me 3 examples from the REFUND category` → then `Show me 3 more` (uses `offset` on same filter) |
| Cross-turn context | `How many complaints did we get?` → `What about refunds?` → `What is the total count of the last two?` |
| User profile | Tell the agent your name, ask dataset questions, then `What do you remember about me?` |
| Restart | Exit, run again with the same `--session` — prior messages are still in context |

## Query recommender (Bonus B)

When you ask **what to explore next**, the agent suggests a follow-up from **conversation history** and your **user profile**, without running tools until you confirm.

**Example flow (CLI or Streamlit):**

1. `How many refund requests did we get?` — builds context (997 refunds).
2. `What should I query next?` — suggestion only (e.g. intent distribution in REFUND); ends with “Should I go ahead?”
3. `I'd rather see examples instead.` — refines the pending suggestion (still no tools).
4. `Yes, do it.` — runs tools (`filter_data` → `get_examples`, etc.) and shows results.

Phrases that trigger suggestions: `What should I query next?`, `Suggest a follow-up query`, etc. Confirm with `Yes, do it`, `Go ahead`, etc.

### Example queries to try

| Query | What it tests |
|-------|----------------|
| `What categories exist in the dataset?` | Structured — unique values |
| `How many refund requests did we get?` | Multi-step — filter → count |
| `Show me 5 examples of the SHIPPING category.` | Filter → examples |
| `What is the distribution of intents in the ACCOUNT category?` | Filter → value counts |
| `Summarize how agents respond to complaint intents.` | Unstructured — summary tool |
| `What's the best CRM software for handling complaints?` | Out-of-scope decline |
| `Who is the president of France?` | Out-of-scope decline |

## Architecture

```
User → router → agent ⟷ tools → update_profile → END
         ↓    ↓      ↓
    recommend  profile_recall  decline ──→ update_profile
    (Bonus B: suggest/refine, no tools until user confirms)
```

- **Checkpoints** (`memory.py`): `SqliteSaver` persists graph state per `thread_id` (= `--session`).
- **User profile** (`profile.py`): JSON file per `--user` with distilled name, topics, facts (updated after each turn).

- **Router** (`router.py`): Dedicated LLM call classifies each question as `structured`, `unstructured`, or `out_of_scope` before any tools run. Out-of-scope questions go to a polite decline node (no general-knowledge answers).
- **Agent** (`agent.py`): LangGraph ReAct loop — Llama 3.3 70B chooses one tool per step, up to **12 iterations**, then returns a fallback message if stuck.
- **Tools** (`tools.py`): Six pandas-backed tools with Pydantic schemas; a mutable working DataFrame is reset at the start of each question. The working DataFrame is process-global, so the CLI and MCP server are intended for single-user use at a time.

### Tools

| Tool | Purpose |
|------|---------|
| `get_unique_values` | List distinct categories/intents |
| `filter_data` | Subset by category or intent |
| `count_rows` | Count rows in current subset |
| `get_examples` | Sample rows (instruction, response, …) |
| `get_value_counts` | Distribution of values in a column |
| `get_data_for_summary` | Text sample for open-ended summarization |

### Model choice

| Role | Model | Why |
|------|-------|-----|
| Router + agent | `meta-llama/Llama-3.3-70B-Instruct` (Nebius) | Strong instruction-following and tool use for classification and multi-step reasoning. Same model keeps setup simple; temperature 0 for deterministic routing. |

All LLM calls use the Nebius OpenAI-compatible API (`https://api.studio.nebius.ai/v1/`). The API allows **one tool call per turn**; the agent enforces that in software.

## MCP server (Task 3)

Six dataset tools are exposed via [FastMCP](https://github.com/PrefectHQ/fastmcp) in `mcp_server.py`:

| MCP tool | Purpose |
|----------|---------|
| `get_unique_values` | List distinct categories or intents |
| `filter_data` | Filter by category/intent |
| `count_rows` | Count rows in current view |
| `get_examples` | Sample customer/agent rows |
| `get_value_counts` | Distribution in a column |
| `reset_dataset` | Clear filters back to full dataset |

### Start the server

From the repo root (loads the dataset on startup; first run may take ~1 minute):

```bash
source venv/bin/activate
python3.10 mcp_server.py
```

The server uses **stdio** transport (default for `mcp.run()`). It waits for an MCP client to connect on stdin/stdout — you normally do not interact with it directly in the terminal.

### Connect a client (example: Cursor)

1. Copy `mcp_config.example.json` and set the **absolute path** to this repo and your Python 3.10+ binary.
2. In Cursor: **Settings → MCP → Add new global MCP server** (or merge into your MCP config).
3. Restart Cursor / reload MCP. You should see tools under `bitext-analyst`.
4. In chat, ask the model to use a tool, e.g.:

   > Use the `filter_data` tool with column `intent` and value `get_refund`, then call `count_rows`.

**Example `mcp_config.example.json` entry:**

```json
{
  "mcpServers": {
    "bitext-analyst": {
      "command": "/opt/homebrew/bin/python3.10",
      "args": ["/ABSOLUTE/PATH/TO/assignment-3-ruth-adler/mcp_server.py"],
      "cwd": "/ABSOLUTE/PATH/TO/assignment-3-ruth-adler"
    }
  }
}
```

### Test one tool from the command line

With the FastMCP client (Python 3.10+, repo deps installed), call a tool over stdio:

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("mcp_server.py") as client:
        await client.call_tool("filter_data", {"column": "intent", "value": "get_refund"})
        result = await client.call_tool("count_rows", {})
        print(result.data)   # expects 997

asyncio.run(main())
```

## Project layout

```
agent.py      # LangGraph graph (router, agent, tools, profile, checkpoints)
router.py     # Query classifier
tools.py      # Tool definitions
profile.py    # Per-user distilled profile (JSON files)
memory.py     # SQLite checkpointer setup
data.py       # Hugging Face dataset loader
main.py          # Interactive CLI (--session, --user)
streamlit_app.py   # Streamlit chat UI (Bonus A)
recommender.py     # Bonus B query recommender
mcp_server.py    # FastMCP server (Task 3, Python 3.10+)
mcp_config.example.json
requirements.txt
```

## Debugging

- **Console**: The CLI already prints each tool call and observation.
- **LangGraph Studio**: Optional — point Studio at this project to visualize the graph step by step.

## Author

**Ruth Adler** — solo submission (one student; allowed by course staff).

Repo name: `assignment-3-ruth-adler` (first and last name in the repository name).
