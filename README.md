# Agents Under the Hood

Three implementations of an AI agent loop, each peeling back a layer of abstraction to show how tool calling and reasoning actually work. All three solve the same task: "What is the price of a laptop after applying a gold discount?"

## What This Covers

### 1. LangChain Tool Calling (`1_agent_loop_langchain_tool_calling.py`)

Uses LangChain's `@tool` decorator and `.bind_tools()` to build an agent loop. The `@tool` decorator auto-generates a JSON schema from the function's type hints and docstring. The LLM returns `AIMessage` objects with a `.tool_calls` list; each call includes a unique `id` that the `ToolMessage` must reference so the LLM can match results back to requests.

### 2. Raw Function Calling (`1_agent_loop_raw_function_calling.py`)

Removes LangChain abstractions entirely. Tool schemas are defined as raw JSON dictionaries (the same format LangChain generates automatically). Uses `ollama.chat()` directly with a local Qwen model. The response uses attribute access (`.function.name`) rather than dict access. This shows exactly what `@tool` and `.bind_tools()` hide from you.

### 3. ReAct Prompting (`3_raw_react_prompt.py`)

Implements the ReAct (Reason + Act) pattern from scratch with no tool-calling API at all. A single prompt string instructs the LLM to follow a strict `Thought → Action → Action Input → Observation` text format. Regex parses the LLM's output to extract tool calls. A scratchpad accumulates the full reasoning history across iterations. The `stop=["\\nObservation"]` parameter prevents the LLM from hallucinating tool results.

## Agent Loop Pattern

All three implementations follow the same core loop:

```
User Question
     ↓
┌─── Agent Loop ───────────────────────┐
│  LLM receives messages/prompt        │
│       ↓                              │
│  Tool call detected? ──No──→ Return final answer
│       │ Yes                          │
│       ↓                              │
│  Execute tool function               │
│       ↓                              │
│  Append result to history            │
│       ↓                              │
│  Loop back to LLM                    │
└──────────────────────────────────────┘
```

## Project Structure

```
1_agent_loop_langchain_tool_calling.py   # LangChain @tool + bind_tools
1_agent_loop_raw_function_calling.py     # Raw JSON schemas + ollama.chat()
3_raw_react_prompt.py                    # ReAct prompt + regex parsing
React_Agent.png                          # Diagram: ReAct agent flow
React Prompt VS Function Calling.png     # Diagram: comparison
pyproject.toml                           # Dependencies
```

## Setup

```bash
uv sync
```

Install [Ollama](https://ollama.ai) and pull the model:
```bash
ollama pull qwen3:1.7b
```

Create a `.env` file (for LangSmith tracing, optional):
```
LANGSMITH_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
```

## Run

```bash
# LangChain tool calling
uv run python 1_agent_loop_langchain_tool_calling.py

# Raw function calling (no LangChain tools)
uv run python 1_agent_loop_raw_function_calling.py

# ReAct prompt (no tool-calling API)
uv run python 3_raw_react_prompt.py
```

## Dependencies

`langchain`, `langchain-openai`, `langchain-ollama`, `python-dotenv`
