# Search Agent & Structured Output

Builds a web search agent using LangChain's `create_agent` (LangGraph-based) with Tavily for real-time internet search. Progresses from a basic agent to structured output schemas, covering five schema types and two output strategies.

## What This Covers

### Basic Search Agent (`tool.py`)

Uses `create_agent()` to create a LangGraph state machine with two nodes: "agent" (calls the LLM) and "tools" (executes functions). The agent loops between these nodes until the LLM produces a final answer with no tool calls. `TavilySearch()` provides web search as a pre-built tool.

### Structured Output with Schema (`tool_schema.py`)

Adds `response_format=AgentResponse` to `create_agent()`, which forces the agent's final response into a validated Pydantic model with `answer` and `sources` fields. The structured response is available via `result["structured_response"]`.

### Schema Types & Strategies (`tool_schema_parameters_example.py`)

A comprehensive reference covering:

**Two Output Strategies:**
  - **ProviderStrategy**: uses the model provider's native structured output API (highest reliability, no Union support)
  - **ToolStrategy**: uses tool calling to enforce the schema (supports Union types and automatic error retry)

**Five Schema Types:**
  - **Pydantic Model**: full validation (ge, le, Literal), nested models, returns a validated instance
  - **TypedDict**: lightweight, returns a plain dict
  - **Dataclass**: lightweight, returns a plain dict
  - **JSON Schema**: raw dict definition, useful for dynamic schemas
  - **Union Types**: the model picks whichever schema best fits the input (ToolStrategy only)

## Project Structure

```
tool.py                              # Basic search agent
tool_schema.py                       # Agent with Pydantic response schema
tool_schema_parameters_example.py    # All schema types and strategies
pyproject.toml                       # Dependencies
```

## Setup

```bash
uv sync
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Run

```bash
# Basic search agent
uv run python tool.py

# Structured output agent
uv run python tool_schema.py

# Schema examples (interactive menu)
uv run python tool_schema_parameters_example.py
```

The schema examples script presents an interactive menu to run individual examples (2a, 2b, 3a-3e) or all at once.

## Dependencies

`langchain`, `langchain-openai`, `langchain-tavily`, `tavily-python`, `python-dotenv`
