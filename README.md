# Hello World: LangChain Basics

Introduction to LangChain's core abstractions by building a simple person-summarizer chain. Given a block of text about a person, the chain uses a prompt template and an LLM to produce a short summary and two interesting facts.

## What This Covers

**Prompt Templates** define reusable prompt structures with placeholders. `PromptTemplate` takes `input_variables` and a `template` string, and produces a `PromptValue` when invoked.

**LLM Wrappers** provide a unified interface across providers. This project demonstrates three interchangeable backends:
  - `ChatOpenAI` (OpenAI GPT)
  - `ChatOllama` (local models via Ollama)
  - `ChatGoogleGenerativeAI` (Google Gemini)

**Chaining with the Pipe Operator** connects runnables into a `RunnableSequence`. The expression `prompt_template | llm` creates a chain where the prompt output feeds directly into the LLM. Calling `chain.invoke({"information": ...})` runs both steps and returns an `AIMessage`.

## Project Structure

```
main.py           # Prompt template + LLM chain
pyproject.toml    # Dependencies
.env              # API keys (not committed)
```

## Setup

```bash
uv sync
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

For local models, install [Ollama](https://ollama.ai) and pull a model (e.g., `ollama pull gemma3:270m`).

## Run

```bash
uv run python main.py
```

## Dependencies

`langchain`, `langchain-openai`, `langchain-ollama`, `langchain-google-genai`, `python-dotenv`
