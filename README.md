# LangChain Hello World

A minimal introduction to LangChain's core abstractions: `PromptTemplate`, `ChatOpenAI`, and the LCEL pipe (`|`) operator.

## What it does

Takes a block of text about a person, formats it into a prompt, and asks an LLM to produce a short summary plus two interesting facts. Simple, but it demonstrates the fundamental building blocks that every LangChain application uses.

## Key concept: LCEL chaining

```python
chain = summary_prompt_template | llm
response = chain.invoke({"information": information})
```

The `|` operator creates a `RunnableSequence` — the output of `PromptTemplate` (a formatted prompt) becomes the input to `ChatOpenAI` (which returns an `AIMessage`). This pattern scales to arbitrarily complex pipelines.

## Setup

```bash
uv sync
cp .env.example .env  # add your OPENAI_API_KEY
```

## Run

```bash
uv run main.py
```

## Stack

- `langchain-openai` — OpenAI chat model wrapper
- `langchain-ollama` — local model alternative (commented out, swap in as needed)
- `python-dotenv` — environment variable loading
- `uv` — dependency management
