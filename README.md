# RAG Pipeline with Pinecone

A complete Retrieval-Augmented Generation pipeline using Pinecone as a managed vector database. Includes a separate ingestion script and three implementations of the retrieval chain: raw LLM (no RAG), manual step-by-step RAG, and declarative RAG with LCEL.

## What This Covers

**Ingestion Pipeline** (`ingestion.py`) loads a text file, splits it into 1000-character chunks with no overlap, embeds each chunk with `OpenAIEmbeddings`, and upserts them into a Pinecone index.

**Manual RAG** (`retrieval_chain_without_lcel`) executes each step explicitly: retrieve top-k documents, format them into a context string, inject into a prompt template, send to the LLM, and extract the response content. Useful for understanding the internals, but verbose.

**LCEL RAG** (`create_retrieval_chain_with_lcel`) uses LangChain Expression Language to compose the same pipeline declaratively with the pipe operator. `RunnablePassthrough.assign()` keeps the original input dict while adding a `context` key from the retriever. The chain supports streaming, async, and batch processing out of the box.

**Comparison** The script runs all three approaches (raw LLM, manual RAG, LCEL RAG) on the same query so you can compare outputs and see how retrieval augmentation improves answer quality.

## Pipeline

```
ingestion.py:
  TextLoader → CharacterTextSplitter → OpenAIEmbeddings → Pinecone

main.py:
  User Query → Pinecone Retriever (k=3) → format_docs → Prompt Template → ChatOpenAI → Answer
```

## Project Structure

```
ingestion.py          # Data ingestion into Pinecone
main.py               # Three RAG implementations
mediumblog1.txt       # Source text document
Intro_RAG.ipynb       # Jupyter notebook walkthrough
Embedding.jpeg        # Diagram: how embeddings work
RAG_Pipeline.jpeg     # Diagram: RAG architecture
Vector_Database.jpeg  # Diagram: vector database concept
pyproject.toml        # Dependencies
```

## Setup

```bash
uv sync
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
INDEX_NAME=your-pinecone-index-name
```

You need a Pinecone account and an existing index configured for OpenAI embedding dimensions (1536).

## Run

First, ingest the source document:
```bash
uv run python ingestion.py
```

Then run the retrieval chain:
```bash
uv run python main.py
```

## Dependencies

`langchain`, `langchain-openai`, `langchain-pinecone`, `langchain-community`, `langchainhub`, `python-dotenv`
