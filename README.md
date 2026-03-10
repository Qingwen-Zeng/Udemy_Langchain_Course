# Chat With Your PDF

A question-answering system over PDF documents using Retrieval-Augmented Generation (RAG) with FAISS as the local vector store. Loads a PDF, splits it into chunks, embeds them, and answers natural language questions grounded in the document content.

## What This Covers

**Document Loading** uses `PyPDFLoader` to read a PDF file and return a list of `Document` objects, one per page.

**Text Splitting** with `CharacterTextSplitter` breaks documents into chunks of 1000 characters with 30-character overlap, splitting on newlines. This ensures each chunk fits within the embedding model's context while preserving some continuity between chunks.

**Vector Storage** with FAISS embeds all chunks using `OpenAIEmbeddings` and stores them in a local FAISS index. The index is saved to disk (`faiss_index_react/`) so it can be reloaded without re-embedding.

**Retrieval Chain** combines `create_stuff_documents_chain` (which stuffs retrieved documents into the prompt) with `create_retrieval_chain` (which handles the retrieve-then-answer flow). The prompt template is pulled from LangChain Hub (`langchain-ai/retrieval-qa-chat`).

## Pipeline

```
PDF → PyPDFLoader → CharacterTextSplitter → OpenAIEmbeddings → FAISS
                                                                  ↓
User Query → Retriever → Stuff Documents Chain → OpenAI LLM → Answer
```

## Project Structure

```
main.py           # Full RAG pipeline
react.pdf         # Sample PDF (ReAct paper)
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
```

## Run

```bash
uv run python main.py
```

The script embeds the PDF, saves the FAISS index locally, then queries it with "Give me the gist of ReAct in 3 sentences."

## Dependencies

`langchain`, `langchain-openai`, `langchain-community`, `faiss-cpu`, `pypdf`, `python-dotenv`
