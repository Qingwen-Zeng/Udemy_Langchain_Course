# Udemy LangChain Course

Personal learning repository for the Udemy course **"LangChain: Develop AI Agents with LangChain & LangGraph"**. Each branch contains a self-contained project covering a specific LangChain concept, from basic prompt chaining to full agent loops.

## Repository Structure

Each branch is a standalone project with its own dependencies and README.

| Branch | Topic | Key Concepts |
|--------|-------|-------------|
| [`hello-world`](../../tree/hello-world) | LangChain Basics | Prompt templates, LLM chaining, multi-provider support |
| [`chat_with_your_pdf`](../../tree/chat_with_your_pdf) | PDF Q&A with RAG | Document loading, FAISS vector store, retrieval chains |
| [`RAG-gist`](../../tree/RAG-gist) | RAG Pipeline with Pinecone | Ingestion pipeline, LCEL, manual vs declarative RAG |
| [`agents-under-the-hood`](../../tree/agents-under-the-hood) | Agent Internals | Tool calling, agent loops, ReAct prompting from scratch |
| [`search-agent`](../../tree/search-agent) | Search Agents & Structured Output | Tavily search, `create_agent`, Pydantic/TypedDict/Union schemas |

## Tech Stack

All projects use Python 3.11+ with dependency management via [uv](https://docs.astral.sh/uv/). Common libraries across branches include LangChain, LangChain OpenAI, and python-dotenv. Individual branches add specific integrations (FAISS, Pinecone, Ollama, Tavily, Google GenAI).

## Getting Started

1. Clone the repository and check out a branch:
   ```bash
   git clone https://github.com/Qingwen-Zeng/Udemy_Langchain_Course.git
   cd Udemy_Langchain_Course
   git checkout <branch-name>
   ```

2. Install dependencies (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```

3. Create a `.env` file with the required API keys (see each branch's README for details).

4. Run the project:
   ```bash
   uv run python main.py
   ```

## Author

Qingwen Zeng

