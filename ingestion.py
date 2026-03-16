# =============================================================================
# Documentation Ingestion Pipeline
# =============================================================================
# This script crawls a documentation website (LangChain docs), extracts content,
# splits it into manageable chunks, and indexes those chunks into a vector store
# (Chroma) for later retrieval by a RAG (Retrieval-Augmented Generation) system.
#
# High-level flow:
#   1. Crawl the LangChain documentation site using TavilyCrawl
#   2. Convert raw crawl results into LangChain Document objects
#   3. Split documents into smaller chunks using RecursiveCharacterTextSplitter
#   4. Embed and store chunks in a Chroma vector store (async, batched)
# =============================================================================

import asyncio          # For running async functions (concurrent batch indexing)
import os               # For setting environment variables (SSL cert paths)
import ssl              # For creating a custom SSL context with certifi certs
from typing import Any, Dict, List  # Type hints for function signatures

import certifi          # Provides Mozilla's CA bundle for SSL certificate verification
from dotenv import load_dotenv 

# --- LangChain ecosystem imports ---
from langchain_chroma import Chroma                          # Chroma vector store wrapper
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    # Splits text recursively by trying different separators (\n\n, \n, " ", "")
    # to keep chunks close to the target size while preserving semantic boundaries
from langchain_core.documents import Document                # Standard document container
                                                             # with page_content + metadata
from langchain_openai import OpenAIEmbeddings               
from langchain_pinecone import PineconeVectorStore         

# --- Tavily tools for web crawling and extraction ---
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
    # TavilyCrawl  – Crawls a website and extracts content from each page
    # TavilyExtract – Extracts structured content from specific URLs
    # TavilyMap     – Maps out the site structure (sitemap discovery)

# --- Custom logging utilities ---
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)
    # Colored console logging helpers for tracking pipeline progress


load_dotenv()

# =============================================================================
# SSL Configuration
# =============================================================================
# Here we explicitly point Python's SSL and the `requests` library to the
# certifi-provided CA bundle, ensuring HTTPS connections work reliably.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()       # Used by Python's ssl module
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()   # Used by the requests library

# =============================================================================
# Embedding Model Setup
# =============================================================================
# OpenAI's text-embedding-3-small is a compact, cost-effective embedding model.
# - model: which OpenAI embedding model to use
# - show_progress_bar: disable tqdm progress bar for cleaner logs
# - chunk_size: number of texts to send per API call (batching for efficiency)
# - retry_min_seconds: minimum wait before retrying a failed API call
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# =============================================================================
# Vector Store Setup
# =============================================================================
# Using Chroma as the local vector store. It persists data to the "chroma_db"
# directory so embeddings survive across script runs (no re-embedding needed).
vectorstore = Chroma(persist_directory="langchain-doc-index", embedding_function=embeddings)

# --- Alternative: Pinecone (cloud-hosted vector store) ---
# Uncomment the lines below to use Pin
# econe instead of Chroma.
# Requires PINECONE_API_KEY in .env and a pre-created index named "langchain-docs-2025".
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-2025", embedding=embeddings
# )

# =============================================================================
# Tavily Tool Instances
# Tavily is a search/extraction API service
# =============================================================================
# TavilyExtract: extracts content from a list of specific URLs
tavily_extract = TavilyExtract()

# TavilyMap: discovers the sitemap / link structure of a website
# - max_depth: how many link levels deep to follow from the root URL
# - max_breadth: max number of links to follow per page
# - max_pages: upper bound on total pages to discover
tavily_map = TavilyMap(max_depth=7, max_breadth=20, max_pages=1000)

# TavilyCrawl: crawls a website, following links and extracting page content
tavily_crawl = TavilyCrawl()


# =============================================================================
# Async Batch Indexing Function
# =============================================================================
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """
    Embeds and stores documents in the vector store in parallel batches.

    Why batching?
      - Embedding APIs have rate limits and payload size limits.
      - Sending all documents at once could exceed these limits or cause timeouts.
      - Batching allows concurrent processing while staying within limits.

    Args:
        documents: List of LangChain Document objects to embed and store.
        batch_size: Number of documents per batch (default 50).
                    Larger batches = fewer API calls but higher memory/payload.
    """
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Split the flat list of documents into sublists of `batch_size` each.
    # Example: 1200 docs with batch_size=500 → [500, 500, 200]
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # --- Inner async function to process a single batch ---
    async def add_batch(batch: List[Document], batch_num: int):
        """
        Adds a single batch of documents to the vector store asynchronously.
        Uses the vector store's async interface (aadd_documents) so multiple
        batches can be in-flight concurrently via asyncio.gather.

        Args:
            batch: Subset of documents to add.
            batch_num: 1-indexed batch number for logging.

        Returns:
            True if the batch was added successfully, False otherwise.
        """
        try:
            # aadd_documents: async version of add_documents.
            # Under the hood, this:
            #   1. Sends document texts to OpenAI Embeddings API → gets vectors
            #   2. Upserts (vector, metadata, text) tuples into the vector store
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            # Log the error but don't crash the whole pipeline —
            # other batches can still succeed.
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Create a list of coroutine tasks, one per batch.
    # Each task will run add_batch with its respective batch and index.
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]

    # asyncio.gather runs all tasks concurrently and waits for all to complete.
    # return_exceptions=True means exceptions are returned as values instead of
    # being raised, so one failed batch doesn't cancel the others.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count how many batches succeeded (returned True).
    successful = sum(1 for result in results if result is True)

    # Report final status
    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================
async def main():
    """
    Main async function that orchestrates the full ingestion pipeline:
      1. Crawl → 2. Convert to Documents → 3. Chunk → 4. Index

    This is the entry point for the pipeline logic.
    """
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    # =========================================================================
    # STEP 1: Crawl the documentation website
    # =========================================================================
    # TavilyCrawl.invoke() crawls the given URL and returns structured results.
    # TavilyCrawl() is a tool(Runnable)
    # Parameters:
    #   - url: the root URL to start crawling from
    #   - max_depth: how many link levels deep to follow (2 = root + 2 levels)
    #   - extract_depth: "advanced" uses deeper content extraction (more thorough
    #     than "basic"), pulling raw_content with better formatting preservation
    #   - instructions: optional string to guide the crawler's extraction (not used here) 
    #                   you could add max-depth since the instructions are passed to the underlying LLM that controls the crawl, but we already have a max_depth parameter that does this at a higher level, so it's not necessary here.
    # Returns a dict with a "results" key containing a list of crawled pages,
    # each with keys like: url, raw_content, title, etc.
    res = tavily_crawl.invoke(
    {
        "url": "https://docs.langchain.com/oss/python/deepagents/overview",
        "max_depth": 5,
        "max_breadth": 100,   # follow more links per page
        "limit": 500,          # allow up to 500 total pages
        "extract_depth": "advanced",
    }
)
    print(type(res))
    print(res.keys())

    # =========================================================================
    # STEP 2: Convert crawl results to LangChain Document objects
    # =========================================================================
    # LangChain's Document is the standard container that all downstream
    # components (splitters, vector stores, retrievers) expect.
    # Each Document has:
    #   - page_content (str): the actual text content
    #   - metadata (dict): arbitrary metadata (here we store the source URL)
    """
        Res return
        {
        "results": [
            {
                "url": "https://python.langchain.com/docs/introduction/",
                "raw_content": "LangChain is a framework for...",
                "title": "Introduction | LangChain"
            },
            {
                "url": "https://python.langchain.com/docs/concepts/",
                "raw_content": "Core concepts include...",
                "title": "Concepts | LangChain"
            },
            # ... one entry per crawled page
        ]
    }
    """
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                    # raw_content: the full extracted text of the page
                metadata={"source": tavily_crawl_result_item["url"]},
                    # metadata.source: lets us trace each chunk back to its origin URL
            )
        )

    # =========================================================================
    # STEP 3: Split documents into smaller chunks
    # =========================================================================
    # Why chunk?
    #   - Embedding models have token limits (text-embedding-3-small: 8191 tokens)
    #   - Smaller chunks improve retrieval precision: a 4000-char chunk about
    #     "tool calling" is more relevant to a query about tools than a 50,000-char
    #     page that mentions tools once.
    #
    # RecursiveCharacterTextSplitter tries separators in order:
    #   ["\n\n", "\n", " ", ""] — preferring to split at paragraph boundaries,
    #   then line breaks, then spaces, and only individual chars as last resort.
    #
    # Parameters:
    #   - chunk_size=4000: target maximum characters per chunk
    #   - chunk_overlap=200: overlap between consecutive chunks so that context
    #     at chunk boundaries isn't lost (e.g., a sentence split across chunks
    #     will appear in both, preserving its meaning in at least one)
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    # So many chunk strategies can be selected
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    # split_documents preserves metadata from the parent Document on each chunk,
    # so every chunk still knows which URL it came from.
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # =========================================================================
    # STEP 4: Embed and store chunks in the vector store
    # =========================================================================
    # batch_size=500 here (overriding the default 50) because we're writing to
    # a local Chroma instance with no network rate limits, so larger batches
    # reduce overhead. If using Pinecone, you might want a smaller batch_size.
    await index_documents_async(splitted_docs, batch_size=500)

    # =========================================================================
    # Pipeline Summary
    # =========================================================================
    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


# =============================================================================
# Script Entry Point
# =============================================================================
# asyncio.run() creates a new event loop, runs main() until it completes,
# then closes the loop. This is the standard way to run async code from
# a synchronous __main__ block.
if __name__ == "__main__":
    asyncio.run(main())