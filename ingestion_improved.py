# =============================================================================
# Documentation Ingestion Pipeline
# =============================================================================
# This script ingests the LangChain documentation site (docs.langchain.com),
# which is built on Mintlify (a JS-rendered SPA).
#
# Why not just use TavilyCrawl/TavilyMap?
#   Mintlify renders sidebar navigation client-side. Tavily's crawler doesn't
#   execute JavaScript, so it only discovers ~20 top-level links instead of
#   the 150+ pages actually available. To work around this, we:
#     1. Fetch each root page ourselves (requests can see the server-rendered HTML)
#     2. Parse all sidebar links from the HTML using BeautifulSoup
#     3. Resolve relative URLs → absolute URLs
#     4. Deduplicate across all roots
#     5. Extract content via TavilyExtract in batches
#     6. Chunk and index into a vector store
#
# High-level flow:
#   1. Discover URLs by parsing sidebar links from each root page
#   2. Deduplicate across all roots
#   3. Extract content from all discovered URLs (TavilyExtract, batched)
#   4. Convert to LangChain Document objects
#   5. Split documents into smaller chunks
#   6. Embed and store chunks in Chroma (async, batched)
# =============================================================================

import asyncio
import os
import ssl
from typing import Any, Dict, List, Set
from urllib.parse import urljoin, urlparse

import certifi
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- LangChain ecosystem imports ---
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Tavily tools ---
from langchain_tavily import TavilyExtract

# --- Custom logging utilities ---
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)


load_dotenv()

# =============================================================================
# SSL Configuration
# =============================================================================
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# =============================================================================
# Embedding Model Setup
# =============================================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# =============================================================================
# Vector Store Setup
# =============================================================================
# --- Alternative: Pinecone ---
# vectorstore = Chroma(persist_directory="langchain-doc-index", embedding_function=embeddings)


vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)

# =============================================================================
# Tavily Extract Instance
# =============================================================================
tavily_extract = TavilyExtract()

# =============================================================================
# Root URLs to Ingest
# =============================================================================
BASE_URL = "https://docs.langchain.com"

ROOT_URLS = [
    f"{BASE_URL}/oss/python/deepagents/overview",
    f"{BASE_URL}/oss/python/langchain/overview",
    f"{BASE_URL}/oss/python/langgraph/overview",
    f"{BASE_URL}/oss/python/integrations/providers/overview",
    f"{BASE_URL}/oss/python/learn",
]

# =============================================================================
# URL Path Prefixes to Include
# =============================================================================
# Only keep URLs under these prefixes to avoid crawling unrelated sections
# (e.g., LangSmith docs, blog posts, external links).
ALLOWED_PREFIXES = [
    "/oss/python/deepagents/",
    "/oss/python/langchain/",
    "/oss/python/langgraph/",
    "/oss/python/integrations/",
    "/oss/python/learn",
    "/oss/python/concepts/",
    "/oss/python/reference/",
    "/oss/python/contributing/",
    "/oss/python/releases/",
]


# =============================================================================
# Step 1: Discover URLs by Parsing Sidebar Links from HTML
# =============================================================================
def discover_urls_from_sidebar(root_urls: List[str]) -> List[str]:
    """
    Fetches each root page's HTML, parses all <a href="..."> links,
    resolves relative URLs to absolute, filters by allowed prefixes,
    and deduplicates.

    Why this approach?
      docs.langchain.com is built on Mintlify, a JS-rendered SPA.
      TavilyMap/TavilyCrawl don't execute JavaScript, so they miss
      most sidebar links. However, the server-rendered HTML still
      contains the sidebar <a> tags — we just need to parse them
      with BeautifulSoup instead of relying on Tavily's link follower.

    Args:
        root_urls: List of root documentation URLs to parse.

    Returns:
        Deduplicated, sorted list of all discovered documentation URLs.
    """
    log_header("URL DISCOVERY PHASE (HTML Sidebar Parsing)")
    all_urls: Set[str] = set()

    for root_url in root_urls:
        try:
            log_info(f"🔍 Parsing sidebar links from: {root_url}", Colors.PURPLE)

            response = requests.get(root_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find all <a> tags with href attributes
            page_urls: Set[str] = set()
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]

                # Skip anchors, javascript:, mailto:, external links
                if href.startswith(("#", "javascript:", "mailto:")):
                    continue

                # Resolve relative URLs against the page's base URL
                absolute_url = urljoin(root_url, href)

                # Parse to inspect the domain and path
                parsed = urlparse(absolute_url)

                # Only keep URLs on docs.langchain.com
                if parsed.netloc and parsed.netloc != "docs.langchain.com":
                    continue

                # Only keep URLs under our allowed path prefixes
                path = parsed.path.rstrip("/")
                if not any(path.startswith(prefix.rstrip("/")) for prefix in ALLOWED_PREFIXES):
                    continue

                # Strip query params and fragments for clean dedup
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                page_urls.add(clean_url)

            before = len(all_urls)
            all_urls.update(page_urls)
            new_count = len(all_urls) - before

            log_success(
                f"Parsed {root_url} → {len(page_urls)} links "
                f"({new_count} new, {len(page_urls) - new_count} duplicates skipped)"
            )

        except Exception as e:
            log_error(f"Failed to parse {root_url} — {e}")

    # Also include the root URLs themselves
    for url in root_urls:
        all_urls.add(url)

    sorted_urls = sorted(all_urls)
    log_success(f"Total unique documentation URLs discovered: {len(sorted_urls)}")

    # Print a breakdown by section for visibility
    section_counts: Dict[str, int] = {}
    for url in sorted_urls:
        path = urlparse(url).path
        parts = path.split("/")
        # /oss/python/<section>/...
        section = parts[3] if len(parts) >= 4 else "other"
        section_counts[section] = section_counts.get(section, 0) + 1

    log_info("📊 URLs by section:", Colors.BOLD)
    for section, count in sorted(section_counts.items()):
        log_info(f"   • {section}: {count} pages")

    return sorted_urls


# =============================================================================
# Step 2: Extract Content via TavilyExtract (batched)
# =============================================================================
def extract_content(urls: List[str], batch_size: int = 20) -> List[Dict[str, Any]]:
    """
    Extracts content from all discovered URLs in batches.
    TavilyExtract accepts up to 20 URLs per call.

    Args:
        urls: List of URLs to extract content from.
        batch_size: URLs per extraction call (max 20 per Tavily API).

    Returns:
        List of result dicts with 'url' and 'raw_content' keys.
    """
    log_header("CONTENT EXTRACTION PHASE (TavilyExtract)")
    log_info(
        f"📄 TavilyExtract: Extracting content from {len(urls)} URLs "
        f"in batches of {batch_size}",
        Colors.DARKCYAN,
    )

    all_results: List[Dict[str, Any]] = []
    total_batches = (len(urls) + batch_size - 1) // batch_size

    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            log_info(
                f"📄 TavilyExtract: Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} URLs)"
            )
            extracted = tavily_extract.invoke({"urls": batch})
            results = extracted.get("results", [])
            all_results.extend(results)

            # Log any URLs that failed extraction
            failed = extracted.get("failed_results", [])
            if failed:
                for fail in failed:
                    log_warning(f"   ⚠ Failed: {fail.get('url', 'unknown')}")

            log_success(
                f"TavilyExtract: Batch {batch_num}/{total_batches} — "
                f"extracted {len(results)} pages"
            )
        except Exception as e:
            log_error(f"TavilyExtract: Batch {batch_num} failed — {e}")

    log_success(f"TavilyExtract: Total pages extracted: {len(all_results)}")
    return all_results


# =============================================================================
# Async Batch Indexing Function
# =============================================================================
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """
    Embeds and stores documents in the vector store in parallel batches.
    """
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches "
        f"of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch "
                f"{batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(
                f"VectorStore Indexing: Failed to add batch {batch_num} - {e}"
            )
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! "
            f"({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} "
            f"batches successfully"
        )


# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================
async def main():
    """
    Orchestrates the full ingestion pipeline:
      1. Parse sidebar links → discover all doc pages
      2. Extract content via TavilyExtract
      3. Convert to LangChain Documents
      4. Chunk documents
      5. Embed and index into vector store
    """
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(f"🎯 Target roots: {len(ROOT_URLS)} documentation sections", Colors.BOLD)
    for url in ROOT_URLS:
        log_info(f"   • {url}")

    # =========================================================================
    # STEP 1: Discover all URLs by parsing sidebar HTML
    # =========================================================================
    all_urls = discover_urls_from_sidebar(ROOT_URLS)

    if not all_urls:
        log_error("No URLs discovered. Exiting pipeline.")
        return

    # =========================================================================
    # STEP 2: Extract content from all discovered URLs
    # =========================================================================
    raw_results = extract_content(all_urls, batch_size=20)

    if not raw_results:
        log_error("No content extracted. Exiting pipeline.")
        return

    # =========================================================================
    # STEP 3: Convert to LangChain Document objects
    # =========================================================================
    log_header("DOCUMENT CONVERSION PHASE")
    all_docs = []
    for item in raw_results:
        content = item.get("raw_content", "")
        url = item.get("url", "unknown")
        if content and content.strip():
            all_docs.append(
                Document(
                    page_content=content,
                    metadata={"source": url},
                )
            )
        else:
            log_warning(f"Skipped (empty content): {url}")

    log_success(f"Converted {len(all_docs)} pages to Document objects")

    # =========================================================================
    # STEP 4: Split documents into smaller chunks
    # =========================================================================
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents "
        f"with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks "
        f"from {len(all_docs)} documents"
    )

    # =========================================================================
    # STEP 5: Embed and store chunks in the vector store
    # =========================================================================
    await index_documents_async(splitted_docs, batch_size=500)

    # =========================================================================
    # Pipeline Summary
    # =========================================================================
    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Root URLs targeted: {len(ROOT_URLS)}")
    log_info(f"   • Unique URLs discovered: {len(all_urls)}")
    log_info(f"   • Pages extracted: {len(raw_results)}")
    log_info(f"   • Documents converted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    asyncio.run(main())