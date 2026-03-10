import os
from operator import itemgetter
# person = {"name": "Alice", "age": 30, "city": "New York"}
# name = itemgetter("name")(person)  # Extracts the value of "name"
# name_and_city = itemgetter("name", "city")(person)  # Extracts both "name" and "city"
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env (OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME)
load_dotenv()

print("Initializing components...")

# --- COMPONENT INITIALIZATION ---

# Embedding model: converts text into vectors for semantic search
# Automatically reads OPENAI_API_KEY from environment
embeddings = OpenAIEmbeddings()

# LLM: the model that generates the final answer
# Automatically reads OPENAI_API_KEY from environment
llm = ChatOpenAI()

# Connect to an existing Pinecone index using the embedding model
# The index must already be populated (via ingestion.py)
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Return VectorStoreRetriever initialized from this VectorStore.
# This VectorStoreRetriever will be used to retrieve relevant 
# chunks from Pinecone based on the user's query.
# k=3 → return the 3 most semantically similar chunks for each query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- PROMPT TEMPLATE ---

# Define the prompt structure sent to the LLM
# {context} → filled with retrieved document chunks
# {question} → filled with the user's query
# "based only on the following context" → prevents the LLM from hallucinating
#  outside of the retrieved knowledge
# question is original prompt from user, context is retrieved chunks from Pinecone, and the
# rest is the augmentation instructions to guide the LLM's response. 
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """
    Format a list of retrieved Document objects into a single string.

    Each document's page_content is joined with double newlines,
    making it easy for the LLM to parse multiple chunks as separate passages.

    Args:
        docs: List of LangChain Document objects returned by the retriever

    Returns:
        A single formatted string containing all retrieved content
    """
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Manual RAG pipeline without using LCEL (LangChain Expression Language).

    Executes each step of the RAG pipeline explicitly and sequentially.
    Useful for understanding what happens under the hood, but not ideal
    for production due to its verbosity and limited composability.

    Limitations:
    - Manual step-by-step execution (no declarative composition)
    - No built-in streaming support (chain.stream() unavailable)
    - No async support without additional code
    - Harder to compose with other chains or add middleware
    - More verbose and error-prone

    Args:
        query: The user's natural language question

    Returns:
        The LLM's answer as a plain string
    """
    # Step 1: Embed the query and retrieve the top-k most relevant chunks
    # from Pinecone based on cosine similarity
    docs = retriever.invoke(query)

    # Step 2: Merge all retrieved Document objects into one context string
    context = format_docs(docs)

    # Step 3: Inject context and question into the prompt template,
    # producing a list of formatted chat messages
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Send the formatted messages to the LLM for completion
    response = llm.invoke(messages)

    # Step 5: Extract and return the text content from the LLM response object
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Build a RAG pipeline using LCEL (LangChain Expression Language).

    LCEL uses the pipe operator (|) to declaratively chain runnables together.
    Each component in the chain receives the output of the previous one,
    making the data flow explicit and composable.

    Chain breakdown:
        1. RunnablePassthrough.assign(context=...)
           → Keeps the original input dict {"question": "..."} intact,
             and adds a new "context" key by:
               a. Extracting "question" from the dict via itemgetter
               b. Passing it to the retriever to fetch relevant chunks
               c. Formatting the chunks into a string via format_docs
            some tips: a problem is that format_docs is a normal function, not a Runnable, 
            do not has invoke() method, so we cannot use it directly in the chain.
            retriever(Runnable) | format_docs(function) | propt_template(Runnable)
            LinaChain will automatically wrap format_docs into a Runnable,
            retriever(Runnable) | RunnableLambda(format_docs(function))) | propt_template(Runnable)
            
            The prompt template need 2 variables: {context} and {question}, 
            but the retriever only returns the context, so we need to use RunnablePassthrough.assign()
            to keep the original input dict and add the context to it, 
            so that the prompt template can access both {context} and {question}.
            
        2. prompt_template
           → Receives {"question": "...", "context": "..."} and renders
             the final prompt string ready for the LLM

        3. llm
           → Receives the rendered prompt and generates a response

        4. StrOutputParser()
           → Extracts the plain text string from the LLM response object,
             equivalent to response.content in the manual approach

    Advantages over non-LCEL approach:
    - Declarative and composable: easy to extend or swap components
    - Built-in streaming:  chain.stream({"question": "..."})
    - Built-in async:      chain.ainvoke() and chain.astream()
    - Batch processing:    chain.batch([...]) for multiple inputs at once
    - Less code: more concise and readable
    - Better observability: integrates natively with LangSmith tracing

    Returns:
        A compiled LCEL chain that accepts {"question": "..."} as input
        and returns the LLM's answer as a plain string
    """
    
    retrieval_chain = (
        # Pass through the original input and enrich it with retrieved context
        RunnablePassthrough.assign(
            # .assign() Merge the Dict input with the output produced by the mapping argument.
            # context=... → adds a new "context" key to the input dict
            # itemgetter("question") → pulls the question from the input dict
            # | retriever            → retrieves relevant chunks from Pinecone
            # | format_docs          → formats chunks into a single context string
            # itemgetter("question") equals lambda x: x["question"], it extracts 
            # the "question" value from the input dict,
            
            # Output:
            #    {"question": "what is Pinecone?",   # ← kept by RunnablePassthrough
            #    "context":  "chunk1\n\nchunk2..."}  # ← added by .assign()
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template   # Render the prompt with {context} and {question}
        | llm               # Generate the answer
        | StrOutputParser() # Extract plain text from the LLM response
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # The query that will be answered using all three implementations
    query = "what is Pinecone in machine learning?"

    # --- IMPLEMENTATION 0: Raw LLM — No RAG ---
    # Sends the question directly to the LLM with no context injection.
    # The LLM relies purely on its training data, which may be outdated
    # or lack domain-specific knowledge, leading to potential hallucinations.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # --- IMPLEMENTATION 1: RAG without LCEL ---
    # Manual step-by-step RAG: retrieve → format → prompt → LLM → parse.
    # Good for understanding the internals, but verbose and less extensible.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # --- IMPLEMENTATION 2: RAG with LCEL ---
    # Declarative RAG pipeline using the pipe operator.
    # Preferred for production: supports streaming, async, batching,
    # and integrates with LangSmith for tracing and debugging.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    # Build and invoke the LCEL chain with the input dict
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)