import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from .env file (OPENAI_API_KEY, INDEX_NAME, PINECONE_API_KEY)
load_dotenv()

if __name__ == "__main__":

    # Step 1: Load the source document from the same directory as this script
    print("Ingesting...")
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "mediumblog1.txt"), encoding="utf-8")
    document = loader.load()

    # Step 2: Split the document into smaller chunks for embedding
    # chunk_size=1000 → each chunk is max 1000 characters
    # chunk_overlap=0 → no overlap between chunks
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts is a list of Document objects, each containing a chunk of the original document
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Step 3: Initialize the embedding model
    # Converts each text chunk into a vector representation
    embeddings = OpenAIEmbeddings()

    # Step 4: Embed all chunks and store them in Pinecone vector database
    # from_documents() handles both embedding and upserting into the vector store in one call
    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")