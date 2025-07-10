from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.gemini import GeminiEmbedding
from langchain_core.tools import tool


from src.langgraph_multiAgent.utils.gemini_o2 import GoogleGenAI, GeminiEmbedding
import chromadb

# === Set up ChromaDB client ===
persist_directory = "src/langgraph_multiAgent/tools/chroma_db1"
chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection("pdf_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)

# === Load PDF documents directly ===
pdf_reader = PDFReader()

documents = pdf_reader.load_data(file="src/langgraph_multiAgent/tools/knowledge_base/Laptop_product_descriptions.pdf")

# === Create index and store in ChromaDB ===
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)

query_engine = index.as_query_engine()

# === Wrap query_engine as a LangChain tool ===
@tool
def laptop_query(query: str) -> str:
    """Query the PDF knowledge base for information about laptops only."""
    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


if __name__ == "__main__":
    # === Rehydrate index from ChromaDB for querying (optional; for later sessions) ===
    # index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    # === Query the index ===
    response = query_engine.query("Name all the laptops")
    print("OUTPUT: ", response)