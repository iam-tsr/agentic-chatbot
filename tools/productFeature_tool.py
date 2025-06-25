__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    # contents="What is the meaning of life?"
)

# Load, chunk and index the contents of the product featuers document.
loader=PyPDFLoader("./data/Laptop product descriptions.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
splits = text_splitter.split_documents(docs)

#Create a vector store with Chroma
prod_feature_store = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)

get_product_features = create_retriever_tool(
    prod_feature_store.as_retriever(search_kwargs={"k": 1}),
    name="Get_Product_Features",
    description="""
    This store contains details about Laptops.
    Return only asked available laptops and their features including CPU, memory, storage, design and advantages
    when asked about product features.
    Check for the very nearest match of the product name in the store.
    If the product is not available, return "Product not available".
    If the product is available, return the product name and its features.
    Do not make up any product features.
    """
)