from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

#Setup the LLM for the agent

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM object with the Gemini model
llm = init_chat_model("google_genai:gemini-2.5-flash")

# Initialize the embeddings object with the Gemini model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07"
)