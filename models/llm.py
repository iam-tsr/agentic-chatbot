from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

#Setup the LLM for the agent

#API info. Replace with your own keys and end points
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the embeddings object with the Gemini model
gemini = init_chat_model("google_genai:gemini-2.5-flash")

# Initialize the embeddings object with the Gemini model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    # contents="What is the meaning of life?"
)

# print(model.invoke("Hello, how are you?"))