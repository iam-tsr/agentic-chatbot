from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings


Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash"
)

Settings.embed_model = GeminiEmbedding(
    model="models/gemini-embedding-exp-03-07"
)