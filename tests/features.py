import sys
sys.path.append('/mnt/Linux/Projects/Ecommerce-Agentic-Chatbot')

from tools.productFeature_tool import prod_feature_store

print(prod_feature_store.as_retriever().invoke("Tell me about the AlphaBook Pro"))