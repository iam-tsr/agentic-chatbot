from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

import sys
sys.path.append('/mnt/Linux/Projects/Ecommerce-Agentic-Chatbot')

from models.llm import gemini
from tools.productFeature_tool import get_product_features
from tools.productPrice_tool import get_laptop_price

#Create a System prompt to provide a persona to the chatbot
system_prompt = SystemMessage("""
    You are professional chatbot that answers questions about laptops sold by your company.
    To answer questions about laptops, you will ONLY use the available tools and NOT your own memory.
    You will handle small talk and greetings by producing professional responses.
    """
)

#Create a list of tools available
tools = [get_laptop_price, get_product_features]

#Create memory across questions in a conversation (conversation memory)
checkpointer=MemorySaver()

#Create a Product QnA Agent. This is actual a graph in langGraph
product_QnA_agent=create_react_agent(
                                model=gemini, #LLM to use
                                tools=tools, #List of tools to use
                                state_modifier=system_prompt, #The system prompt
                                debug=True, #Debugging turned on if needed
                                checkpointer=checkpointer #For conversation memory
)


if __name__ == "__main__":
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages=[HumanMessage(content="Tell me about the features of SpectraBook")]
    result=product_QnA_agent.invoke({"messages":messages},config)
    print(f"Agent returned : {result['messages'][-1].content} \n")