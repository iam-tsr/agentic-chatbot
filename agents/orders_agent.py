import sys
sys.path.append('/mnt/Linux/Projects/Ecommerce-Agentic-Chatbot')

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import json
from models.llm import gemini
from tools.orderDetails_tool import get_order_details
from tools.quantityUpdate_tool import update_quantity
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


#Note that this is a string, since the model init only accepts a string.
system_prompt = """
    You are professional chatbot that manages orders for laptops sold by our company.
    The tools allow for retrieving order details as well as update order quantity.
    Do NOT reveal information about other orders than the one requested.
    You will handle small talk and greetings by producing professional responses.
    """

tools = [get_order_details, update_quantity]

checkpointer = MemorySaver()

prebuiltOrders_agent = create_react_agent(
    model=gemini,
    tools=tools,
    state_modifier=system_prompt,
    debug=True,
    checkpointer=checkpointer
)


if __name__ == "__main__":
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages=[HumanMessage(content="What are the details of ORD-2050?")]
    result=prebuiltOrders_agent.invoke({"messages":messages},config)
    print(f"Agent returned : {result['messages'][-1].content} \n")