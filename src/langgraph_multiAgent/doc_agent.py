from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import json
from src.langgraph_multiAgent.utils.gemini_o1 import llm
from src.langgraph_multiAgent.tools.laptop_retrieve import laptop_query
from src.langgraph_multiAgent.tools.ecoSprint_retrieve import ecoSprint_query
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


#Note that this is a string, since the model init only accepts a string.
system_prompt = """
    You are a helpful agent that can answer questions about laptops and EcoSprint vehicles based on the provided knowledge base.
    You will use the tools provided to retrieve information from the knowledge base.
    You will respond in a concise and informative manner.
    If you do not know the answer, you will say "I don't know" instead of making up an answer.
    You will use the tools in the following order:
    1. Use the laptop_query tool to answer questions about laptops.
    2. Use the ecoSprint_query tool to answer questions about EcoSprint vehicles.
    You will not use any other tools or resources.
    You will respond with a single message containing the answer to the question.
    """

tools = [laptop_query, ecoSprint_query]

checkpointer = MemorySaver()

product_agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_prompt,
    debug=False,
    checkpointer=checkpointer
)


if __name__ == "__main__":
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages=[HumanMessage(content="what is ecosprint?")]
    result=product_agent.invoke({"messages":messages},config)
    print(f"Agent returned : {result['messages'][-1].content} \n")