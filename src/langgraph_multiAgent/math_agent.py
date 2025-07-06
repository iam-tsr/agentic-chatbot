from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import json
from src.langgraph_multiAgent.utils.gemini_o1 import llm

# import sys
# sys.path.append('/mnt/Linux/Projects/agentic-chatbot/src/langgraph_multiAgent/tools')

from src.langgraph_multiAgent.tools.multiply_tool import multiply
from src.langgraph_multiAgent.tools.addition_tool import addition
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


#Note that this is a string, since the model init only accepts a string.
system_prompt = """
    You are a helpful assistant that can perform mathematical operations only.
    """

tools = [multiply, addition]

checkpointer = MemorySaver()

maths_agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_prompt,
    debug=False,
    checkpointer=checkpointer
)


if __name__ == "__main__":
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages=[HumanMessage(content="what is 2 times 3 plus 4?")]
    result=maths_agent.invoke({"messages":messages},config)
    print(f"Agent returned : {result['messages'][-1].content} \n")