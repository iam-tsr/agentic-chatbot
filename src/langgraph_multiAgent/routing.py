import sys
sys.path.append(('../agentic-chatbot'))

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
import operator
from src.langgraph_multiAgent.utils.gemini_o1 import llm
from src.langgraph_multiAgent.doc_agent import product_agent
from src.langgraph_multiAgent.math_agent import maths_agent
import uuid


import functools
# Helper function to invoke an agent
def agent_node(state, agent, name, config):

    #extract thread-id from request for conversation memory
    thread_id=config["metadata"]["thread_id"]
    #Set the config for calling the agent
    agent_config = {"configurable": {"thread_id": thread_id}}

    #Pass the thread-id to establish memory for chatbot
    #Invoke the agent with the state
    result = agent.invoke(state, agent_config)

    # Convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        final_result=AIMessage(result['messages'][-1].content)
    return {
        "messages": [final_result]
    }

# Create a reAct_agent node
math_node=functools.partial(agent_node, 
                                   agent=maths_agent, 
                                   name="Math_Agent")

#Create further agent node
# For a custom agent, the agent graph need to be provided as input
product_node=functools.partial(agent_node,
                              agent=product_agent,
                              name="Product_Agent")


# Router state memory
class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class RouterAgent:

    def __init__(self, model, system_prompt, smalltalk_prompt, debug=False):
        
        self.system_prompt=system_prompt
        self.smalltalk_prompt=smalltalk_prompt
        self.model=model
        self.debug=debug
        
        router_graph=StateGraph(RouterAgentState)
        router_graph.add_node("Router",self.call_llm)
        router_graph.add_node("Math_Agent",math_node)
        router_graph.add_node("Product_Agent",product_node)
        router_graph.add_node("Small_Talk", self.respond_smalltalk)
                              
        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {
                "MATH": "Math_Agent",
                "PRODUCT": "Product_Agent",
                "SMALLTALK": "Small_Talk"
            }
        )

        #Set where there graph starts
        router_graph.set_entry_point("Router")

        #One way routing, not coming back to router
        router_graph.add_edge("Math_Agent",END)
        router_graph.add_edge("Product_Agent",END)
        router_graph.add_edge("Small_Talk",END)
        
        self.router_graph = router_graph.compile()

    def call_llm(self, state:RouterAgentState):
        messages=state["messages"]
        if self.debug:
            print(f"Call LLM received {messages}")
            
        #If system prompt exists, add to messages in the front
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        #invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Call LLM result {result}")
        return { "messages":[result]}

    def respond_smalltalk(self, state:RouterAgentState):
        messages=state["messages"]
        if self.debug:
            print(f"Small talk received: {messages}")
            
        #If system prompt exists, add to messages in the front
        
        messages = [SystemMessage(content=self.smalltalk_prompt)] + messages

        #invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Small talk result {result}")
        return { "messages":[result] }
        
    def find_route(self, state:RouterAgentState):
        last_message = state["messages"][-1]
        if self.debug: 
            print("Router: Last result from LLM : ", last_message)

        #Set the last message as the destination
        destination=last_message.content

        if self.debug:
            print(f"Destination chosen : {destination}")
        return destination
    
#Setup the system problem
system_prompt = """ 
You are a Router, that analyzes the input query and chooses 4 options:
SMALLTALK: If the user input is small talk, like greetings and good byes.
PRODUCT: If the query is a product question about laptops or vehicles, like features, specifications and pricing.
MATH: If the query is about orders for addition and multiplication".
END: Default, when its neither PRODUCT or MATH.

The output should only be just one word out of the possible 3 : SMALLTALK, PRODUCT, MATH.
"""

smalltalk_prompt="""
If the user request is small talk, like greetings and goodbyes, respond professionally.
Mention that you will be able to answer questions about laptop product features and provide order status and updates.
"""

router_agent = RouterAgent(llm, 
                           system_prompt, 
                           smalltalk_prompt,
                           debug=False)



def main():

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        while True:
            try:
                user_input = input("USER : ")

                if user_input.lower() == "exit":
                    break

                user_message = {"messages":[HumanMessage(user_input)]}

                result = router_agent.router_graph.invoke(user_message, config=config)

                print(f"AGENT : {result['messages'][-1].content}\n")
                
            except Exception as e:
                print(f"Error occurred: {e}")
                break



if __name__ == "__main__":
    main()