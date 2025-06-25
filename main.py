#Example custom
import uuid
from langchain_core.messages import HumanMessage
from agents.routing import router_agent

def main():

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        try:
            user_input = input("USER : ")

            if user_input.lower() == "exit":
                break

            # user_message = {"messages":[HumanMessage(user_input)]}

            result = router_agent.router_graph.invoke(user_input, config=config)

            print(f"AGENT : {result['messages'][-1].content}\n")
            
        except Exception as e:
            print(f"Error occurred: {e}")
            break


if __name__ == "__main__":
    main()