## Multi-Agents

We have utilized two specialized agents, each focused on a specific task. To manage these agents, we implemented a LangGraph workflow with added routing capabilities.

for more in-deapth detail go through langgraph documentation - [Link](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

## Routing

Routing classifies an input and directs it to a followup task.

> Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.
> 
> When to use this workflow: Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

![Routing Workflow Diagram](../images/routing-workflow.png)