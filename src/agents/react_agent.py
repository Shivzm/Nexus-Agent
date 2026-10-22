"""ReAct Agent using LangGraph.

Implements the ReAct (Reasoning + Acting) pattern for autonomous agents
that can reason about problems and execute tools.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

from src.agents.base import AgentState, get_llm, get_memory_checkpointer, get_react_model
from config.settings import settings
from tools import search_web, multiply, search_documents


def build_react_agent(model: str | None = None, checkpointer=None):
    """Build and compile the ReAct agent graph.
    
    Args:
        model: Optional model override. Uses settings.default_model if None.
        checkpointer: Optional state persister. Uses MemorySaver if None.
    
    Returns:
        Compiled LangGraph with agent and tools nodes.
    """
    if model is None:
        model = settings.default_model
    
    llm = get_llm(get_react_model())
    tools = [search_web, multiply, search_documents]
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState):
        """Agent node that calls the LLM."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    if checkpointer is None:
        checkpointer = get_memory_checkpointer()
    
    return builder.compile(checkpointer=checkpointer)


# Convenience export
graph = build_react_agent()


if __name__ == "__main__":
    # Demo/test code
    response = graph.invoke({"messages": [HumanMessage(content="What is 5 + 3?")]})
    print(response["messages"][-1].content)