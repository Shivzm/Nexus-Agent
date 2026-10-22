"""Multi-Agent Workflow using LangGraph.

Implements a researcher-writer agent architecture where researchers find information
and writers summarize findings.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.base import get_llm, get_memory_checkpointer
from config.settings import settings
from tools import search_web, write_summary, search_documents


def build_multi_agent_graph(model: str | None = None, checkpointer=None):
    """Build and compile the multi-agent workflow.
    
    Args:
        model: Optional model override. Uses settings.default_model if None.
        checkpointer: Optional state persister for thread management.
    
    Returns:
        Compiled LangGraph with researcher and writer agents.
    """
    if model is None:
        model = settings.default_model
    
    llm = get_llm(model)

    research_tools = [search_documents, search_web]
    researcher_llm = llm.bind_tools(research_tools)

    def researcher_agent(state: MessagesState):
        system_msg = SystemMessage(
            content=(
                "You are a research assistant. Use search_documents for questions about "
                "the user's own documents and search_web for current events or anything "
                "outside them. Call a tool before answering, then give your findings and "
                "cite the sources the tools return (file and page, or URL)."
            )
        )
        response = researcher_llm.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def writer_agent(state: MessagesState):
        """Writer agent that creates summaries."""
        messages = state["messages"]
        system_msg = SystemMessage(
            content="You are a technical writer. Review the conversation and create a clear, concise summary of the findings."
        )
        response = llm.invoke([system_msg] + messages)
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("research_tools", ToolNode(research_tools))   # NEW
    workflow.add_node("writer", writer_agent)

    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(                                  # replaces add_edge("researcher", "writer")
        "researcher", tools_condition, {"tools": "research_tools", END: "writer"}
    )
    workflow.add_edge("research_tools", "researcher")                # NEW
    workflow.add_edge("writer", END)
    
    if checkpointer is None:
        checkpointer = get_memory_checkpointer()
    
    return workflow.compile(checkpointer=checkpointer)


# Convenience export
graph = build_multi_agent_graph()


if __name__ == "__main__":
    # Demo
    from langchain_core.messages import HumanMessage
    
    graph = build_multi_agent_graph()
    response = graph.invoke({"messages": [HumanMessage(content="Research about the usecase of agentic ai in business")]})
    print(response["messages"][-1].content)