"""ReAct Agent using LangGraph.

Implements the ReAct (Reasoning + Acting) pattern for autonomous agents
that can reason about problems and execute tools.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agents.base import AgentState, get_llm, get_memory_checkpointer, get_react_model
from config.settings import settings
from tools import search_web, multiply, search_documents

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research assistant with a private knowledge base of research papers "
    "(RAG, retrieval, LLM agents, language models) and access to the web.\n"
    "- For questions on these topics, call search_documents FIRST and base your answer "
    "on the passages it returns.\n"
    "- Use search_web only for current events, or when search_documents finds nothing relevant.\n"
    "- Cite sources as [file, p.N]. Do not state details the retrieved passages do not "
    "support; if they do not contain the answer, say so.\n"
    "- Use multiply for arithmetic."
))

def compact_history(messages):
    """Replace tool outputs from earlier turns with a short stub.

    Keeps the conversation (questions, answers, tool calls) but drops the bulky
    retrieved passages / web results of previous questions, so the prompt does not
    grow without bound (Groq's free tier caps a request at 8,000 tokens).
    """
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
    compacted = []
    for i, m in enumerate(messages):
        if m.type == "tool" and i < last_human:
            m = ToolMessage(content="[earlier tool output omitted]",
                            tool_call_id=m.tool_call_id, name=m.name)
        compacted.append(m)
    return compacted

def build_react_agent(model: str | None = None, checkpointer=None):
    """Build and compile the ReAct agent graph.
    
    Args:
        model: Optional model override. Uses settings.default_model if None.
        checkpointer: Optional state persister. Uses MemorySaver if None.
    
    Returns:
        Compiled LangGraph with agent and tools nodes.
    """
    
    llm = get_llm(get_react_model())
    tools = [search_web, multiply, search_documents]
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState):
        """Agent node that calls the LLM."""
        return {"messages": [llm_with_tools.invoke([SYSTEM_PROMPT] + compact_history(state["messages"]))]}

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