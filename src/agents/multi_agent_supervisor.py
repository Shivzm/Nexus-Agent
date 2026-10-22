"""Multi-Agent Supervisor Workflow using LangGraph.

Implements a supervisor-orchestrated multi-agent architecture where a central
supervisor LLM dynamically routes tasks between researcher, analyst, and writer
agents based on the current state of the workflow.

Architecture:
    supervisor → researcher → supervisor → analyst → supervisor → writer → END

Unlike the linear multi_agent.py, this pattern allows the supervisor to:
    - Decide agent order dynamically based on task completion state
    - Re-route to an agent if its output is insufficient
    - Short-circuit steps if they are not needed for a given task
"""

from typing import Dict, Hashable, Literal
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from src.agents.base import get_llm, get_memory_checkpointer
from config.settings import settings
from tools import search_documents, search_web, write_summary


# ============================================================
# State
# ============================================================

class SupervisorState(MessagesState):
    """Extended state for the supervisor multi-agent system.

    Attributes:
        next_agent:     Which agent the supervisor has chosen to run next.
        research_data:  Raw findings produced by the researcher agent.
        analysis:       Insights and recommendations from the analyst agent.
        final_report:   Formatted executive report produced by the writer agent.
        task_complete:  Set to True by the writer; triggers workflow termination.
        current_task:   The original user request, passed through to every agent.
    """

    next_agent: str
    research_data: str
    analysis: str
    final_report: str
    task_complete: bool
    current_task: str


# ============================================================
# Supervisor
# ============================================================

def _create_supervisor_chain(llm):
    """Build the LangChain chain that powers the supervisor's routing decision."""

    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of specialist agents:

1. researcher  — gathers facts, background, trends, and data
2. analyst     — analyses the research and extracts insights / recommendations
3. writer      — produces the final formatted executive report

Given the current task and the work already completed, decide which agent
should act next.  If the full workflow is done (report exists), reply 'DONE'.

Current progress:
  - Research completed : {has_research}
  - Analysis completed : {has_analysis}
  - Report completed   : {has_report}

Reply with ONLY one word: researcher | analyst | writer | DONE
"""),
        ("human", "{task}"),
    ])

    return supervisor_prompt | llm


def supervisor_agent(state: SupervisorState, llm) -> Dict:
    """Central supervisor that uses the LLM to decide which agent runs next.

    Reads the current state flags (has_research, has_analysis, has_report) and
    asks the LLM to return the name of the next agent or 'DONE'.

    Args:
        state: Current supervisor workflow state.
        llm:   Instantiated chat model.

    Returns:
        State delta with updated ``next_agent`` and a status message.
    """

    messages = state["messages"]
    task = messages[-1].content if messages else "No task provided"

    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report   = bool(state.get("final_report", ""))

    chain    = _create_supervisor_chain(llm)
    decision = chain.invoke({
        "task":         task,
        "has_research": has_research,
        "has_analysis": has_analysis,
        "has_report":   has_report,
    })

    decision_text = decision.content.strip().lower()

    # Map LLM decision → internal routing token
    if "done" in decision_text or has_report:
        next_agent    = "end"
        status_msg    = " Supervisor: All tasks complete!"
    elif "researcher" in decision_text or not has_research:
        next_agent    = "researcher"
        status_msg    = " Supervisor: Starting research phase…"
    elif "analyst" in decision_text or (has_research and not has_analysis):
        next_agent    = "analyst"
        status_msg    = " Supervisor: Research done — moving to analysis…"
    elif "writer" in decision_text or (has_analysis and not has_report):
        next_agent    = "writer"
        status_msg    = " Supervisor: Analysis done — generating report…"
    else:
        next_agent    = "end"
        status_msg    = " Supervisor: Task appears complete."

    return {
        "messages":     [AIMessage(content=status_msg)],
        "next_agent":   next_agent,
        "current_task": task,
    }


# ============================================================
# Researcher
# ============================================================

RESEARCH_TOOLS = [search_documents, search_web]
MAX_TOOL_ROUNDS = 4


def _run_tool_call(call: dict) -> ToolMessage:
    """Execute one tool call and always return a ToolMessage (errors included)."""
    tools_by_name = {t.name: t for t in RESEARCH_TOOLS}
    tool = tools_by_name.get(call["name"])
    if tool is None:
        return ToolMessage(content=f"Unknown tool: {call['name']}",
                           tool_call_id=call["id"], name=call["name"])
    try:
        return tool.invoke(call)
    except Exception as exc:  # keep the loop alive; the LLM can read the error
        return ToolMessage(content=f"Tool error: {exc}",
                           tool_call_id=call["id"], name=call["name"])


def researcher_agent(state: SupervisorState, llm) -> Dict:
    task = state.get("current_task", "the requested topic")

    prompt = f"""You are a research specialist. Research the following topic:

{task}

Tools:
- search_documents: the user's own indexed documents (PDFs, text files)
- search_web: current information from the internet

Call the tools you need, then write structured findings covering:
1. Key facts and background
2. Current trends and recent developments
3. Important statistics or data points
4. Notable examples or case studies

Cite the sources the tools return (file and page, or URL). If the tools return
nothing useful, say so instead of guessing. Be thorough yet concise."""

    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    messages = [HumanMessage(content=prompt)]
    response = llm_with_tools.invoke(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        if not response.tool_calls:
            break
        messages.append(response)
        messages.extend(_run_tool_call(call) for call in response.tool_calls)
        response = llm_with_tools.invoke(messages)

    if response.tool_calls:  # round budget used up: force a written answer
        messages.append(HumanMessage(content="Do not call more tools. Write your findings now."))
        response = llm_with_tools.invoke(messages)

    research_data = response.content or "No findings were produced."

    agent_msg = (
        f"🔍 Researcher: Research on '{task}' complete.\n\n"
        f"Preview:\n{research_data[:400]}…"
    )
    return {
        "messages":      [AIMessage(content=agent_msg)],
        "research_data": research_data,
        "next_agent":    "supervisor",
    }


# ============================================================
# Analyst
# ============================================================

def analyst_agent(state: SupervisorState, llm) -> Dict:
    """Analyses the researcher's findings and surfaces actionable insights.

    Takes ``research_data`` from state and produces a structured analysis
    covering patterns, strategic implications, risks, opportunities, and
    recommendations.

    Args:
        state: Current supervisor workflow state.
        llm:   Instantiated chat model.

    Returns:
        State delta with ``analysis`` populated.
    """

    research_data = state.get("research_data", "")
    task          = state.get("current_task", "")

    prompt = f"""You are a senior data analyst. Analyse the research below and deliver insights.

Research data:
{research_data}

Provide:
1. Key patterns and trends
2. Strategic implications
3. Risks and opportunities
4. Prioritised recommendations

Focus on actionable insights relevant to: {task}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content

    agent_msg = (
        f"📊 Analyst: Analysis complete.\n\n"
        f"Top insights:\n{analysis[:400]}…"
    )

    return {
        "messages":   [AIMessage(content=agent_msg)],
        "analysis":   analysis,
        "next_agent": "supervisor",
    }


# ============================================================
# Writer
# ============================================================

def writer_agent(state: SupervisorState, llm) -> Dict:
    """Synthesises research and analysis into a polished executive report.

    Combines ``research_data`` and ``analysis`` from state into a structured
    document with executive summary, key findings, analysis, recommendations,
    and conclusion.

    Args:
        state: Current supervisor workflow state.
        llm:   Instantiated chat model.

    Returns:
        State delta with ``final_report`` populated and ``task_complete`` set.
    """

    research_data = state.get("research_data", "")
    analysis      = state.get("analysis", "")
    task          = state.get("current_task", "")

    prompt = f"""You are a professional business writer. Produce an executive report based on:

Topic: {task}

Research Findings:
{research_data[:1200]}

Analysis:
{analysis[:1200]}

Structure:
1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Recommendations
5. Conclusion

Write in a clear, professional tone suitable for senior stakeholders."""

    response = llm.invoke([HumanMessage(content=prompt)])
    report   = response.content

    final_report = (
        f"\n📄 FINAL REPORT\n"
        f"{'=' * 52}\n"
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Topic     : {task}\n"
        f"{'=' * 52}\n\n"
        f"{report}\n\n"
        f"{'=' * 52}\n"
        f"Compiled by Multi-Agent AI System\n"
    )

    return {
        "messages":     [AIMessage(content="✍️ Writer: Executive report ready.")],
        "final_report": final_report,
        "next_agent":   "supervisor",
        "task_complete": True,
    }


# ============================================================
# Router
# ============================================================

def router(
    state: SupervisorState,
) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    """Conditional edge function — reads ``next_agent`` from state and routes.

    Returns ``"__end__"`` when ``task_complete`` is True or ``next_agent`` is
    'end'.  Otherwise returns the name of the next node to execute.

    Args:
        state: Current supervisor workflow state.

    Returns:
        Name of the next LangGraph node.
    """

    if state.get("task_complete", False):
        return "__end__"

    next_agent = state.get("next_agent", "supervisor")

    if next_agent == "end":
        return "__end__"

    if next_agent == "supervisor":
        return "supervisor"
    if next_agent == "researcher":
        return "researcher"
    if next_agent == "analyst":
        return "analyst"
    if next_agent == "writer":
        return "writer"

    return "supervisor"   # safe fallback


# ============================================================
# Graph builder
# ============================================================

def build_supervisor_graph(model: str | None = None, checkpointer=None):
    """Build and compile the supervisor multi-agent workflow.

    Creates a LangGraph ``StateGraph`` with four nodes (supervisor, researcher,
    analyst, writer) connected via conditional edges driven by the router
    function.  All nodes share the same ``llm`` instance via closures so the
    model can be swapped at build time.

    Args:
        model:        Optional model override. Falls back to ``settings.default_model``.
        checkpointer: Optional LangGraph checkpointer for thread persistence.
                      Defaults to an in-memory ``MemorySaver``.

    Returns:
        A compiled ``LangGraph`` ready to ``invoke`` or ``stream``.

    Example::

        graph = build_supervisor_graph()
        from typing import cast
        result = graph.invoke(cast(SupervisorState, {"messages": [HumanMessage(content="Research AI in healthcare")], "next_agent": "", "research_data": "", "analysis": "", "final_report": "", "task_complete": False, "current_task": ""}))
        print(result["final_report"])
    """

    if model is None:
        model = settings.default_model

    llm = get_llm(model)

    # Bind llm into each agent via closure
    def _supervisor(state):  return supervisor_agent(state, llm)
    def _researcher(state):  return researcher_agent(state, llm)
    def _analyst(state):     return analyst_agent(state, llm)
    def _writer(state):      return writer_agent(state, llm)

    # Build graph
    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", _supervisor)
    workflow.add_node("researcher", _researcher)
    workflow.add_node("analyst",    _analyst)
    workflow.add_node("writer",     _writer)

    # Entry point — always start at the supervisor
    workflow.set_entry_point("supervisor")

    # Every node routes back through the conditional router
    routing_map: dict[Hashable, str] = {
        "supervisor": "supervisor",
        "researcher": "researcher",
        "analyst":    "analyst",
        "writer":     "writer",
        "__end__":    END,
    }

    for node in ("supervisor", "researcher", "analyst", "writer"):
        workflow.add_conditional_edges(node, router, routing_map)

    if checkpointer is None:
        checkpointer = get_memory_checkpointer()

    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# Convenience export
# ============================================================

graph = build_supervisor_graph()


# ============================================================
# Demo
# ============================================================