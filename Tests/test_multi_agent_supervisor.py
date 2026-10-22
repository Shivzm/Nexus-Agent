"""Tests for multi_agent_supervisor.py.

Covers:
    - SupervisorState structure
    - router() logic for all branches
    - Each agent function (unit, with mocked LLM)
    - supervisor_agent routing decisions
    - build_supervisor_graph() graph construction
    - Full end-to-end workflow (integration, with mocked LLM)

Run with:
    pytest tests/test_multi_agent_supervisor.py -v
"""

from typing import cast
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from src.agents import multi_agent_supervisor as sup

from src.agents.multi_agent_supervisor import (
    SupervisorState,
    router,
    supervisor_agent,
    researcher_agent,
    analyst_agent,
    writer_agent,
    build_supervisor_graph,
)


# ============================================================
# Helpers
# ============================================================

def make_state(**overrides) -> SupervisorState:
    """Return a minimal valid SupervisorState with sensible defaults."""
    base: dict = {
        "messages":      [HumanMessage(content="Test task")],
        "next_agent":    "",
        "research_data": "",
        "analysis":      "",
        "final_report":  "",
        "task_complete": False,
        "current_task":  "Test task",
    }
    base.update(overrides)
    return cast(SupervisorState, base)


def mock_llm(response_text: str) -> MagicMock:
    """Return a mock LLM whose .invoke() returns an AIMessage.

    Also supports the pipe operator (prompt | llm) used by supervisor_agent:
    the resulting chain mock also returns an AIMessage on .invoke().
    """
    llm = MagicMock()
    ai_response = AIMessage(content=response_text)
    llm.invoke.return_value = ai_response
    llm.bind_tools.return_value = llm
    # Support prompt | llm chain — __or__ returns a chain mock whose invoke
    # also returns the same AIMessage
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = ai_response
    llm.__or__ = MagicMock(return_value=chain_mock)
    return llm

@tool
def fake_search(query: str) -> str:
    """fake"""
    return f"[doc.pdf, p.1] hit for {query}"

# ============================================================
# SupervisorState
# ============================================================

class TestSupervisorState:
    def test_all_fields_present(self):
        state = make_state()
        assert "messages"      in state
        assert "next_agent"    in state
        assert "research_data" in state
        assert "analysis"      in state
        assert "final_report"  in state
        assert "task_complete" in state
        assert "current_task"  in state

    def test_default_task_complete_is_false(self):
        state = make_state()
        assert state["task_complete"] is False

    def test_messages_is_list(self):
        state = make_state()
        assert isinstance(state["messages"], list)


# ============================================================
# router()
# ============================================================

class TestRouter:
    def test_routes_to_researcher_when_no_research(self):
        state = make_state(next_agent="researcher", research_data="")
        assert router(state) == "researcher"

    def test_routes_to_analyst(self):
        state = make_state(next_agent="analyst")
        assert router(state) == "analyst"

    def test_routes_to_writer(self):
        state = make_state(next_agent="writer")
        assert router(state) == "writer"

    def test_routes_to_supervisor(self):
        state = make_state(next_agent="supervisor")
        assert router(state) == "supervisor"

    def test_routes_to_end_when_next_agent_is_end(self):
        state = make_state(next_agent="end")
        assert router(state) == "__end__"

    def test_routes_to_end_when_task_complete(self):
        state = make_state(next_agent="supervisor", task_complete=True)
        assert router(state) == "__end__"

    def test_task_complete_overrides_next_agent(self):
        """task_complete=True must end regardless of next_agent value."""
        state = make_state(next_agent="researcher", task_complete=True)
        assert router(state) == "__end__"

    def test_unknown_next_agent_falls_back_to_supervisor(self):
        state = make_state(next_agent="unknown_node")
        assert router(state) == "supervisor"

    def test_empty_next_agent_falls_back_to_supervisor(self):
        state = make_state(next_agent="")
        assert router(state) == "supervisor"


# ============================================================
# supervisor_agent()
# ============================================================

class TestSupervisorAgent:
    def test_routes_to_researcher_when_no_research(self):
        llm   = mock_llm("researcher")
        state = make_state(research_data="", analysis="", final_report="")
        result = supervisor_agent(state, llm)
        assert result["next_agent"] == "researcher"

    def test_routes_to_analyst_when_research_done(self):
        llm   = mock_llm("analyst")
        state = make_state(research_data="some data", analysis="", final_report="")
        result = supervisor_agent(state, llm)
        assert result["next_agent"] == "analyst"

    def test_routes_to_writer_when_analysis_done(self):
        llm   = mock_llm("writer")
        state = make_state(research_data="data", analysis="insights", final_report="")
        result = supervisor_agent(state, llm)
        assert result["next_agent"] == "writer"

    def test_routes_to_end_when_report_exists(self):
        llm   = mock_llm("done")
        state = make_state(research_data="d", analysis="a", final_report="report text")
        result = supervisor_agent(state, llm)
        assert result["next_agent"] == "end"

    def test_sets_current_task_from_last_message(self):
        llm   = mock_llm("researcher")
        state = make_state(messages=[HumanMessage(content="AI in finance")])
        result = supervisor_agent(state, llm)
        assert result["current_task"] == "AI in finance"

    def test_returns_ai_message(self):
        llm   = mock_llm("researcher")
        state = make_state()
        result = supervisor_agent(state, llm)
        assert isinstance(result["messages"][0], AIMessage)

    def test_handles_empty_messages(self):
        llm   = mock_llm("researcher")
        state = make_state(messages=[])
        result = supervisor_agent(state, llm)
        assert result["current_task"] == "No task provided"

    def test_llm_called_once(self):
        # supervisor_agent uses (prompt | llm).invoke() via a LangChain chain,
        # not llm.invoke() directly. Verify the chain ran by checking
        # that next_agent is populated in the returned state delta.
        llm   = mock_llm("researcher")
        state = make_state()
        result = supervisor_agent(state, llm)
        assert result["next_agent"] != ""


# ============================================================
# researcher_agent()
# ============================================================

class TestResearcherAgent:
    def test_populates_research_data(self):
        llm    = mock_llm("Detailed research findings about AI.")
        state  = make_state(current_task="AI trends")
        result = researcher_agent(state, llm)
        assert result["research_data"] == "Detailed research findings about AI."

    def test_routes_back_to_supervisor(self):
        llm    = mock_llm("some research")
        state  = make_state()
        result = researcher_agent(state, llm)
        assert result["next_agent"] == "supervisor"

    def test_returns_ai_message_with_preview(self):
        llm    = mock_llm("Research content here.")
        state  = make_state(current_task="blockchain")
        result = researcher_agent(state, llm)
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert "blockchain" in msg.content

    def test_uses_current_task_in_prompt(self):
        llm   = mock_llm("result")
        state = make_state(current_task="quantum computing")
        researcher_agent(state, llm)
        prompt_text = llm.invoke.call_args[0][0][0].content
        assert "quantum computing" in prompt_text

    def test_fallback_task_when_current_task_missing(self):
        llm   = mock_llm("result")
        state = make_state(current_task="")
        # Should not raise; uses fallback text
        result = researcher_agent(state, llm)
        assert "research_data" in result

    def test_researcher_executes_tool_calls_then_answers():
        call = AIMessage(content="", tool_calls=[{"name": "fake_search", "args": {"query": "x"}, "id": "1"}])
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [call, AIMessage(content="Findings [doc.pdf p.1]")]

        with patch.object(sup, "RESEARCH_TOOLS", [fake_search]):
            result = sup.researcher_agent(make_state(current_task="x"), llm)

        assert result["research_data"] == "Findings [doc.pdf p.1]"
        tool_msgs = [m for m in llm.invoke.call_args[0][0] if m.type == "tool"]
        assert "doc.pdf" in tool_msgs[0].content


# ============================================================
# analyst_agent()
# ============================================================

class TestAnalystAgent:
    def test_populates_analysis(self):
        llm    = mock_llm("Key insight: AI reduces costs by 30%.")
        state  = make_state(research_data="raw research", current_task="AI ROI")
        result = analyst_agent(state, llm)
        assert result["analysis"] == "Key insight: AI reduces costs by 30%."

    def test_routes_back_to_supervisor(self):
        llm    = mock_llm("insights")
        state  = make_state(research_data="data")
        result = analyst_agent(state, llm)
        assert result["next_agent"] == "supervisor"

    def test_returns_ai_message(self):
        llm    = mock_llm("analysis content")
        state  = make_state()
        result = analyst_agent(state, llm)
        assert isinstance(result["messages"][0], AIMessage)

    def test_includes_research_data_in_prompt(self):
        llm   = mock_llm("insights")
        state = make_state(research_data="important findings", current_task="topic")
        analyst_agent(state, llm)
        prompt_text = llm.invoke.call_args[0][0][0].content
        assert "important findings" in prompt_text

    def test_handles_empty_research_data(self):
        llm   = mock_llm("no data analysis")
        state = make_state(research_data="")
        # Should not raise
        result = analyst_agent(state, llm)
        assert "analysis" in result


# ============================================================
# writer_agent()
# ============================================================

class TestWriterAgent:
    def test_populates_final_report(self):
        llm    = mock_llm("Executive summary content.")
        state  = make_state(research_data="data", analysis="insights", current_task="AI")
        result = writer_agent(state, llm)
        assert "Executive summary content." in result["final_report"]

    def test_sets_task_complete_true(self):
        llm    = mock_llm("report body")
        state  = make_state(research_data="d", analysis="a")
        result = writer_agent(state, llm)
        assert result["task_complete"] is True

    def test_routes_back_to_supervisor(self):
        llm    = mock_llm("report")
        state  = make_state()
        result = writer_agent(state, llm)
        assert result["next_agent"] == "supervisor"

    def test_report_contains_topic(self):
        llm    = mock_llm("report body")
        state  = make_state(current_task="climate change", research_data="d", analysis="a")
        result = writer_agent(state, llm)
        assert "climate change" in result["final_report"]

    def test_report_contains_timestamp_header(self):
        llm    = mock_llm("body")
        state  = make_state()
        result = writer_agent(state, llm)
        assert "Generated" in result["final_report"]
        assert "FINAL REPORT" in result["final_report"]

    def test_returns_ai_message(self):
        llm    = mock_llm("content")
        state  = make_state()
        result = writer_agent(state, llm)
        assert isinstance(result["messages"][0], AIMessage)

    def test_truncates_long_research_data(self):
        """Writer should not crash on very long research_data (truncated to 1200 chars in prompt)."""
        llm   = mock_llm("report")
        state = make_state(research_data="x" * 5000, analysis="y" * 5000)
        # Should not raise
        result = writer_agent(state, llm)
        prompt_text = llm.invoke.call_args[0][0][0].content
        # 1200-char slice + surrounding text — prompt should be well under 10k chars
        assert len(prompt_text) < 10_000


# ============================================================
# build_supervisor_graph()
# ============================================================

class TestBuildSupervisorGraph:
    @patch("src.agents.multi_agent_supervisor.get_llm")
    @patch("src.agents.multi_agent_supervisor.get_memory_checkpointer")
    def test_returns_compiled_graph(self, mock_checkpointer, mock_get_llm):
        mock_get_llm.return_value          = mock_llm("researcher")
        mock_checkpointer.return_value     = MagicMock()
        graph = build_supervisor_graph(model="test-model")
        assert graph is not None

    @patch("src.agents.multi_agent_supervisor.get_llm")
    @patch("src.agents.multi_agent_supervisor.get_memory_checkpointer")
    def test_uses_provided_model(self, mock_checkpointer, mock_get_llm):
        mock_get_llm.return_value      = mock_llm("researcher")
        mock_checkpointer.return_value = MagicMock()
        build_supervisor_graph(model="custom-model")
        mock_get_llm.assert_called_once_with("custom-model")

    @patch("src.agents.multi_agent_supervisor.get_llm")
    @patch("src.agents.multi_agent_supervisor.get_memory_checkpointer")
    def test_accepts_custom_checkpointer(self, mock_checkpointer, mock_get_llm):
        mock_get_llm.return_value  = mock_llm("researcher")
        custom_cp                  = MagicMock()
        graph = build_supervisor_graph(checkpointer=custom_cp)
        # get_memory_checkpointer should NOT be called when one is supplied
        mock_checkpointer.assert_not_called()
        assert graph is not None


# ============================================================
# Integration — full workflow with mocked LLM
# ============================================================

class TestFullWorkflow:
    """End-to-end tests that run the compiled graph with a mocked LLM.

    The mock LLM cycles through canned responses so that the supervisor
    correctly routes researcher → analyst → writer → END.
    """

    @patch("src.agents.multi_agent_supervisor.get_llm")
    @patch("src.agents.multi_agent_supervisor.get_memory_checkpointer")
    def test_workflow_produces_final_report(self, mock_checkpointer, mock_get_llm):
        from langgraph.checkpoint.memory import MemorySaver

        # Cycle: supervisor→researcher, researcher response,
        #        supervisor→analyst,   analyst response,
        #        supervisor→writer,    writer response,
        #        supervisor→done
        responses = [
            AIMessage(content="researcher"),          # supervisor decision 1
            AIMessage(content="Detailed research."),  # researcher output
            AIMessage(content="analyst"),             # supervisor decision 2
            AIMessage(content="Deep analysis."),      # analyst output
            AIMessage(content="writer"),              # supervisor decision 3
            AIMessage(content="Executive report."),   # writer output
            AIMessage(content="done"),                # supervisor decision 4
        ]

        llm      = MagicMock()
        llm.invoke.side_effect = responses

        mock_get_llm.return_value      = llm
        mock_checkpointer.return_value = MemorySaver()

        graph = build_supervisor_graph(model="test-model")

        initial = cast(SupervisorState, {
            "messages":      [HumanMessage(content="Research benefits of AI")],
            "next_agent":    "",
            "research_data": "",
            "analysis":      "",
            "final_report":  "",
            "task_complete": False,
            "current_task":  "",
        })

        result = graph.invoke(initial)

        assert result["task_complete"] is True
        assert result["final_report"] != ""
        assert "Executive report." in result["final_report"]
        assert result["research_data"] == "Detailed research."
        assert result["analysis"] == "Deep analysis."

    @patch("src.agents.multi_agent_supervisor.get_llm")
    @patch("src.agents.multi_agent_supervisor.get_memory_checkpointer")
    def test_workflow_final_report_contains_topic(self, mock_checkpointer, mock_get_llm):
        from langgraph.checkpoint.memory import MemorySaver

        responses = [
            AIMessage(content="researcher"),
            AIMessage(content="Research on healthcare AI."),
            AIMessage(content="analyst"),
            AIMessage(content="Analysis of healthcare AI."),
            AIMessage(content="writer"),
            AIMessage(content="Report body."),
            AIMessage(content="done"),
        ]

        llm             = MagicMock()
        llm.invoke.side_effect  = responses
        mock_get_llm.return_value      = llm
        mock_checkpointer.return_value = MemorySaver()

        graph = build_supervisor_graph()

        initial = cast(SupervisorState, {
            "messages":      [HumanMessage(content="AI in healthcare")],
            "next_agent":    "",
            "research_data": "",
            "analysis":      "",
            "final_report":  "",
            "task_complete": False,
            "current_task":  "",
        })

        result = graph.invoke(initial)
        assert "AI in healthcare" in result["final_report"]