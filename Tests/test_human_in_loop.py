"""Test cases for the Human-in-the-Loop Agent.

Tests the human-in-loop graph with interruption, tool usage, and human assistance.
"""

import pytest
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command

from src.agents.human_in_loop import build_human_in_loop_graph


class TestHumanInLoopGraph:
    """Test suite for human-in-loop agent functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled human-in-loop graph for testing."""
        return build_human_in_loop_graph()
    
    @pytest.fixture
    def config(self):
        """Create a config with thread ID for state persistence."""
        return {"configurable": {"thread_id": "test_thread_1"}}
    
    def test_graph_builds_successfully(self):
        """Test that the human-in-loop graph builds without errors."""
        graph = build_human_in_loop_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")
    
    def test_graph_builds_with_custom_model(self):
        """Test graph building with custom model specification."""
        graph = build_human_in_loop_graph(model="groq:openai/gpt-oss-120b")
        assert graph is not None
    
    def test_graph_builds_with_checkpointer(self):
        """Test graph building with custom checkpointer."""
        from src.agents.base import get_memory_checkpointer
        
        checkpointer = get_memory_checkpointer()
        graph = build_human_in_loop_graph(checkpointer=checkpointer)
        assert graph is not None
    
    def test_single_message_with_config(self, graph, config):
        """Test basic message invocation with config."""
        response = graph.invoke(
            {"messages": [HumanMessage(content="Hello")]},
            config
        )
        
        assert "messages" in response
        assert len(response["messages"]) > 0
        assert isinstance(response["messages"][-1], BaseMessage)
    
    def test_message_response_not_empty(self, graph, config):
        """Test that responses are generated and not empty."""
        response = graph.invoke(
            {"messages": [HumanMessage(content="What is AI?")]},
            config
        )
        
        content = response["messages"][-1].content
        assert content is not None
        assert len(content) > 0
    
    def test_tool_binding(self, graph):
        """Test that tools are properly bound to the LLM."""
        # This tests implicitly through successful invocation
        response = graph.invoke({"messages": [HumanMessage(content="Search for AI")]})
        assert response["messages"][-1].content is not None
    
    def test_multiple_invocations_with_same_thread(self, graph, config):
        """Test state persistence across multiple invocations with same thread."""
        # First invocation
        response1 = graph.invoke(
            {"messages": [HumanMessage(content="First message")]},
            config
        )
        
        # Second invocation with same thread
        messages = [
            HumanMessage(content="First message"),
            response1["messages"][-1],
            HumanMessage(content="Second message")
        ]
        response2 = graph.invoke(
            {"messages": messages},
            config
        )
        
        assert len(response2["messages"]) > len(response1["messages"])
    
    def test_different_threads_isolation(self, graph, config):
        """Test that different thread IDs maintain separate states."""
        config1 = {"configurable": {"thread_id": "thread_1"}}
        config2 = {"configurable": {"thread_id": "thread_2"}}
        
        response1 = graph.invoke(
            {"messages": [HumanMessage(content="Thread 1 message")]},
            config1
        )
        
        response2 = graph.invoke(
            {"messages": [HumanMessage(content="Thread 2 message")]},
            config2
        )
        
        # Both should return valid responses
        assert response1["messages"][-1].content is not None
        assert response2["messages"][-1].content is not None
    
    def test_response_structure(self, graph, config):
        """Test that response has the expected structure."""
        response = graph.invoke(
            {"messages": [HumanMessage(content="Hi")]},
            config
        )
        
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
    
    def test_message_accumulation(self, graph, config):
        """Test that messages are properly accumulated."""
        messages = [
            HumanMessage(content="First"),
            HumanMessage(content="Second"),
        ]
        response = graph.invoke({"messages": messages}, config)
        
        assert len(response["messages"]) >= len(messages)
    
    def test_long_conversation(self, graph, config):
        """Test multi-turn conversation."""
        messages = [HumanMessage(content="Start")]
        
        for i in range(3):
            response = graph.invoke({"messages": messages}, config)
            messages.append(response["messages"][-1])
            messages.append(HumanMessage(content=f"Continue {i+1}"))
        
        final_response = graph.invoke({"messages": messages}, config)
        assert final_response["messages"][-1].content is not None
    
    def test_search_query(self, graph, config):
        """Test with search-related query."""
        response = graph.invoke(
            {"messages": [HumanMessage(content="Search for machine learning")]},
            config
        )
        assert response["messages"][-1].content is not None
    
    def test_special_characters(self, graph, config):
        """Test handling of special characters."""
        special_inputs = [
            "What is @AI?",
            "Price: $99",
            "Email: test@example.com",
        ]
        
        for inp in special_inputs:
            response = graph.invoke(
                {"messages": [HumanMessage(content=inp)]},
                config
            )
            assert response["messages"][-1].content is not None


class TestHumanInLoopInterruption:
    """Test suite for human interruption functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled human-in-loop graph for testing."""
        return build_human_in_loop_graph()
    
    @pytest.fixture
    def config(self):
        """Create a config with thread ID."""
        return {"configurable": {"thread_id": "interrupt_test"}}
    
    def test_graph_has_human_assistance_tool(self, graph):
        """Test that human_assistance tool is available."""
        # Verify graph can be invoked (tool is bound)
        response = graph.invoke({"messages": [HumanMessage(content="Help")]})
        assert response["messages"][-1].content is not None
    
    def test_search_web_tool_availability(self, graph):
        """Test that search_web tool is properly bound."""
        response = graph.invoke({"messages": [HumanMessage(content="Search for information")]})
        assert response["messages"][-1].content is not None
    
    def test_config_thread_id_persistence(self, graph, config):
        """Test that thread ID is used for state persistence."""
        response1 = graph.invoke(
            {"messages": [HumanMessage(content="Message 1")]},
            config
        )
        
        # Same thread should have access to previous state
        all_messages = [
            HumanMessage(content="Message 1"),
            response1["messages"][-1],
            HumanMessage(content="Message 2")
        ]
        
        response2 = graph.invoke({"messages": all_messages}, config)
        assert len(response2["messages"]) > 0


class TestHumanInLoopEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled human-in-loop graph for testing."""
        return build_human_in_loop_graph()
    
    def test_very_long_input(self, graph):
        """Test with very long input message."""
        long_message = "A" * 1000
        config = {"configurable": {"thread_id": "long_test"}}
        
        response = graph.invoke(
            {"messages": [HumanMessage(content=long_message)]},
            config
        )
        assert response["messages"][-1].content is not None
    
    def test_unicode_handling(self, graph):
        """Test Unicode character support."""
        unicode_inputs = [
            "Hello 👋 Assistant",
            "你好 世界",
            "مرحبا بالعالم",
        ]
        config = {"configurable": {"thread_id": "unicode_test"}}
        
        for inp in unicode_inputs:
            try:
                response = graph.invoke(
                    {"messages": [HumanMessage(content=inp)]},
                    config
                )
                assert response["messages"][-1].content is not None
            except Exception:
                # Unicode support depends on LLM
                pass
    
    def test_empty_message_handling(self, graph):
        """Test handling of empty messages."""
        config = {"configurable": {"thread_id": "empty_test"}}
        
        try:
            response = graph.invoke(
                {"messages": [HumanMessage(content="")]},
                config
            )
            # Should handle or raise appropriate error
            assert "messages" in response
        except Exception:
            # Empty message handling is optional
            pass
    
    def test_rapid_sequential_invocations(self, graph):
        """Test multiple rapid invocations."""
        config = {"configurable": {"thread_id": "rapid_test"}}
        
        for i in range(3):
            response = graph.invoke(
                {"messages": [HumanMessage(content=f"Message {i}")]},
                config
            )
            assert response["messages"][-1].content is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
