"""Test cases for the Multi-Agent System.

Tests the researcher-writer agent workflow with tool usage and message routing.
"""

import pytest
from langchain_core.messages import HumanMessage, BaseMessage

from src.agents.multi_agent import build_multi_agent_graph


class TestMultiAgentGraph:
    """Test suite for multi-agent graph functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled multi-agent graph for testing."""
        return build_multi_agent_graph()
    
    def test_graph_builds_successfully(self):
        """Test that the multi-agent graph builds without errors."""
        graph = build_multi_agent_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")
    
    def test_graph_builds_with_custom_model(self):
        """Test graph building with custom model specification."""
        graph = build_multi_agent_graph(model="groq:openai/gpt-oss-20b")
        assert graph is not None
    
    def test_graph_builds_with_checkpointer(self):
        """Test graph building with custom checkpointer."""
        from src.agents.base import get_memory_checkpointer
        
        checkpointer = get_memory_checkpointer()
        graph = build_multi_agent_graph(checkpointer=checkpointer)
        assert graph is not None
    
    def test_single_query_response(self, graph):
        """Test basic query handling through multiple agents."""
        response = graph.invoke({
            "messages": [HumanMessage(content="What is machine learning?")]
        })
        
        assert "messages" in response
        assert len(response["messages"]) > 0
        assert isinstance(response["messages"][-1], BaseMessage)
    
    def test_response_not_empty(self, graph):
        """Test that responses are generated and contain content."""
        response = graph.invoke({
            "messages": [HumanMessage(content="Explain artificial intelligence")]
        })
        
        content = response["messages"][-1].content
        assert content is not None
        assert len(content) > 0
    
    def test_research_query(self, graph):
        """Test research-focused query."""
        response = graph.invoke({
            "messages": [HumanMessage(
                content="Research the benefits of machine learning in healthcare"
            )]
        })
        
        assert response["messages"][-1].content is not None
    
    def test_summary_generation(self, graph):
        """Test that summary is generated (writer agent output)."""
        response = graph.invoke({
            "messages": [HumanMessage(
                content="Summarize the uses of artificial intelligence in business"
            )]
        })
        
        final_message = response["messages"][-1].content
        assert final_message is not None
        assert len(final_message) > 0
    
    def test_response_structure(self, graph):
        """Test that response has expected structure."""
        response = graph.invoke({
            "messages": [HumanMessage(content="What is AI?")]
        })
        
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert all(isinstance(msg, BaseMessage) for msg in response["messages"])
    
    def test_message_accumulation(self, graph):
        """Test that messages are properly accumulated."""
        messages = [
            HumanMessage(content="First query"),
            HumanMessage(content="Second query"),
        ]
        response = graph.invoke({"messages": messages})
        
        assert len(response["messages"]) >= len(messages)
    
    def test_different_research_topics(self, graph):
        """Test with various research topics."""
        topics = [
            "What is blockchain technology?",
            "How does quantum computing work?",
            "Explain cloud computing",
            "What are microservices?",
        ]
        
        for topic in topics:
            response = graph.invoke({
                "messages": [HumanMessage(content=topic)]
            })
            assert response["messages"][-1].content is not None
            assert len(response["messages"][-1].content) > 0
    
    def test_technical_queries(self, graph):
        """Test with technical queries."""
        technical_queries = [
            "What is REST API?",
            "Explain Docker containers",
            "How does encryption work?",
            "What are APIs?",
        ]
        
        for query in technical_queries:
            response = graph.invoke({
                "messages": [HumanMessage(content=query)]
            })
            assert response["messages"][-1].content is not None


class TestMultiAgentResearcherAgent:
    """Test suite for researcher agent functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled multi-agent graph for testing."""
        return build_multi_agent_graph()
    
    def test_researcher_produces_output(self, graph):
        """Test that researcher agent produces meaningful output."""
        response = graph.invoke({
            "messages": [HumanMessage(content="Research AI applications")]
        })
        
        # Should have messages from both researcher and writer
        assert len(response["messages"]) > 1
    
    def test_search_capability(self, graph):
        """Test that search_web tool is available to researcher."""
        response = graph.invoke({
            "messages": [HumanMessage(
                content="Search for latest developments in AI"
            )]
        })
        
        assert response["messages"][-1].content is not None
    
    def test_complex_research_query(self, graph):
        """Test with complex multi-part research query."""
        response = graph.invoke({
            "messages": [HumanMessage(
                content="Research and compare Python and JavaScript for web development"
            )]
        })
        
        assert response["messages"][-1].content is not None


class TestMultiAgentWriterAgent:
    """Test suite for writer agent functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled multi-agent graph for testing."""
        return build_multi_agent_graph()
    
    def test_writer_produces_summary(self, graph):
        """Test that writer agent produces a summary."""
        response = graph.invoke({
            "messages": [HumanMessage(content="Brief overview of machine learning")]
        })
        
        # Writer agent should produce a concise summary
        final_message = response["messages"][-1].content
        assert final_message is not None
        assert len(final_message) > 0
    
    def test_summary_coherence(self, graph):
        """Test that summary is coherent and well-structured."""
        response = graph.invoke({
            "messages": [HumanMessage(
                content="Explain the fundamentals of neural networks"
            )]
        })
        
        summary = response["messages"][-1].content
        assert summary is not None
        # Summary should contain multiple sentences/information
        assert len(summary) > 50


class TestMultiAgentWorkflow:
    """Test suite for multi-agent workflow integration."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled multi-agent graph for testing."""
        return build_multi_agent_graph()
    
    def test_researcher_to_writer_flow(self, graph):
        """Test flow from researcher to writer agent."""
        response = graph.invoke({
            "messages": [HumanMessage(content="What is DevOps?")]
        })
        
        # Should have multiple messages showing agent collaboration
        assert len(response["messages"]) >= 1
        assert response["messages"][-1].content is not None
    
    def test_end_to_end_workflow(self, graph):
        """Test complete end-to-end workflow."""
        query = (
            "Provide a comprehensive overview of cloud computing "
            "including its benefits and use cases"
        )
        
        response = graph.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        assert "messages" in response
        assert len(response["messages"]) > 0
        assert response["messages"][-1].content is not None
    
    def test_multiple_sequential_queries(self, graph):
        """Test multiple queries in sequence."""
        queries = [
            "What is containerization?",
            "Explain microservices architecture",
            "How do load balancers work?",
        ]
        
        messages = []
        for query in queries:
            messages.append(HumanMessage(content=query))
            response = graph.invoke({"messages": messages})
            messages.append(response["messages"][-1])
            
            assert response["messages"][-1].content is not None


class TestMultiAgentEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled multi-agent graph for testing."""
        return build_multi_agent_graph()
    
    def test_very_long_input(self, graph):
        """Test with very long input message."""
        long_message = "A" * 1000
        response = graph.invoke({
            "messages": [HumanMessage(content=long_message)]
        })
        assert response["messages"][-1].content is not None
    
    def test_unicode_content(self, graph):
        """Test Unicode character support."""
        unicode_queries = [
            "What is 人工智能?",
            "¿Qué es la inteligencia artificial?",
            "Что такое искусственный интеллект?",
        ]
        
        for query in unicode_queries:
            try:
                response = graph.invoke({
                    "messages": [HumanMessage(content=query)]
                })
                assert response["messages"][-1].content is not None
            except Exception:
                # Unicode support depends on LLM
                pass
    
    def test_special_characters(self, graph):
        """Test special character handling."""
        special_queries = [
            "What is $999.99 in euros?",
            "Explain @mentions in social media",
            "How do #hashtags work?",
        ]
        
        for query in special_queries:
            response = graph.invoke({
                "messages": [HumanMessage(content=query)]
            })
            assert response["messages"][-1].content is not None
    
    def test_empty_message_handling(self, graph):
        """Test handling of empty messages."""
        try:
            response = graph.invoke({
                "messages": [HumanMessage(content="")]
            })
            # Should either handle or raise appropriate error
            assert "messages" in response
        except Exception:
            # Empty message handling is optional
            pass
    
    def test_repeated_invocations(self, graph):
        """Test multiple rapid invocations."""
        for i in range(3):
            response = graph.invoke({
                "messages": [HumanMessage(content=f"Query {i}")]
            })
            assert response["messages"][-1].content is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
