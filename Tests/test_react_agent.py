from unittest.mock import patch

from src.agents import build_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

def test_react_agent_builds():
    graph = build_react_agent()
    assert graph is not None

def test_react_agent_uses_model_override():
    with patch("src.agents.react_agent.get_llm") as get_llm:
        get_llm.return_value.bind_tools.return_value = get_llm.return_value
        build_react_agent(model="groq:test-model")
    get_llm.assert_called_once_with("groq:test-model")

def test_react_agent_responds():
    graph = build_react_agent()
    config: RunnableConfig = {"configurable": {"thread_id": "test"}}
    result = graph.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]}, config=config)
    assert len(result["messages"]) > 0