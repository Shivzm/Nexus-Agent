"""LangGraph Agent CLI Application.

Command-line interface for running various LangGraph agents.
Supports multiple agent types with interactive conversation.
"""

import argparse
import logging
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from src.agents import (
    build_react_agent,
    chatbot_graph,
    human_in_loop_graph,
    build_multi_agent_graph,
    build_supervisor_graph,
)
from src.agents.base import get_memory_checkpointer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_agent(agent_type: str, model: str | None = None, checkpointer=None):
    """Build the specified agent graph.
    
    Args:
        agent_type: Type of agent ("react", "chatbot", "human_in_loop", "multi", "supervisor").
        model: Optional model override.
        checkpointer: Optional state persister.
    
    Returns:
        Compiled LangGraph agent.
    
    Raises:
        ValueError: If agent_type is not recognized.
    """
    if checkpointer is None:
        checkpointer = get_memory_checkpointer()
    
    builders = {
        "react": build_react_agent,
        "chatbot": chatbot_graph,
        "human_in_loop": human_in_loop_graph,
        "multi": build_multi_agent_graph,
        "supervisor": build_supervisor_graph,
    }
    
    if agent_type not in builders:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    builder = builders[agent_type]
    
    # Build with parameters based on builder signature
    try:
        return builder(model=model, checkpointer=checkpointer)
    except TypeError:
        # Fallback for agents that don't accept all parameters
        return builder(model=model) if model else builder()


def main():
    """Run the LangGraph Agent CLI."""
    parser = argparse.ArgumentParser(
        description="LangGraph Agent CLI - Interactive agent conversations"
    )
    parser.add_argument(
        "--agent",
        choices=["react", "chatbot", "human_in_loop", "multi", "supervisor"],
        default="react",
        help="Type of agent to run"
    )
    parser.add_argument(
        "--thread-id",
        default="1",
        help="Thread ID for state persistence"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Building {args.agent} agent...")
        graph = build_agent(args.agent, model=args.model)
        
        config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}
        logger.info(f"Starting {args.agent} agent with thread_id={args.thread_id}")
        
        print(f"\n{'='*60}")
        print(f"Running {args.agent.upper()} Agent")
        print(f"Thread ID: {args.thread_id}")
        print(f"Type 'quit' to exit\n")
        print(f"{'='*60}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == "quit":
                    logger.info("User quit the conversation")
                    print("\nThank you for using LangGraph Agent CLI!")
                    break
                
                if not user_input:
                    continue
                
                logger.debug(f"User input: {user_input}")
                response = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )
                
                agent_response = response["messages"][-1].content
                print(f"\nAgent: {agent_response}\n")
                logger.debug(f"Agent response length: {len(agent_response)}")
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                print("\n\nConversation interrupted.")
                break
            except Exception as e:
                logger.error(f"Error during inference: {e}", exc_info=True)
                print(f"Error: {str(e)}")
                print("Please try again.\n")
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())