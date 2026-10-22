"""Search and summarization tools for LangGraph agents.

Provides tools for web search and content summarization.
"""

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from functools import lru_cache

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Tavily search once (singleton pattern)
_summary_length = 500

@lru_cache(maxsize=1)
def _get_tavily() -> TavilySearchResults:
    """Create the Tavily client on first use (needs TAVILY_API_KEY)."""
    return TavilySearchResults(max_results=3)

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic.
    
    Uses Tavily search to find relevant information on the internet.
    Returns up to 3 relevant results.
    
    Args:
        query: The search query or topic to search for.
    
    Returns:
        Formatted string containing search results.
    
    Raises:
        Exception: If the search fails or returns no results.
    """
    try:
        results = _get_tavily().invoke(query)
        if not results:
            return "No results found for the query."
        return str(results)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def write_summary(content: str) -> str:
    """Create a summary of provided content.
    
    Generates a concise summary by extracting key portions
    of the provided content.
    
    Args:
        content: The content to summarize.
    
    Returns:
        A formatted summary of the content.
    """
    if not content:
        return "No content to summarize."
    
    # Truncate to reasonable length and format
    truncated = content[:_summary_length]
    summary = f"Summary of findings:\n\n{truncated}..."
    return summary
