from typing import Dict, Any
from .web_search import tavily_search

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    WebsearchNode: Runs when query intent is 'fallback' (out of corpus / general).
    Calls Tavily Search API and saves context in 'websearch_context'.
    """
    query = state.get("user_query") or state.get("query") or ""
    print(f"[*] Running Web Search via Tavily for query: '{query}'")
    web_docs = tavily_search(query)
    return {
        "websearch_context": web_docs
    }
