import os
import logging
import requests
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def tavily_search(query: str) -> List[Document]:
    """
    Performs web search using Tavily API.
    Queries the Tavily search endpoint and converts the results into Document objects.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not found in environment. Skipping web search.")
        return []
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        res_json = response.json()
        
        docs = []
        # If Tavily generated a quick answer summary, include it
        tavily_ans = res_json.get("answer")
        if tavily_ans:
            docs.append(Document(
                page_content=f"Tavily Summary Answer: {tavily_ans}",
                metadata={"section": "tavily_web_search_summary", "source": "tavily"}
            ))
            
        results = res_json.get("results", [])
        for r in results:
            title = r.get("title", "Web Page")
            url_str = r.get("url", "")
            content = r.get("content", "")
            if content:
                docs.append(Document(
                    page_content=f"Source: {title} ({url_str})\nContent: {content}",
                    metadata={"section": "tavily_web_search", "source": url_str}
                ))
        logger.info("Tavily web search retrieved %d document chunks.", len(docs))
        return docs
    except Exception as e:
        logger.error("Error during Tavily Search: %s", e)
        return []

from typing import Dict, Any

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    WebsearchNode: Runs when query intent is 'fallback' (out of corpus / general).
    Calls Tavily Search API and saves context in 'retrieved_contexts'.
    """
    query = state.get("query", "")
    logger.info("Running Web Search via Tavily for query: '%s'", query)
    web_docs = tavily_search(query)
    return {
        "retrieved_contexts": web_docs
    }
