import os
import requests
from typing import List
from langchain_core.documents import Document

def tavily_search(query: str) -> List[Document]:
    """
    Performs web search using Tavily API.
    Queries the Tavily search endpoint and converts the results into Document objects.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[*] Warning: TAVILY_API_KEY not found in environment. Skipping web search.")
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
        print(f"[*] Tavily web search retrieved {len(docs)} document chunks.")
        return docs
    except Exception as e:
        print(f"[*] Error during Tavily Search: {e}")
        return []

from typing import Dict, Any

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    WebsearchNode: Runs when query intent is 'fallback' (out of corpus / general).
    Calls Tavily Search API and saves context in 'retrieved_contexts'.
    """
    query = state.get("user_query") or state.get("query") or ""
    print(f"[*] Running Web Search via Tavily for query: '{query}'")
    web_docs = tavily_search(query)
    return {
        "retrieved_contexts": web_docs
    }
