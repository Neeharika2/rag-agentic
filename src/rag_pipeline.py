import os
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from src.vectorstore.chroma_store import ChromaStore

# 1. Shared Graph State Definition
class RAGState(TypedDict):
    """
    State definition for the RAG pipeline.
    Passed between nodes in the LangGraph execution flow.
    """
    query: str               # The user's input query string
    documents: List[Document]# List of retrieved document chunks relevant to the query
    response: str            # The final generated answer string

# 2. ChromaDB Retriever
def chroma_retrieve(query: str, limit: int = 3) -> List[Document]:
    """
    Queries the local ChromaDB database for matches.

    Parameters:
    - query (str): The search query.
    - limit (int): The maximum number of documents to return.

    Returns:
    - List[Document]: A list of matching documents with text and metadata.
    """
    try:
        # Locate the local chroma_db folder relative to the workspace base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = os.path.join(base_dir, "chroma_db")
        
        # Only query if database folder exists
        if os.path.exists(persist_dir):
            store = ChromaStore(persist_dir=persist_dir)
            results = store.search(query, limit=limit)
            if results:
                # Convert the raw database dict result into standard LangChain Document objects
                return [
                    Document(page_content=r["text"], metadata=r["metadata"])
                    for r in results
                ]
    except Exception as e:
        # Catch exceptions (e.g. schema changes or directory locked) and log
        print(f"[*] Info: Chroma retrieval check failed or DB empty: {e}")
    return []

# 3. LangGraph Nodes
def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Retrieve Node:
    Extracts the query from the state, calls the ChromaDB search engine,
    and updates the state with the list of retrieved documents.

    Parameters:
    - state (RAGState): The current state of the execution graph.

    Returns:
    - Dict[str, Any]: A dictionary updating the 'documents' field in state.
    """
    query = state["query"]
    print(f"\n--- [Node: Retrieve] ---")
    print(f"Query: '{query}'")
    
    # Query ChromaDB for relevant document chunks
    docs = chroma_retrieve(query)
    print(f"[*] Retrieved {len(docs)} matching documents from ChromaDB.")
        
    # Print a snippet of each retrieved document for trace debugging
    for i, doc in enumerate(docs):
        source = doc.metadata.get('section', 'general')
        print(f"  Document {i+1} (Source: {source}): {doc.page_content[:90]}...")
        
    return {"documents": docs}

def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Generate Node:
    Constructs a RAG context, compiles a chat prompt, and invokes
    the Gemini model to generate a response. Includes robust fallback
    handling if Google API keys are missing or quota is exceeded.

    Parameters:
    - state (RAGState): The current state of the execution graph.

    Returns:
    - Dict[str, Any]: A dictionary updating the 'response' field in state.
    """
    query = state["query"]
    documents = state["documents"]
    print(f"\n--- [Node: Generate] ---")
    
    # Concatenate the content of all retrieved documents to build the RAG context block
    context_str = "\n\n".join([
        f"Document {i+1} [Section: {doc.metadata.get('section', 'general')}]:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    ])
    
    # Define a clean instruction prompt for the IPL Assistant
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert IPL Assistant. Answer the user query using the provided context context below. "
            "If you cannot answer from the context, rely on your internal knowledge but clearly state that "
            "the source is not in the provided context documents.\n\n"
            "Context:\n{context}"
        )),
        ("user", "{query}")
    ])
    
    try:
        # Initialize Gemini LLM (gemini-1.5-flash).
        # Requires GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Combine prompt and LLM using LCEL (LangChain Expression Language)
        chain = prompt_template | llm
        response = chain.invoke({"context": context_str, "query": query})
        answer = response.content
    except Exception as e:
        # Gracefully handle API failures (such as quota exceeded, network timeouts, or missing keys)
        print(f"[*] Gemini API call failed: {e}")
        print("[*] Falling back to rule-based offline generator...")
        
        # Construct an offline fallback answer listing the matching retrieved text directly
        summary_bullets = []
        for i, doc in enumerate(documents):
            sect = doc.metadata.get("section", "general").upper()
            summary_bullets.append(f"- [{sect}] {doc.page_content}")
            
        answer = (
            f"**[Offline Fallback Answer]**\n"
            f"The Gemini API request failed (e.g., due to API limits, missing key, or quota). "
            f"Here is the retrieved context related to your query '{query}':\n\n" +
            "\n".join(summary_bullets)
        )
    
    print(f"[*] Generated RAG Answer:\n{answer}")
    return {"response": answer}

# 4. Graph Assembly & Compilation
def build_rag_graph():
    """
    Assembles and compiles the LangGraph StateGraph workflow.
    Configures retrieve -> generate sequential edge connections.

    Returns:
    - CompiledGraph: The compiled state graph ready to be invoked.
    """
    # Create the state graph workspace with our custom RAGState
    workflow = StateGraph(RAGState)
    
    # Register the nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Configure graph entry point
    workflow.set_entry_point("retrieve")
    
    # Add sequential transitions: retrieve -> generate -> END
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile the graph workflow
    return workflow.compile()
