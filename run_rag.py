import sys
from src.rag_pipeline import build_rag_graph

def main():
    """
    Main entry point for invoking the LangGraph RAG assistant.
    Compiles the graph and accepts search queries from CLI arguments,
    defaulting to a Virat Kohli query if none are supplied.
    """
    # 1. Compile the LangGraph workflow graph
    graph = build_rag_graph()
    
    # 2. Extract user query from command line parameters, or use a default test query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Tell me about Virat Kohli's performance in 2024"
        
    print("==================================================")
    print(f"[*] Running LangGraph RAG pipeline...")
    print(f"[*] Input Query: '{query}'")
    print("==================================================")
    
    # 3. Initialize state and execute the Compiled graph workflow
    initial_state = {"query": query}
    try:
        final_state = graph.invoke(initial_state)
        
        # 4. Print final output response
        print("\n================== FINAL ANSWER ==================")
        print(final_state.get("response", "No response generated."))
        print("==================================================\n")
    except Exception as e:
        print(f"\n[!] Error running the pipeline: {e}")
        print("Please check that your GOOGLE_API_KEY or GEMINI_API_KEY is configured correctly in your environment.\n")

if __name__ == "__main__":
    main()
