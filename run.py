import argparse
import sys
import uvicorn
from src.rag_pipeline import build_rag_graph

def run_cli(query: str):
    """Compiles the LangGraph and executes a CLI query."""
    graph = build_rag_graph()
    
    print("==================================================")
    print(f"[*] Running LangGraph RAG pipeline...")
    print(f"[*] Input Query: '{query}'")
    print("==================================================")
    
    initial_state = {
        "user_query": query,
        "query": query
    }
    try:
        final_state = graph.invoke(initial_state)
        print("\n================== FINAL ANSWER ==================")
        print(final_state.get("final_answer") or final_state.get("response") or "No response generated.")
        print("==================================================\n")
    except Exception as e:
        print(f"\n[!] Error running the pipeline: {e}")
        print("Please check that your GOOGLE_API_KEY or GEMINI_API_KEY is configured correctly in your environment.\n")

def run_web():
    """Starts the FastAPI Web server using uvicorn."""
    print("==================================================")
    print("[*] Starting Placement Intelligence Assistant Web Server...")
    print("[*] Open the interface in your browser: http://127.0.0.1:8000")
    print("==================================================")
    uvicorn.run("src.web_app:app", host="127.0.0.1", port=8000, reload=True)

def main():
    parser = argparse.ArgumentParser(
        description="Placement Intelligence Assistant - Unified Runner"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--web", "-w",
        action="store_true",
        help="Start the FastAPI web server (default behavior)"
    )
    group.add_argument(
        "--cli", "-c",
        action="store_true",
        help="Run in command-line interface (CLI) mode"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query to run in CLI mode. If provided, CLI mode is active automatically."
    )
    
    args = parser.parse_args()
    
    # Decide the execution mode based on arguments
    if args.web:
        run_web()
    elif args.query:
        query_str = " ".join(args.query)
        run_cli(query_str)
    elif args.cli:
        default_query = "Which company has the highest package?"
        run_cli(default_query)
    else:
        run_web()

if __name__ == "__main__":
    main()
