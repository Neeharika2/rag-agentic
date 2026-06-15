import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from src.rag_pipeline import build_rag_graph

# Initialize FastAPI application
app = FastAPI(
    title="Placement Intelligence Assistant API",
    description="Backend API powering the LangGraph RAG Placement assistant",
    version="1.0.0"
)

# Define request schema
class QueryRequest(BaseModel):
    query: str

# Define response schemas
class DocumentResponse(BaseModel):
    text: str
    section: str

class QueryResponse(BaseModel):
    response: str
    documents: List[DocumentResponse]

# Compile the LangGraph pipeline workflow once at startup
rag_graph = build_rag_graph()

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Executes the user query through the LangGraph RAG assistant.
    Retrieves matching ChromaDB document chunks and generates an answer using Gemini.
    """
    query_str = request.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    try:
        # Construct graph initial state
        initial_state = {"query": query_str}
        
        # Invoke the state graph synchronously
        final_state = rag_graph.invoke(initial_state)
        
        # Format list of retrieved document chunks for response JSON
        docs_response = []
        for doc in final_state.get("documents", []):
            docs_response.append(DocumentResponse(
                text=doc.page_content,
                section=doc.metadata.get("section", "general")
            ))
            
        return QueryResponse(
            response=final_state.get("response", "No response generated."),
            documents=docs_response
        )
    except Exception as e:
        # Return internal error details if execution fails
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

# Mount static files folder to serve static index.html on root `/`
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_frontend():
    """Serves the main single-page index.html file on the root URL."""
    html_file = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="index.html template not found")
