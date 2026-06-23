import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from src.rag_pipeline import build_placement_graph

# Initialize FastAPI application
app = FastAPI(
    title="Placement Intelligence Assistant API",
    description="Backend API powering the LangGraph RAG Placement assistant",
    version="1.0.0"
)

from typing import List, Dict, Any, Optional

# Define request schema
class StudentProfileSchema(BaseModel):
    cgpa: Optional[float] = None
    skills: List[str] = []
    weaknesses: List[str] = []
    interests: List[str] = []
    backlogs: int = 0
    projects_count: int = 0

class QueryRequest(BaseModel):
    query: str
    student_profile: Optional[StudentProfileSchema] = None

# Define response schemas
class DocumentResponse(BaseModel):
    text: str
    section: str

class CompanyOpportunity(BaseModel):
    company: str
    package: float
    min_cgpa: float
    max_backlogs: int
    tech_focus: str
    bond: int
    skill_score: float

class QueryResponse(BaseModel):
    response: str
    documents: List[DocumentResponse]
    student_profile: Optional[Dict[str, Any]] = None
    opportunities: Optional[List[CompanyOpportunity]] = None
    is_strategy_query: Optional[bool] = None

# Compile the LangGraph pipeline workflow once at startup
rag_graph = build_placement_graph()

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
        # Construct graph initial state supporting both user_query and backward-compatible query key
        initial_state = {
            "user_query": query_str,
            "query": query_str
        }
        
        if request.student_profile:
            initial_state["student_profile"] = request.student_profile.dict()
        
        # Invoke the state graph synchronously
        final_state = rag_graph.invoke(initial_state)
        
        # Format list of retrieved document chunks for response JSON
        docs_response = []
        # Collect retrieved documents across all active capability context lists
        all_docs = final_state.get("retrieved_contexts") or []
                
        for doc in all_docs:
            docs_response.append(DocumentResponse(
                text=doc.page_content,
                section=doc.metadata.get("section", "general")
            ))
            
        # Get final answer or backward-compatible response field
        response_text = final_state.get("final_answer") or final_state.get("response") or "No response generated."
            
        return QueryResponse(
            response=response_text,
            documents=docs_response,
            student_profile=final_state.get("student_profile"),
            opportunities=final_state.get("opportunities"),
            is_strategy_query=final_state.get("is_strategy_query")
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
