import os
import logging
from time import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "backend.log")

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(file_handler)

from src.rag_pipeline import build_placement_graph

# Initialize FastAPI application
app = FastAPI(
    title="Placement Intelligence Assistant API",
    description="Backend API powering the LangGraph RAG Placement assistant",
    version="1.0.0"
)

MAX_QUERY_LENGTH = 2000


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Placement Intelligence Assistant"}

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
    readiness_score: Optional[float] = None
    calibration_reason: Optional[str] = None
    baseline_readiness_score: Optional[float] = None
    readiness_increase: Optional[float] = None

class QueryResponse(BaseModel):
    response: str
    documents: List[DocumentResponse]
    student_profile: Optional[Dict[str, Any]] = None
    opportunities: Optional[List[CompanyOpportunity]] = None
    is_strategy_query: Optional[bool] = None
    is_simulation: Optional[bool] = None
    history_logs: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[str]] = None
    strategy_plan: Optional[Dict[str, Any]] = None

# Compile the LangGraph pipeline workflow once at startup
rag_graph = build_placement_graph()

@app.get("/api/history")
async def get_history_endpoint():
    """
    Retrieves the saved progress tracking history logs.
    """
    logger.info("History logs requested via /api/history")
    try:
        from src.agent_nodes.history_tracker import get_history
        return {"history_logs": get_history()}
    except Exception as e:
        logger.error(f"Failed to retrieve history logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear_history")
async def clear_history_endpoint():
    """
    Clears all saved progress tracking logs in history_tracker.
    """
    logger.info("History logs clearing requested via /api/clear_history")
    try:
        from src.agent_nodes.history_tracker import clear_history
        clear_history()
        logger.info("Successfully cleared history logs.")
        return {"status": "success", "message": "History cleared"}
    except Exception as e:
        logger.error(f"Failed to clear history logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Executes the user query through the LangGraph RAG assistant.
    Retrieves matching ChromaDB document chunks and generates an answer using Gemini.
    """
    query_str = request.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
    if len(query_str) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"Query string exceeds maximum length of {MAX_QUERY_LENGTH} characters")
        
    start_time = time()
    logger.info(f"Incoming Request - Query: '{query_str}', Profile Provided: {request.student_profile is not None}")
    if request.student_profile:
        logger.info(f"Student Profile Details: {request.student_profile.dict()}")
        
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
            
        # Retrieve history logs
        from src.agent_nodes.history_tracker import get_history
        history_logs = get_history()

        elapsed = time() - start_time
        logger.info(f"Successful Execution - Query: '{query_str}' - Took: {elapsed:.2f}s - "
                    f"Conflict Detected: {final_state.get('conflict_detected', False)} - "
                    f"Is Simulation: {final_state.get('is_simulation', False)} - "
                    f"Is Strategy: {final_state.get('is_strategy_query', False)}")

        return QueryResponse(
            response=response_text,
            documents=docs_response,
            student_profile=final_state.get("student_profile"),
            opportunities=final_state.get("opportunities"),
            is_strategy_query=final_state.get("is_strategy_query"),
            is_simulation=final_state.get("is_simulation"),
            history_logs=history_logs,
            warnings=final_state.get("warnings"),
            strategy_plan=final_state.get("strategy_plan")
        )

    except Exception as e:
        elapsed = time() - start_time
        logger.error(f"Execution Failed - Query: '{query_str}' - Took: {elapsed:.2f}s - Error: {e}", exc_info=True)
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
