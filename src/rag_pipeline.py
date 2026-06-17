import os
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Import the dynamic nodes from src.nodes
# Import the dynamic nodes from src.nodes
from src.nodes import (
    router_node,
    eligibility_node,
    interview_prep_node,
    hiring_stats_node,
    overall_stats_node,
    trend_node,
    validation_node,
    synthesis_node,
    websearch_node
)

# 1. Placement Agent Shared State Schema
class PlacementAgentState(TypedDict):
    """
    PlacementAgentState defines the shared state schema for the placement RAG assistant,
    passing classification, normalized entities, retrieval contexts, and validation
    details between Graph nodes.
    """
    user_query: str                          # Raw input query
    query: Optional[str]                     # Backward-compatibility fallback
    
    # Classification & Entities (RouterNode)
    query_type: str                          # Intent: 'eligibility', 'interview_prep', 'hiring', 'statistics', 'trend', 'conflict', 'fallback'
    entities: List[str]                      # Normalized canonical company names (e.g. ['Amazon', 'TCS'])
    
    # Capability Contexts (populated by retrieval nodes)
    eligibility_context: List[Document]
    interview_context: List[Document]
    hiring_context: List[Document]
    stats_context: List[Document]
    trend_context: List[Document]
    websearch_context: List[Document]
    
    # Conflict & Verification Attributes
    conflict_detected: bool
    conflict_details: Optional[Dict[str, Any]]
    
    # Synthesis outputs
    final_answer: str                        # Final markdown response
    sources: List[str]                       # Extracted source section titles
    confidence: float                        # Safety/assurance score (0.0 to 1.0)

# 2. Dynamic Routing Logic
def route_query(state: PlacementAgentState) -> str:
    """
    Reads query_type from the state and routes to the appropriate RAG capability node.
    """
    q_type = state.get("query_type", "eligibility")
    
    if q_type == "eligibility":
        return "eligibility"
    elif q_type == "interview_prep":
        return "interview"
    elif q_type == "hiring":
        return "hiring"
    elif q_type == "statistics":
        return "statistics"
    elif q_type == "trend":
        return "trend"
    elif q_type == "fallback":
        return "websearch"
    else:
        # Default route for 'conflict', 'fallback', or other unknown query types
        # eligibility node is safe to execute as a general retrieval baseline
        return "eligibility"

# 3. Graph Compilation
def build_placement_graph():
    """
    Assembles and compiles the Placement Assistant LangGraph workflow.
    Wires up RouterNode -> Retrieval Nodes -> ValidationNode -> SynthesisNode.
    """
    # Initialize state graph with our customized state definition
    workflow = StateGraph(PlacementAgentState)
    
    # 1. Register all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("eligibility", eligibility_node)
    workflow.add_node("interview", interview_prep_node)
    workflow.add_node("hiring", hiring_stats_node)
    workflow.add_node("statistics", overall_stats_node)
    workflow.add_node("trend", trend_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("websearch", websearch_node)
    
    # 2. Set entry point
    workflow.set_entry_point("router")
    
    # 3. Add conditional edges mapping from Router
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "eligibility": "eligibility",
            "interview": "interview",
            "hiring": "hiring",
            "statistics": "statistics",
            "trend": "trend",
            "websearch": "websearch"
        }
    )
    
    # 4. Retrieval node standard edges (forward to Validation)
    workflow.add_edge("eligibility", "validation")
    workflow.add_edge("trend", "validation")
    
    # 5. Retrieval node join sequence (Interview -> Hiring -> Validation)
    workflow.add_edge("interview", "hiring")
    workflow.add_edge("hiring", "validation")
    
    # 6. Overall statistics bypasses validation straight to Synthesis
    workflow.add_edge("statistics", "synthesis")
    
    # 7. Web search bypasses validation straight to Synthesis
    workflow.add_edge("websearch", "synthesis")
    
    # 8. Final validation-to-synthesis flow
    workflow.add_edge("validation", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# 4. Backward Compatibility Wrapper
def build_rag_graph():
    """
    Wrapper mapping the legacy build_rag_graph call to the Compiled Placement Graph.
    """
    return build_placement_graph()
