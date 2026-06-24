import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src.agent_nodes.router import router_node
from src.agent_nodes.validation import validation_node
from src.agent_nodes.synthesis import synthesis_node
from src.agent_nodes.profile_builder import profile_builder_node
from src.agent_nodes.opportunity_detector import opportunity_detector_node
from src.agent_nodes.probability_estimator import probability_estimator_node

from src.agent_nodes.eligibility import eligibility_node
from src.agent_nodes.interview import interview_prep_node
from src.agent_nodes.hiring import hiring_stats_node
from src.agent_nodes.stats import overall_stats_node
from src.agent_nodes.trend import trend_node
from src.agent_nodes.web_search import websearch_node


def reduce_contexts(left: List[Document], right: List[Document]) -> List[Document]:
    """Merges lists of documents, avoiding duplicates by content and section."""
    if left is None:
        left = []
    if right is None:
        right = []
    seen = set()
    merged = []
    for doc in left + right:
        doc_id = (doc.page_content, doc.metadata.get("section", ""))
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append(doc)
    return merged


class PlacementAgentState(TypedDict):
    user_query: str
    query: Optional[str]

    query_type: str
    entities: List[str]

    retrieved_contexts: Annotated[List[Document], reduce_contexts]

    conflict_detected: bool
    conflict_details: Optional[Dict[str, Any]]

    final_answer: str
    sources: List[str]
    confidence: float

    student_profile: Optional[Dict[str, Any]]
    baseline_profile: Optional[Dict[str, Any]]
    is_simulation: Optional[bool]
    opportunities: Optional[List[Dict[str, Any]]]
    is_strategy_query: Optional[bool]
    gaps: Optional[List[Dict[str, Any]]]
    probability_scores: Optional[Dict[str, float]]
    strategy_plan: Optional[Dict[str, Any]]
    warnings: Optional[List[str]]

    _needs_validation: bool


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
        return "eligibility"


def route_after_profile_builder(state: PlacementAgentState) -> str:
    if state.get("is_strategy_query"):
        return "opportunity"
    return "router"


def build_placement_graph():
    workflow = StateGraph(PlacementAgentState)

    # Register all nodes
    workflow.add_node("profile_builder", profile_builder_node)
    workflow.add_node("opportunity", opportunity_detector_node)
    workflow.add_node("probability_estimator", probability_estimator_node)
    workflow.add_node("router", router_node)
    workflow.add_node("eligibility", eligibility_node)
    workflow.add_node("interview", interview_prep_node)
    workflow.add_node("hiring", hiring_stats_node)
    workflow.add_node("statistics", overall_stats_node)
    workflow.add_node("trend", trend_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("websearch", websearch_node)

    # Set entry point
    workflow.set_entry_point("profile_builder")

    # Add conditional edges from profile_builder
    workflow.add_conditional_edges(
        "profile_builder",
        route_after_profile_builder,
        {"opportunity": "opportunity", "router": "router"}
    )

    # Add conditional edges from router
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

    workflow.add_edge("opportunity", "probability_estimator")
    workflow.add_edge("probability_estimator", "synthesis")
    
    # Retrieval node standard edges
    workflow.add_edge("eligibility", "validation")
    workflow.add_edge("trend", "validation")
    
    # Sequence/Join: Interview -> Hiring -> Validation
    workflow.add_edge("interview", "hiring")
    workflow.add_edge("hiring", "validation")
    
    # Direct Synthesis flow
    workflow.add_edge("statistics", "synthesis")
    workflow.add_edge("websearch", "synthesis")

    workflow.add_edge("validation", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
