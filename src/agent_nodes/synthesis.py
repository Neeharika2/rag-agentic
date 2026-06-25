from typing import Dict, Any

from .synthesis_utils import (
    generate_simulation_response,
    generate_strategy_response,
    generate_conflict_response,
    generate_web_response,
    generate_normal_response,
    generate_empty_fallback,
    _is_out_of_corpus,
)


def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Check for early-exit messages (e.g., empty DB guard)
    warnings = state.get("warnings")
    if warnings:
        return {
            "final_answer": warnings[0] if isinstance(warnings, list) else str(warnings)
        }

    query = state.get("query", "")
    query_lower = query.lower()

    is_simulation = state.get("is_simulation", False)
    is_strategy = state.get("is_strategy_query", False)
    conflict_detected = state.get("conflict_detected", False)
    conflict_details = state.get("conflict_details")

    retrieved_contexts = state.get("retrieved_contexts") or []
    has_web_context = any(
        doc.metadata and "tavily" in str(doc.metadata.get("section", "")).lower()
        for doc in retrieved_contexts
    )

    if is_simulation:
        return generate_simulation_response(state)

    if is_strategy:
        return generate_strategy_response(state)

    if conflict_detected and conflict_details:
        return generate_conflict_response(state, conflict_details)

    if has_web_context:
        return generate_web_response(query, retrieved_contexts)

    all_docs = list(retrieved_contexts)

    if not all_docs and _is_out_of_corpus(query_lower):
        return generate_empty_fallback(query)

    return generate_normal_response(state, all_docs)
