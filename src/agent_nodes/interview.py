import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store, get_section_all, retrieve_semantic
from .multihop_engine import MultiHopEngine

def interview_prep_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    InterviewPrepNode: Retrieves company selection rounds, topics, and programming focuses.
    Matches companies mentioned in state['entities'] or query text to technical focus sections.
    """
    query = state.get("user_query") or state.get("query") or ""
    entities = state.get("entities", [])
    
    mh_doc = MultiHopEngine.resolve_query(query)
    if mh_doc:
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            if c.lower() in mh_doc.page_content.lower() and c not in entities:
                entities.append(c)
        return {
            "retrieved_contexts": [mh_doc],
            "entities": entities
        }
    
    # Try to scan query for company mentions
    if not entities:
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            clean_c = c.replace(";", "")
            if clean_c.lower() in query.lower():
                entities.append(c)
                
    normalized_entities = list(set([normalize_company_name(e) for e in entities]))
    
    store = get_chroma_store()
    query_lower = query.lower()
    matched_docs = []
    
    # PRIMARY: Semantic vector search across technical_focus sections
    semantic_docs = retrieve_semantic(query, store, limit=20)
    for doc in semantic_docs:
        sec = doc.metadata.get("section", "")
        comp = doc.metadata.get("company", "").lower()
        if "technical_focus" in sec or "section_1" in sec:
            if not normalized_entities or any(nc in sec.lower() or nc in comp for nc in normalized_entities):
                if not any(doc.page_content == d.page_content for d in matched_docs):
                    matched_docs.append(doc)
    
    # SECONDARY: Exact section scans for companies not found via semantic search
    if normalized_entities:
        all_sec1 = get_section_all(store, "section_1:_company_eligibility_profiles")
        for company in normalized_entities:
            company_clean = company.lower().replace(";", "").replace(" r&d", "")
            for doc in all_sec1:
                meta = doc.metadata
                if company_clean in doc.page_content.lower() or company_clean in meta.get("company", "").lower():
                    if not any(doc.page_content == d.page_content for d in matched_docs):
                        matched_docs.append(doc)
    
    # Proactively retrieve section_1 profiles matching the technical focus programming languages if requested
    languages = []
    if "python" in query_lower:
        languages.append("python")
    if "java" in query_lower:
        languages.append("java")
    if "c++" in query_lower or "cpp" in query_lower:
        languages.append("c++")
        
    if languages:
        sec1_docs = get_section_all(store, "section_1:_company_eligibility_profiles")
        for doc in sec1_docs:
            if any(lang in doc.page_content.lower() for lang in languages):
                if not any(doc.page_content == d.page_content for d in matched_docs):
                    matched_docs.append(doc)
        
    return {"retrieved_contexts": matched_docs}
