import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store
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
    
    # Fetch all section names from db
    results = store.collection.get(include=["metadatas"])
    metadatas = results["metadatas"]
    all_sections = list(set([m["section"] for m in metadatas if "section" in m]))
    
    matched_docs = []
    for company in normalized_entities:
        company_clean = company.lower().replace(";", "").replace(" r&d", "")
        # Find sections matching: n_<company>_|_technical_focus_...
        matching_sections = [
            sec for sec in all_sections 
            if sec.startswith("n_") and "technical_focus" in sec and company_clean in sec.lower()
        ]
        
        for sec in matching_sections:
            sec_results = store.collection.get(where={"section": sec})
            docs = sec_results["documents"]
            metas = sec_results["metadatas"]
            for d, m in zip(docs, metas):
                matched_docs.append(Document(page_content=d, metadata=m))
                
    # Fallback to similarity search if no specific sections matched
    if not matched_docs:
        raw_results = store.search(query, limit=5, filter_dict={"type": "tabular"})
        # Filter raw results to matching sections if any
        matched_docs = [
            Document(page_content=r["text"], metadata=r["metadata"]) 
            for r in raw_results 
            if "technical_focus" in r["metadata"].get("section", "")
        ]
        
    return {"retrieved_contexts": matched_docs}
