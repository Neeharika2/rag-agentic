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
        
        has_sec = False
        for sec in matching_sections:
            sec_results = store.collection.get(where={"section": sec})
            docs = sec_results["documents"]
            metas = sec_results["metadatas"]
            for d, m in zip(docs, metas):
                matched_docs.append(Document(page_content=d, metadata=m))
                has_sec = True
                
        # If no specific technical_focus section was found for this company, fetch its section_1 profile
        if not has_sec:
            try:
                sec1_results = store.collection.get(where={"section": "section_1:_company_eligibility_profiles"})
                for d, m in zip(sec1_results["documents"], sec1_results["metadatas"]):
                    if company_clean in d.lower() or company_clean in m.get("company", "").lower():
                        matched_docs.append(Document(page_content=d, metadata=m))
            except Exception as e:
                print(f"[*] Warning: Could not retrieve fallback section_1 for {company}: {e}")
                
    # Fallback to similarity search if no specific sections matched
    if not matched_docs:
        raw_results = store.search(query, limit=10, filter_dict={"type": "tabular"})
        for r in raw_results:
            sec = r["metadata"].get("section", "")
            if "technical_focus" in sec or "section_1" in sec:
                matched_docs.append(Document(page_content=r["text"], metadata=r["metadata"]))
                
    # Proactively retrieve section_1 profiles matching the technical focus programming languages if requested
    query_lower = query.lower()
    languages = []
    if "python" in query_lower:
        languages.append("python")
    if "java" in query_lower:
        languages.append("java")
    if "c++" in query_lower or "cpp" in query_lower:
        languages.append("c++")
        
    if languages:
        try:
            sec1_results = store.collection.get(where={"section": "section_1:_company_eligibility_profiles"})
            for d, m in zip(sec1_results["documents"], sec1_results["metadatas"]):
                content_lower = d.lower()
                if any(lang in content_lower for lang in languages):
                    if not any(d == md.page_content for md in matched_docs):
                        matched_docs.append(Document(page_content=d, metadata=m))
        except Exception as e:
            print(f"[*] Warning: Could not retrieve language-focused section_1 profiles: {e}")
        
    return {"retrieved_contexts": matched_docs}
