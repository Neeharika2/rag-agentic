import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store, get_section_cached
from .multihop_engine import MultiHopEngine

def hiring_stats_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    HiringStatsNode: Manages queries about numbers of SDE, Analyst, Intern, or Officer roles.
    Integrates with interview prep focus data to execute multi-column joins (e.g. Python-focused intern counts).
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
    
    store = get_chroma_store()
    
    # 3. Retrieve and parse hiring data to dynamically extract available roles
    hiring_results = get_section_cached(store, "hiring_distribution_data_table_(text_representation_of_all_charts_above)")
    docs = hiring_results["documents"]
    metas = hiring_results["metadatas"]
 
    # 1. Detect target hiring role dynamically based on metadata keys
    role = None
    available_roles = set()
    for meta in metas:
        for key in meta.keys():
            if key not in ["company", "section", "type"]:
                available_roles.add(key.lower())
                
    for r in available_roles:
        if r in query.lower():
            role = r
            break
            
    # Fallback synonym matching for common role terms if no direct match found
    if not role:
        if "software" in query.lower() or "developer" in query.lower():
            if "sde" in available_roles:
                role = "sde"
        elif "internship" in query.lower():
            if "intern" in available_roles:
                role = "intern"
        
    # 2. Check if join query (e.g. tech-focused company join query)
    # Collect candidate tech focus terms dynamically from the metadata
    tech_candidates = set()
    elig_results = get_section_cached(store, "section_1:_company_eligibility_profiles")
    for meta in elig_results.get("metadatas", []):
        for field in ["tech_focus", "key_topics"]:
            val = meta.get(field)
            if val:
                tech_candidates.add(val.strip().lower())
                parts = re.split(r'[,/;]', val)
                for part in parts:
                    p_clean = part.strip().lower()
                    if p_clean:
                        tech_candidates.add(p_clean)
 
    target_tech = None
    sorted_candidates = sorted(list(tech_candidates), key=len, reverse=True)
    for cand in sorted_candidates:
        if cand in ["dsa", "aptitude"]:  # skip generic terms to avoid false joins
            continue
        pattern = r'\b' + re.escape(cand) + r'\b'
        if cand == "c++":
            pattern = r'(?:^|\s)c\+\+(?:\s|$|\b|[.,;])'
        if re.search(pattern, query.lower()):
            target_tech = cand
            break
 
    tech_companies = []
    if target_tech:
        # Dynamic check in eligibility profiles for tech focus or key topics
        for meta in elig_results.get("metadatas", []):
            company = meta.get("company", "")
            tech_focus = (meta.get("tech_focus") or "").lower()
            key_topics = (meta.get("key_topics") or "").lower()
            if target_tech in tech_focus or target_tech in key_topics:
                tech_companies.append(normalize_company_name(company))
                
        # Also check sections with the target tech focus
        results = store.collection.get(include=["metadatas"])
        all_sections = list(set([m["section"] for m in results["metadatas"] if "section" in m]))
        canonical_companies = get_canonical_companies()
        for sec in all_sections:
            if sec.startswith("n_") and "technical_focus" in sec and target_tech in sec.lower():
                for c in canonical_companies:
                    if c.lower() in sec.lower():
                        tech_companies.append(normalize_company_name(c))
                        
        # Check retrieved_contexts if present to find companies associated with the tech focus
        retrieved_contexts = state.get("retrieved_contexts", [])
        for doc in retrieved_contexts:
            meta = doc.metadata
            if meta:
                comp = meta.get("company")
                sec = meta.get("section", "").lower()
                if comp and (target_tech in sec or target_tech in doc.page_content.lower()):
                    tech_companies.append(normalize_company_name(comp))
                        
        tech_companies = list(set(tech_companies))
    
    hiring_records = []
    for doc, meta in zip(docs, metas):
        if meta.get("type") != "tabular":
            continue
        comp = meta.get("company", "")
        norm_comp = normalize_company_name(comp)
        
        # Apply tech-focused filter if join query
        if target_tech and norm_comp not in tech_companies:
            continue
            
        hiring_records.append((doc, meta, norm_comp))
        
    # Sort by specific role count if requested
    role_count_records = []
    if role:
        for doc, meta, norm_comp in hiring_records:
            count_val = meta.get(role, "0")
            try:
                count_int = int(count_val)
            except ValueError:
                count_int = 0
            role_count_records.append((doc, meta, norm_comp, count_int))
            
        role_count_records.sort(key=lambda x: x[3], reverse=True)
        
    # Build synthetic result document
    summary_text = "Python Hiring Distribution Analysis:\n"
    if role:
        summary_text += f"Ranked companies by hiring numbers for role '{role.upper()}':\n"
        for idx, (doc, meta, comp, count) in enumerate(role_count_records):
            summary_text += f"{idx+1}. Company: {comp}, Hires: {count} (Details: {doc})\n"
    else:
        summary_text += "Hiring numbers for all roles:\n"
        for idx, (doc, meta, comp) in enumerate(hiring_records):
            summary_text += f"- {comp}: {doc}\n"
            
    summary_doc = Document(
        page_content=summary_text,
        metadata={"section": "hiring_distribution_data_table_(text_representation_of_all_charts_above)", "type": "python_summary"}
    )
    
    # Compile outputs
    return_docs = [Document(page_content=d, metadata=m) for d, m, _ in hiring_records]
    return_docs.append(summary_doc)
    
    return {"retrieved_contexts": return_docs}
