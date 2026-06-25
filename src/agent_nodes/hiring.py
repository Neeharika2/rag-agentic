import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store, get_section_all, retrieve_semantic
from .multihop_engine import MultiHopEngine

def hiring_stats_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    HiringStatsNode: Manages queries about numbers of SDE, Analyst, Intern, or Officer roles.
    Integrates with interview prep focus data to execute multi-column joins (e.g. Python-focused intern counts).
    """
    query = state.get("query", "")
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
    
    # PRIMARY: Semantic retrieval of hiring data
    hiring_docs = retrieve_semantic(query, store, section="hiring_distribution_data_table_(text_representation_of_all_charts_above)", limit=30)
    
    # Fallback: exact section scan if semantic search yields nothing (e.g. listing queries)
    if not hiring_docs:
        hiring_docs = get_section_all(store, "hiring_distribution_data_table_(text_representation_of_all_charts_above)")
    
    # Detect available hiring roles from metadata
    role = None
    available_roles = set()
    for doc in hiring_docs:
        for key in doc.metadata.keys():
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
        
    # 2. Check if join query (e.g. tech-focused company + hiring)
    # Collect candidate tech focus terms dynamically from eligibility metadata
    tech_candidates = set()
    elig_docs = get_section_all(store, "section_1:_company_eligibility_profiles")
    for doc in elig_docs:
        for field in ["tech_focus", "key_topics"]:
            val = doc.metadata.get(field)
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
        if cand in ["dsa", "aptitude"]:
            continue
        pattern = r'\b' + re.escape(cand) + r'\b'
        if cand == "c++":
            pattern = r'(?:^|\s)c\+\+(?:\s|$|\b|[.,;])'
        if re.search(pattern, query.lower()):
            target_tech = cand
            break
 
    tech_companies = []
    if target_tech:
        for doc in elig_docs:
            company = doc.metadata.get("company", "")
            tech_focus = (doc.metadata.get("tech_focus") or "").lower()
            key_topics = (doc.metadata.get("key_topics") or "").lower()
            if target_tech in tech_focus or target_tech in key_topics:
                tech_companies.append(normalize_company_name(company))
        
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            c_lower = c.lower()
            # Check if this company appears alongside the tech focus
            semantic_matches = retrieve_semantic(f"{c} {target_tech}", store, limit=5)
            for sd in semantic_matches:
                sec = sd.metadata.get("section", "").lower()
                if target_tech in sec or target_tech in sd.page_content.lower():
                    tech_companies.append(normalize_company_name(c))
                        
        tech_companies = list(set(tech_companies))
    
    hiring_records = []
    for doc in hiring_docs:
        meta = doc.metadata
        if meta.get("type") != "tabular":
            continue
        comp = meta.get("company", "")
        norm_comp = normalize_company_name(comp)
        
        if target_tech and norm_comp not in tech_companies:
            continue
            
        hiring_records.append((doc.page_content, meta, norm_comp))
        
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
