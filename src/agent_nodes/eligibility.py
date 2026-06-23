import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import (
    normalize_company_name,
    get_canonical_companies,
    get_chroma_store,
    parse_cgpa_from_text,
    parse_backlogs_from_text,
    check_academic_eligibility
)
from .multihop_engine import MultiHopEngine

def eligibility_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    EligibilityNode: Handles CGPA, backlogs, bond, and package threshold filters.
    Reads 'user_query' (or 'query') and optional 'entities' from state,
    filters the database results in Python, and updates 'retrieved_contexts'.
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
    
    # If no entities in state, try to scan query for company mentions
    if not entities:
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            clean_c = c.replace(";", "")
            if clean_c.lower() in query.lower():
                entities.append(c)
                
    store = get_chroma_store()
    
    # Retrieve all eligibility profiles (only ~19-21 companies)
    results = store.collection.get(where={"section": "section_1:_company_eligibility_profiles"})
    docs = results["documents"]
    metas = results["metadatas"]
    
    # Parse filtering thresholds from the user query
    is_student_profile = any(x in query.lower() for x in ["student with", "i have", "my cgpa", "eligible with", "i've gpa"])
    cgpa_val = parse_cgpa_from_text(query)
    backlog_val = parse_backlogs_from_text(query)
    
    student_cgpa = None
    student_backlogs = None
    min_cgpa_above = None
    min_backlogs_at_least = None
    
    if is_student_profile:
        student_cgpa = cgpa_val
        student_backlogs = backlog_val
    else:
        if any(x in query.lower() for x in ["above", "greater than", ">"]):
            min_cgpa_above = cgpa_val
        if any(x in query.lower() for x in ["at least", "allow", "minimum"]):
            min_backlogs_at_least = backlog_val
                
    no_bond = any(x in query.lower() for x in ["no bond", "0 bond", "bond-free", "without bond", "bond free", "bond is 0"])
    
    filtered_records = []
    normalized_entities = [normalize_company_name(e) for e in entities] if entities else []
    
    for doc, meta in zip(docs, metas):
        if meta.get("type") != "tabular":
            continue
            
        r_company = meta.get("company", "")
        if normalized_entities:
            if normalize_company_name(r_company) not in normalized_entities:
                continue
                
        try:
            r_cgpa = float(meta.get("min_cgpa", 0.0))
            r_backlogs = int(meta.get("max_backlogs", 0))
            r_package = float(meta.get("package_(lpa)", 0.0))
            r_bond = int(meta.get("bond_(yrs)", 0))
        except (ValueError, TypeError):
            continue
            
        # Apply eligibility check helper
        if not check_academic_eligibility(
            student_cgpa, student_backlogs, r_cgpa, r_backlogs, no_bond, r_bond, min_cgpa_above, min_backlogs_at_least
        ):
            continue
            
        filtered_records.append((doc, meta, r_company, r_package))
        
    if any(x in query.lower() for x in ["maximum", "highest", "max", "best", "most pay", "top pay"]):
        filtered_records.sort(key=lambda x: x[3], reverse=True)
        
    summary_text = "Python Eligibility Filter Results:\n"
    if filtered_records:
        for idx, (doc, meta, comp, pkg) in enumerate(filtered_records):
            summary_text += f"{idx+1}. Company: {comp}, Cutoff CGPA: {meta.get('min_cgpa')}, Allowed Backlogs: {meta.get('max_backlogs')}, Package: {pkg} LPA, Bond: {meta.get('bond_(yrs)')} Yrs\n"
    else:
        summary_text += f"No company in this dataset has a CGPA cutoff <= {student_cgpa or cgpa_val or ''}.\n"
        
    summary_doc = Document(
        page_content=summary_text,
        metadata={"section": "section_1:_company_eligibility_profiles", "type": "python_summary"}
    )
    
    return_docs = [Document(page_content=d, metadata=m) for d, m, _, _ in filtered_records]
    return_docs.append(summary_doc)
    
    return {"retrieved_contexts": return_docs}
