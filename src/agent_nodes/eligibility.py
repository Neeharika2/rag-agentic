import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store
from .multihop_engine import MultiHopEngine

def eligibility_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    EligibilityNode: Handles CGPA, backlogs, bond, and package threshold filters.
    Reads 'user_query' (or 'query') and optional 'entities' from state,
    filters the database results in Python, and updates 'eligibility_context'.
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
            "eligibility_context": [mh_doc],
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
    student_cgpa = None
    student_backlogs = None
    min_cgpa_above = None
    min_backlogs_at_least = None
    no_bond = False
    
    is_student_profile = any(x in query.lower() for x in ["student with", "i have", "my cgpa", "eligible with", "i've gpa"])
    
    # Clean query by removing anything followed by LPA or Lakhs to avoid package/CGPA confusion
    clean_query = re.sub(r"\d+(?:\.\d+)?\s*(?:lpa|lakh|lakhs)", "", query, flags=re.IGNORECASE)
    
    # Extract CGPA candidates from the clean query (between 5.0 and 10.0)
    cgpa_floats = [float(x) for x in re.findall(r"\b\d+\.\d+\b", clean_query)]
    cgpa_ints = [float(x) for x in re.findall(r"\b[56789]\b|\b10\b", clean_query)]
    cgpa_candidates = [c for c in (cgpa_floats + cgpa_ints) if 5.0 <= c <= 10.0]
    
    # Extract backlog integers (number directly before "backlog" or "backlogs")
    backlog_match = re.search(r"(\d+)\s*(?:active\s*)?backlog", query, re.IGNORECASE)
    backlog_val = int(backlog_match.group(1)) if backlog_match else None
    
    if is_student_profile:
        if cgpa_candidates:
            student_cgpa = cgpa_candidates[0]
        if backlog_val is not None:
            student_backlogs = backlog_val
    else:
        # e.g., "CGPA above 8.0" or "CGPA > 8.0"
        if any(x in query.lower() for x in ["above", "greater than", ">"]):
            if cgpa_candidates:
                min_cgpa_above = cgpa_candidates[0]
        # e.g., "allow at least 2 backlogs"
        if any(x in query.lower() for x in ["at least", "allow", "minimum"]):
            if backlog_val is not None:
                min_backlogs_at_least = backlog_val
                
    no_bond = any(x in query.lower() for x in ["no bond", "0 bond", "bond-free", "without bond", "bond free", "bond is 0"])
    
    filtered_records = []
    
    # Normalise entities list if present
    normalized_entities = [normalize_company_name(e) for e in entities] if entities else []
    
    for doc, meta in zip(docs, metas):
        if meta.get("type") != "tabular":
            continue
            
        r_company = meta.get("company", "")
        # Fuzzy match company filter
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
            
        # Apply profile-based filters (Student's attributes vs Company cutoffs)
        if student_cgpa is not None and r_cgpa > student_cgpa:
            continue
        if student_backlogs is not None and r_backlogs < student_backlogs:
            continue
        if no_bond and r_bond > 0:
            continue
            
        # Apply metric-specific filters
        if min_cgpa_above is not None and r_cgpa <= min_cgpa_above:
            continue
        if min_backlogs_at_least is not None and r_backlogs < min_backlogs_at_least:
            continue
            
        filtered_records.append((doc, meta, r_company, r_package))
        
    # Sort by package if query requests max/highest pay
    if any(x in query.lower() for x in ["maximum", "highest", "max", "best", "most pay", "top pay"]):
        filtered_records.sort(key=lambda x: x[3], reverse=True)
        
    # Build synthetic document representing processed results
    summary_text = "Python Eligibility Filter Results:\n"
    if filtered_records:
        for idx, (doc, meta, comp, pkg) in enumerate(filtered_records):
            summary_text += f"{idx+1}. Company: {comp}, Cutoff CGPA: {meta.get('min_cgpa')}, Allowed Backlogs: {meta.get('max_backlogs')}, Package: {pkg} LPA, Bond: {meta.get('bond_(yrs)')} Yrs\n"
    else:
        summary_text += "No matching companies found in dataset for target criteria.\n"
        
    summary_doc = Document(
        page_content=summary_text,
        metadata={"section": "section_1:_company_eligibility_profiles", "type": "python_summary"}
    )
    
    # Combine original matching docs and the summary doc
    return_docs = [Document(page_content=d, metadata=m) for d, m, _, _ in filtered_records]
    return_docs.append(summary_doc)
    
    return {"eligibility_context": return_docs}
