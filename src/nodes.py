import os
import re
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from src.vectorstore.chroma_store import ChromaStore

# 1. Canonical Company Names
_CANONICAL_COMPANIES_CACHE = None
_ALIAS_RESOLVER_CACHE = {}

def get_canonical_companies() -> List[str]:
    """
    Dynamically retrieves all unique company names from the ChromaDB collection.
    Results are cached in memory to avoid redundant database reads.
    """
    global _CANONICAL_COMPANIES_CACHE
    if _CANONICAL_COMPANIES_CACHE is not None:
        return _CANONICAL_COMPANIES_CACHE
        
    try:
        store = get_chroma_store()
        # Retrieve all documents to extract unique company names from metadata
        results = store.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        companies = set()
        for meta in metadatas:
            if meta and "company" in meta:
                companies.add(meta["company"])
        if companies:
            _CANONICAL_COMPANIES_CACHE = sorted(list(companies))
            return _CANONICAL_COMPANIES_CACHE
    except Exception as e:
        print(f"[*] Info: Could not get canonical companies from DB dynamically: {e}")
    return []

def normalize_company_name(name: str) -> str:
    """
    Fuzzy-matches and normalizes company name variations to canonical database keys.
    Uses substring overlap, acronym checks, and dynamic Chroma DB lookups to avoid hardcoding.
    """
    if not name:
        return ""
    
    # Strip common punctuation, suffixes, and lowercase
    name_clean = name.lower().strip()
    name_clean = re.sub(r'[.,;]', '', name_clean)
    name_clean = name_clean.replace(" corporation", "").replace(" inc", "").replace(" llc", "").replace(" services", "")
    name_clean = name_clean.strip()
    
    canonical_companies = get_canonical_companies()
    
    # 1. Check exact or substring overlap (case-insensitive)
    for c in canonical_companies:
        c_clean = c.lower().replace(";", "").replace("&", "and")
        clean_target = name_clean.replace("&", "and")
        if clean_target and (clean_target in c_clean or c_clean in clean_target):
            return c
            
    # 2. Check for initials/acronym matching (e.g., Tech Mahindra -> tm or techm)
    for c in canonical_companies:
        c_clean = c.lower().replace(";", "").replace("&", "and")
        c_words = [w for w in re.split(r'[^a-zA-Z0-9]', c_clean) if w]
        acronym = "".join([w[0] for w in c_words])
        if name_clean == acronym:
            return c
        if len(c_words) >= 2:
            prefix_acronym = c_words[0] + c_words[1][0] # e.g. "tech" + "m" = "techm"
            if name_clean == prefix_acronym:
                return c
                
    # 3. Fallback: Query LLM dynamically to resolve alias to one of the canonical companies
    global _ALIAS_RESOLVER_CACHE
    if name_clean in _ALIAS_RESOLVER_CACHE:
        return _ALIAS_RESOLVER_CACHE[name_clean]
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        canonical_list_str = ", ".join(canonical_companies)
        prompt = (
            f"You are a company name normalizer.\n"
            f"Given the user-specified company alias/variation: '{name}'\n"
            f"And the list of canonical companies: [{canonical_list_str}]\n"
            f"Determine if the user query refers to one of the canonical companies in the list.\n"
            f"If it does, output ONLY the exact canonical company name from the list. "
            f"If it does not, output ONLY the original name '{name}'.\n"
            f"Do not include any explanation, punctuation, or other text."
        )
        response = llm.invoke(prompt)
        resolved_name = response.content.strip()
        if resolved_name in canonical_companies:
            _ALIAS_RESOLVER_CACHE[name_clean] = resolved_name
            return resolved_name
    except Exception as e:
        print(f"[*] Info: Dynamic LLM alias resolution failed for '{name}': {e}")
        
    return name

def get_chroma_store() -> ChromaStore:
    """Helper to initialize the local ChromaStore."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(base_dir, "chroma_db")
    return ChromaStore(persist_dir=persist_dir)

# 2. Eligibility Node
def eligibility_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    EligibilityNode: Handles CGPA, backlogs, bond, and package threshold filters.
    Reads 'user_query' (or 'query') and optional 'entities' from state,
    filters the database results in Python, and updates 'eligibility_context'.
    """
    query = state.get("user_query") or state.get("query") or ""
    entities = state.get("entities", [])
    
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

# 3. Interview Prep Node
def interview_prep_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    InterviewPrepNode: Retrieves company selection rounds, topics, and programming focuses.
    Matches companies mentioned in state['entities'] or query text to technical focus sections.
    """
    query = state.get("user_query") or state.get("query") or ""
    entities = state.get("entities", [])
    
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
        
    return {"interview_context": matched_docs}

# 4. Hiring Stats Node
def hiring_stats_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    HiringStatsNode: Manages queries about numbers of SDE, Analyst, Intern, or Officer roles.
    Integrates with interview prep focus data to execute multi-column joins (e.g. Python-focused intern counts).
    """
    query = state.get("user_query") or state.get("query") or ""
    
    store = get_chroma_store()
    
    # 3. Retrieve and parse hiring data to dynamically extract available roles
    hiring_results = store.collection.get(where={"section": "hiring_distribution_data_table_(text_representation_of_all_charts_above)"})
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
        
    # 2. Check if join query (e.g. Python-focused company)
    python_companies = []
    is_python_query = "python" in query.lower()
    
    if is_python_query:
        # Dynamic check in eligibility profiles for Python tech focus or key topics
        elig_results = store.collection.get(where={"section": "section_1:_company_eligibility_profiles"})
        for meta in elig_results["metadatas"]:
            company = meta.get("company", "")
            tech_focus = meta.get("tech_focus", "").lower()
            key_topics = meta.get("key_topics", "").lower()
            if "python" in tech_focus or "python" in key_topics:
                python_companies.append(normalize_company_name(company))
                
        # Also check sections with python focus
        results = store.collection.get(include=["metadatas"])
        all_sections = list(set([m["section"] for m in results["metadatas"] if "section" in m]))
        canonical_companies = get_canonical_companies()
        for sec in all_sections:
            if sec.startswith("n_") and "technical_focus" in sec and "python" in sec.lower():
                for c in canonical_companies:
                    if c.lower() in sec.lower():
                        python_companies.append(normalize_company_name(c))
                        
        python_companies = list(set(python_companies))
    
    hiring_records = []
    for doc, meta in zip(docs, metas):
        if meta.get("type") != "tabular":
            continue
        comp = meta.get("company", "")
        norm_comp = normalize_company_name(comp)
        
        # Apply Python-focused filter if join query
        if is_python_query and norm_comp not in python_companies:
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
    
    return {"hiring_context": return_docs}

# 5. Overall Stats Node
def overall_stats_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    OverallStatsNode: Computes arithmetic placement statistics and aggregations.
    Calculates package-to-CGPA ratios and ranks offers from section 7.
    """
    query = state.get("user_query") or state.get("query") or ""
    entities = state.get("entities", [])
    
    if not entities:
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            clean_c = c.replace(";", "")
            if clean_c.lower() in query.lower():
                entities.append(c)
                
    store = get_chroma_store()
    
    results = store.collection.get(where={"section": "section_7:_overall_placement_statistics"})
    docs = results["documents"]
    metas = results["metadatas"]
    
    parsed_companies = []
    for doc, meta in zip(docs, metas):
        if meta.get("type") != "tabular":
            continue
        comp = meta.get("company", "")
        norm_comp = normalize_company_name(comp)
        
        try:
            avg_pkg = float(meta.get("avg_package", 0.0))
            avg_cgpa = float(meta.get("avg_cgpa_cutoff", 0.0))
            max_off = int(meta.get("max_offers", 0))
            min_off = int(meta.get("min_offers", 0))
        except (ValueError, TypeError):
            continue
            
        parsed_companies.append({
            "company": comp,
            "norm_company": norm_comp,
            "avg_package": avg_pkg,
            "avg_cgpa_cutoff": avg_cgpa,
            "max_offers": max_off,
            "min_offers": min_off,
            "bond_free": meta.get("bond-free?", ""),
            "meta": meta,
            "text": doc
        })
        
    summary_text = "Python Overall Statistics Analysis:\n"
    
    # Detect calculations requested
    is_ratio_query = any(x in query.lower() for x in ["ratio", "package-to-cgpa", "package to cgpa"])
    is_max_offers = any(x in query.lower() for x in ["max offers", "most offers", "maximum offers"])
    is_avg_pkg = any(x in query.lower() for x in ["average package", "avg package", "highest average"])
    
    if is_ratio_query:
        for c in parsed_companies:
            if c["avg_cgpa_cutoff"] > 0:
                c["ratio"] = c["avg_package"] / c["avg_cgpa_cutoff"]
            else:
                c["ratio"] = 0.0
        parsed_companies.sort(key=lambda x: x.get("ratio", 0.0), reverse=True)
        summary_text += "Ranked by Package-to-CGPA Ratio (Avg Package / Avg CGPA Cutoff):\n"
        for idx, c in enumerate(parsed_companies):
            summary_text += f"{idx+1}. {c['company']}: Ratio = {c['ratio']:.3f} (Avg Package: {c['avg_package']} LPA, Avg CGPA Cutoff: {c['avg_cgpa_cutoff']})\n"
    elif is_max_offers:
        parsed_companies.sort(key=lambda x: x["max_offers"], reverse=True)
        summary_text += "Ranked by Maximum Offers:\n"
        for idx, c in enumerate(parsed_companies):
            summary_text += f"{idx+1}. {c['company']}: Max Offers = {c['max_offers']}\n"
    elif is_avg_pkg:
        parsed_companies.sort(key=lambda x: x["avg_package"], reverse=True)
        summary_text += "Ranked by Average Package (LPA):\n"
        for idx, c in enumerate(parsed_companies):
            summary_text += f"{idx+1}. {c['company']}: Avg Package = {c['avg_package']} LPA\n"
    else:
        # Simple entity filter list
        if entities:
            norm_entities = [normalize_company_name(e) for e in entities]
            parsed_companies = [c for c in parsed_companies if c["norm_company"] in norm_entities]
        summary_text += "Placement Statistics Summary:\n"
        for c in parsed_companies:
            summary_text += f"- {c['text']}\n"
            
    summary_doc = Document(
        page_content=summary_text,
        metadata={"section": "section_7:_overall_placement_statistics", "type": "python_summary"}
    )
    
    return_docs = [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]
    return_docs.append(summary_doc)
    
    return {"stats_context": return_docs}

# 6. Trend Node
def trend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    TrendNode: Tracks package growth trends from 2021 to 2024.
    Computes growth = Package(2024) - Package(2021) and identifies highest absolute rise.
    """
    query = state.get("user_query") or state.get("query") or ""
    
    store = get_chroma_store()
    
    results = store.collection.get(where={"section": "n_rag_challenge_-_temporal_reasoning"})
    docs = results["documents"]
    metas = results["metadatas"]
    
    parsed_trends = []
    min_year, max_year = None, None
    for doc, meta in zip(docs, metas):
        if meta.get("type") != "tabular":
            continue
        comp = meta.get("company", "")
        if not comp:
            continue
            
        # Extract all year keys dynamically (e.g. "2021_(lpa)", "2024")
        year_keys = []
        for key in meta.keys():
            match = re.search(r"(\d{4})", key)
            if match:
                try:
                    year_val = int(match.group(1))
                    year_keys.append((year_val, key))
                except ValueError:
                    continue
                    
        if len(year_keys) < 2:
            continue
            
        # Find the earliest and latest years in the dataset
        year_keys.sort()
        earliest_yr_val, earliest_key = year_keys[0]
        latest_yr_val, latest_key = year_keys[-1]
        
        # Keep track of years for the summary header/text
        if min_year is None or earliest_yr_val < min_year:
            min_year = earliest_yr_val
        if max_year is None or latest_yr_val > max_year:
            max_year = latest_yr_val
            
        try:
            pkg_earliest = float(meta.get(earliest_key, 0.0))
            pkg_latest = float(meta.get(latest_key, 0.0))
        except (ValueError, TypeError):
            continue
            
        growth = pkg_latest - pkg_earliest
        parsed_trends.append({
            "company": comp,
            "earliest_year": earliest_yr_val,
            "latest_year": latest_yr_val,
            "earliest_pkg": pkg_earliest,
            "latest_pkg": pkg_latest,
            "growth": growth,
            "text": doc,
            "meta": meta
        })
        
    # Sort by growth descending
    parsed_trends.sort(key=lambda x: x["growth"], reverse=True)
    
    # Use fallback labels if no years were dynamically parsed
    start_label = str(min_year) if min_year is not None else "Start Year"
    end_label = str(max_year) if max_year is not None else "End Year"
    
    summary_text = f"Python Trend Analysis (Growth {start_label} to {end_label}):\n"
    for idx, t in enumerate(parsed_trends):
        summary_text += f"{idx+1}. {t['company']}: Absolute Growth = {t['growth']:.2f} LPA ({t['earliest_year']}: {t['earliest_pkg']} LPA -> {t['latest_year']}: {t['latest_pkg']} LPA)\n"
        
    summary_doc = Document(
        page_content=summary_text,
        metadata={"section": "n_rag_challenge_-_temporal_reasoning", "type": "python_summary"}
    )
    
    return_docs = [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]
    return_docs.append(summary_doc)
    
    return {"trend_context": return_docs}
