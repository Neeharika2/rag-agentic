import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store, get_section_cached
from .multihop_engine import MultiHopEngine

def trend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    TrendNode: Tracks package growth trends from 2021 to 2024.
    Computes growth = Package(2024) - Package(2021) and identifies highest absolute rise.
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
    
    results = get_section_cached(store, "n_rag_challenge_-_temporal_reasoning")
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
    
    return {"retrieved_contexts": return_docs}
