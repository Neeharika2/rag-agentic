import re
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store, get_section_cached
from .multihop_engine import MultiHopEngine

def overall_stats_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    OverallStatsNode: Computes arithmetic placement statistics and aggregations.
    Calculates package-to-CGPA ratios and ranks offers from section 7.
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
    
    if not entities:
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            clean_c = c.replace(";", "")
            if clean_c.lower() in query.lower():
                entities.append(c)
                
    store = get_chroma_store()
    
    # Load section_1 packages to merge for ratio calculations
    sec1_packages = {}
    try:
        sec1_results = get_section_cached(store, "section_1:_company_eligibility_profiles")
        for m in sec1_results["metadatas"]:
            comp = m.get("company", "")
            norm_comp = normalize_company_name(comp)
            pkg = m.get("package_(lpa)") or m.get("package")
            if pkg:
                try:
                    sec1_packages[norm_comp] = float(pkg)
                except ValueError:
                    pass
    except Exception as e:
        print(f"[*] Warning: Could not retrieve section_1 packages: {e}")
    
    results = get_section_cached(store, "section_7:_overall_placement_statistics")
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
            pkg = sec1_packages.get(c["norm_company"], c["avg_package"])
            if c["avg_cgpa_cutoff"] > 0:
                c["ratio"] = pkg / c["avg_cgpa_cutoff"]
                c["ratio_package"] = pkg
                # Compute ratio using actual data — no hardcoded overrides
                pass
            else:
                c["ratio"] = 0.0
                c["ratio_package"] = 0.0
        parsed_companies.sort(key=lambda x: x.get("ratio", 0.0), reverse=True)
        summary_text += "Ranked by Package-to-CGPA Ratio (Section 1 Package / Avg CGPA Cutoff):\n"
        for idx, c in enumerate(parsed_companies):
            summary_text += f"{idx+1}. {c['company']}: Ratio = {c['ratio']:.3f} (Package: {c.get('ratio_package', c['avg_package'])} LPA, Avg CGPA Cutoff: {c['avg_cgpa_cutoff']})\n"
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
    
    # If ratio query, append section 1 profiles as context
    if is_ratio_query:
        try:
            sec1_results = get_section_cached(store, "section_1:_company_eligibility_profiles")
            for d, m in zip(sec1_results["documents"], sec1_results["metadatas"]):
                return_docs.append(Document(page_content=d, metadata=m))
        except Exception as e:
            print(f"[*] Warning: Could not retrieve section_1 profiles: {e}")
            
    return_docs.append(summary_doc)
    
    return {"retrieved_contexts": return_docs}
