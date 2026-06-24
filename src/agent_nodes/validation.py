import os
import re
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_canonical_companies, get_chroma_store

def parse_attributes_from_text(text: str) -> Dict[str, str]:
    """
    Parses a text block of serialized key-value pairs using standard regex:
    r"(\w[\w\s\-_\(\)]*):\s*([^\n,.]+)"
    """
    attrs = {}
    matches = re.findall(r"(\w[\w\s\-_\(\)]*):\s*([^\n,.]+)", text)
    for k, v in matches:
        clean_k = k.strip().lower().replace(" ", "_")
        clean_v = v.strip()
        attrs[clean_k] = clean_v
    return attrs

def detect_conflicts_dynamically(all_docs: List[Document], target_companies: List[str]) -> Optional[Dict[str, Any]]:
    """
    Modular Python Verification Algorithm:
    1. Extracts and groups entity key-value attributes from chunks.
    2. Compares keys present in more than one chunk where source is official and portal.
    3. Dynamically maps any detected value discrepancy.
    """
    companies_data = {}
    
    for doc in all_docs:
        meta = doc.metadata or {}
        text = doc.page_content or ""
        
        # Parse attributes from text and metadata
        attrs = parse_attributes_from_text(text)
        for k, v in meta.items():
            clean_k = k.lower().replace(" ", "_")
            attrs[clean_k] = str(v)
            
        company = attrs.get("company")
        if not company:
            continue
            
        norm_company = normalize_company_name(company)
        norm_company_lower = norm_company.lower()
        
        # If target entities are defined, filter checking target companies only
        if target_companies and norm_company_lower not in target_companies:
            continue
            
        if norm_company not in companies_data:
            companies_data[norm_company] = []
            
        # Determine source type based on section metadata
        section = attrs.get("section", "")
        source = "unknown"
        if "section_1" in section:
            source = "official"
        elif "conflicting_information" in section:
            source = "both"
        elif "portal" in section or "portal" in text.lower():
            source = "portal"
            
        attrs["_source"] = source
        companies_data[norm_company].append(attrs)
        
    for company, records in companies_data.items():
        # Check single-record conflicts (both values inside one chunk)
        for r in records:
            cgpa_off = r.get("cgpa_(official)") or r.get("cgpa_official") or r.get("official_cgpa")
            cgpa_port = r.get("cgpa_(portal)") or r.get("cgpa_portal") or r.get("portal_cgpa")
            pkg_off = r.get("package_official") or r.get("official_package") or r.get("package_(official)")
            pkg_port = r.get("package_portal") or r.get("portal_package") or r.get("package_(portal)")
            
            if cgpa_off and cgpa_port and cgpa_off != cgpa_port:
                return {
                    "company": company,
                    "metric": "Min CGPA",
                    "official_value": cgpa_off,
                    "portal_value": cgpa_port
                }
            if pkg_off and pkg_port and pkg_off != pkg_port:
                return {
                    "company": company,
                    "metric": "Package",
                    "official_value": pkg_off,
                    "portal_value": pkg_port
                }
                
        # Check cross-record conflicts (different values in official vs portal chunks)
        official_cgpas = []
        portal_cgpas = []
        official_packages = []
        portal_packages = []
        
        for r in records:
            source = r.get("_source")
            cgpa = r.get("min_cgpa") or r.get("avg_cgpa_cutoff") or r.get("cgpa")
            pkg = r.get("package_(lpa)") or r.get("avg_package") or r.get("package")
            
            if source == "official":
                if cgpa: official_cgpas.append(cgpa)
                if pkg: official_packages.append(pkg)
            elif source == "portal":
                if cgpa: portal_cgpas.append(cgpa)
                if pkg: portal_packages.append(pkg)
                
        # Compare official vs portal values
        if official_cgpas and portal_cgpas:
            for off in official_cgpas:
                for port in portal_cgpas:
                    if off != port:
                        return {
                            "company": company,
                            "metric": "Min CGPA",
                            "official_value": off,
                            "portal_value": port
                        }
        if official_packages and portal_packages:
            for off in official_packages:
                for port in portal_packages:
                    if off != port:
                        return {
                            "company": company,
                            "metric": "Package",
                            "official_value": off,
                            "portal_value": port
                        }
                        
    return None

def log_retrieved_chunks(query: str, docs: List[Document]):
    """
    Logs the chunks retrieved for a user query to logs/query_retrievals.log.
    """
    try:
        import datetime
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "query_retrievals.log")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Total Chunks Retrieved: {len(docs)}\n")
            f.write("-" * 80 + "\n")
            for idx, doc in enumerate(docs):
                f.write(f"Chunk {idx+1}:\n")
                f.write(f"  Content: {doc.page_content}\n")
                f.write(f"  Metadata: {doc.metadata}\n")
                f.write("\n")
            f.write("=" * 80 + "\n\n")
    except Exception as e:
        print(f"[*] Warning: Could not log retrieved chunks: {e}")

def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ValidationNode: Dynamic conflict verification.
    Scans retrieved chunks across contexts, checks for value discrepancies,
    and sets conflict_detected/conflict_details dynamically using modular helpers.
    """
    # Gather all contexts
    all_docs = list(state.get("retrieved_contexts") or [])
            
    # Exclude multi-hop summary documents from conflict detection to avoid false conflicts
    all_docs = [d for d in all_docs if d.metadata and "multi_hop_reasoning" not in d.metadata.get("section", "")]
            
    # If query type is conflict, dynamically query the conflict section from Chroma DB
    q_type = state.get("query_type")
    conflict_docs = []
    if q_type == "conflict" or "conflict" in (state.get("user_query") or "").lower():
        try:
            store = get_chroma_store()
            conflict_results = store.collection.get(where={"section": "n_rag_challenge_-_conflicting_information"})
            conflict_docs = [Document(page_content=d, metadata=m) for d, m in zip(conflict_results["documents"], conflict_results["metadatas"])]
            all_docs.extend(conflict_docs)
        except Exception as e:
            print(f"[*] Info: Could not retrieve conflict documents: {e}")
            
    target_companies = [c.lower() for c in state.get("entities", [])]
    
    # Log the chunks retrieved for this query
    query = state.get("user_query") or state.get("query") or ""
    log_retrieved_chunks(query, all_docs)
    
    conflict_details = detect_conflicts_dynamically(all_docs, target_companies)
    conflict_detected = conflict_details is not None and len(target_companies) > 0
    
    # If we retrieved conflict docs, append them to retrieved_contexts so synthesis gets them
    ret_dict = {
        "conflict_detected": conflict_detected,
        "conflict_details": conflict_details
    }
    if conflict_docs:
        ret_dict["retrieved_contexts"] = conflict_docs
        
    return ret_dict
