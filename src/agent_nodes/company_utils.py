import os
import re
from typing import List, Optional
from src.vectorstore.chroma_store import ChromaStore

_CANONICAL_COMPANIES_CACHE = None
_ALIAS_RESOLVER_CACHE = {}
_SECTION_CACHE: dict = {}
_SECTION_CACHE_ENABLED = True

# Load env vars — uses centralized loader from llm_utils
from .llm_utils import _load_env_file
_load_env_file()

def get_chroma_store() -> ChromaStore:
    """Helper to initialize the local ChromaStore."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    persist_dir = os.path.join(base_dir, "chroma_db")
    return ChromaStore(persist_dir=persist_dir)

def get_canonical_companies() -> List[str]:
    global _CANONICAL_COMPANIES_CACHE
    if _CANONICAL_COMPANIES_CACHE is not None:
        return _CANONICAL_COMPANIES_CACHE
        
    try:
        store = get_chroma_store()
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
    if not name:
        return ""
    
    name_clean = name.lower().strip()
    name_clean = re.sub(r'[.,;]', '', name_clean)
    name_clean = name_clean.replace(" corporation", "").replace(" inc", "").replace(" llc", "").replace(" services", "")
    name_clean = name_clean.strip()
    
    canonical_companies = get_canonical_companies()
    
    for c in canonical_companies:
        c_clean = c.lower().replace(";", "").replace("&", "and")
        clean_target = name_clean.replace("&", "and")
        if clean_target and (clean_target in c_clean or c_clean in clean_target):
            return c
            
    for c in canonical_companies:
        c_clean = c.lower().replace(";", "").replace("&", "and")
        c_words = [w for w in re.split(r'[^a-zA-Z0-9]', c_clean) if w]
        acronym = "".join([w[0] for w in c_words])
        if name_clean == acronym:
            return c
        if len(c_words) >= 2:
            prefix_acronym = c_words[0] + c_words[1][0]
            if name_clean == prefix_acronym:
                return c
                
    return name

def parse_cgpa_from_text(query: str) -> Optional[float]:
    # Remove common false positives: years, package values, percentages, large numbers
    clean_query = re.sub(r"\b\d{4}\b", "", query.lower())
    clean_query = re.sub(r"\d+(?:\.\d+)?\s*(?:lpa|lakh|lakhs|%|percent)", "", clean_query)
    clean_query = re.sub(r"\b\d{3,}\b", "", clean_query)
    cgpa_floats = [float(x) for x in re.findall(r"\b\d+\.\d+\b", clean_query)]
    cgpa_ints = [float(x) for x in re.findall(r"\b[6789]\b|\b10\b", clean_query)]
    all_cgpas = [c for c in (cgpa_floats + cgpa_ints) if 5.0 <= c <= 10.0]
    return all_cgpas[0] if all_cgpas else None

def parse_backlogs_from_text(query: str) -> Optional[int]:
    backlog_match = re.search(r"(\d+)\s*(?:active\s*)?backlog", query, re.IGNORECASE)
    return int(backlog_match.group(1)) if backlog_match else None

def check_academic_eligibility(
    student_cgpa: Optional[float],
    student_backlogs: Optional[int],
    min_cgpa: float,
    max_backlogs: int,
    no_bond: bool = False,
    bond_years: int = 0,
    min_cgpa_above: Optional[float] = None,
    min_backlogs_at_least: Optional[int] = None
) -> bool:
    """Consolidated academic eligibility helper used across nodes."""
    if student_cgpa is not None and min_cgpa > student_cgpa:
        return False
    if student_backlogs is not None and max_backlogs < student_backlogs:
        return False
    if no_bond and bond_years > 0:
        return False
    if min_cgpa_above is not None and min_cgpa <= min_cgpa_above:
        return False
    if min_backlogs_at_least is not None and max_backlogs < min_backlogs_at_least:
        return False
    return True

def retrieve_semantic(query: str, store: ChromaStore, section: str = None, limit: int = 10) -> List[Document]:
    """
    PRIMARY retrieval method for the RAG pipeline.
    Uses vector (semantic) search via ChromaDB's ANN index to find documents
    whose meaning is closest to the query. Optionally filtered by section.

    This is the 'R' in RAG — embedding-based nearest-neighbor search,
    not exact keyword matching. Returns Document objects ready for LLM context.
    """
    if not query.strip():
        return _retrieve_exact_all(store, section, limit)
    filter_dict = {"section": section} if section else None
    raw = store.search(query, limit=limit, filter_dict=filter_dict)
    return [Document(page_content=r["text"], metadata=r["metadata"]) for r in raw]


def _retrieve_exact_all(store: ChromaStore, section: str = None, limit: int = 50) -> List[Document]:
    """Secondary fallback: exact metadata-filtered scan. Use when semantic search isn't applicable
    (e.g. listing all companies, bulk data for filtering)."""
    where = {"section": section} if section else None
    raw = store.collection.get(where=where, limit=limit)
    docs = []
    for d, m in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=d, metadata=m))
    return docs


def get_section_all(store: ChromaStore, section: str) -> List[Document]:
    """
    Exact metadata-filtered retrieval — returns ALL documents in a section.
    This is NOT semantic search. Use retrieve_semantic() for query-driven retrieval.
    Results are cached per-session to avoid repeated DB scans.
    """
    global _SECTION_CACHE
    if _SECTION_CACHE_ENABLED and section in _SECTION_CACHE:
        raw = _SECTION_CACHE[section]
    else:
        raw = store.collection.get(where={"section": section})
        if _SECTION_CACHE_ENABLED:
            _SECTION_CACHE[section] = raw
    docs = []
    for d, m in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=d, metadata=m))
    return docs


def clear_section_cache():
    global _SECTION_CACHE
    _SECTION_CACHE = {}
