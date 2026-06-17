import os
import re
from typing import List
from pathlib import Path
from src.vectorstore.chroma_store import ChromaStore

_CANONICAL_COMPANIES_CACHE = None
_ALIAS_RESOLVER_CACHE = {}

def _load_env_file():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
        
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key] = val

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
