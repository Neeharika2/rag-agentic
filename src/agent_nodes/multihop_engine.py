import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_chroma_store, get_section_all, retrieve_semantic
from .query_resolver import QueryResolver
from .filter_engine import FilterEngine
from .result_ranker import ResultRanker


class MultiHopEngine:
    resolver = QueryResolver()
    filter_engine = FilterEngine()
    ranker = ResultRanker()

    @staticmethod
    def get_unified_profiles() -> List[Dict[str, Any]]:
        store = get_chroma_store()

        elig_docs = get_section_all(store, "section_1:_company_eligibility_profiles")
        elig_metas = [d.metadata for d in elig_docs]

        hiring_docs = get_section_all(store, "hiring_distribution_data_table_(text_representation_of_all_charts_above)")
        hiring_metas = [d.metadata for d in hiring_docs]

        stats_docs = get_section_all(store, "section_7:_overall_placement_statistics")
        stats_metas = [d.metadata for d in stats_docs]

        temporal_docs = get_section_all(store, "n_rag_challenge_-_temporal_reasoning")
        temporal_metas = [d.metadata for d in temporal_docs]

        results = store.collection.get(include=["metadatas"])
        all_metas = results.get("metadatas", [])

        company_tech_focuses = {}
        for meta in all_metas:
            sec = meta.get("section", "")
            if sec.startswith("n_") and "technical_focus" in sec:
                parts = sec.split("_|_")
                if len(parts) > 0:
                    comp_part = parts[0][2:].replace("_", " ").strip()
                    comp_norm = normalize_company_name(comp_part)
                    tech_part = parts[1] if len(parts) > 1 else ""
                    focus_match = re.search(r"technical_focus:\s*(.+)", tech_part, re.IGNORECASE)
                    focus = focus_match.group(1).replace("_", " ") if focus_match else tech_part

                    if comp_norm not in company_tech_focuses:
                        company_tech_focuses[comp_norm] = set()
                    for f in re.split(r'[/,;]', focus):
                        f_clean = f.strip().lower()
                        if f_clean:
                            company_tech_focuses[comp_norm].add(f_clean)

        unified = {}

        for meta in elig_metas:
            comp = meta.get("company", "")
            if not comp:
                continue
            norm_comp = normalize_company_name(comp)
            tf_set = set()
            for tf in [meta.get("tech_focus", ""), meta.get("key_topics", "")]:
                if tf:
                    for part in re.split(r'[/,;]', tf):
                        part_clean = part.strip().lower()
                        if part_clean:
                            tf_set.add(part_clean)

            unified[norm_comp] = {
                "company": comp,
                "norm_company": norm_comp,
                "min_cgpa": float(meta.get("min_cgpa") or 0.0) if meta.get("min_cgpa") else None,
                "max_backlogs": int(meta.get("max_backlogs") or 0) if meta.get("max_backlogs") is not None else None,
                "package": float(meta.get("package_(lpa)") or 0.0) if meta.get("package_(lpa)") else None,
                "bond": int(meta.get("bond_(yrs)") or 0) if meta.get("bond_(yrs)") is not None else None,
                "tech_focus": tf_set
            }

        for meta in hiring_metas:
            comp = meta.get("company", "")
            if not comp:
                continue
            norm_comp = normalize_company_name(comp)
            if norm_comp not in unified:
                unified[norm_comp] = {
                    "company": comp, "norm_company": norm_comp,
                    "min_cgpa": None, "max_backlogs": None,
                    "package": None, "bond": None, "tech_focus": set()
                }
            for role in ["sde", "analyst", "officer", "intern", "total"]:
                val = meta.get(role)
                if val is not None:
                    try:
                        unified[norm_comp][role] = int(val)
                    except ValueError:
                        unified[norm_comp][role] = 0

        for meta in stats_metas:
            comp = meta.get("company", "")
            if not comp:
                continue
            norm_comp = normalize_company_name(comp)
            if norm_comp not in unified:
                unified[norm_comp] = {
                    "company": comp, "norm_company": norm_comp,
                    "min_cgpa": None, "max_backlogs": None,
                    "package": None, "bond": None, "tech_focus": set()
                }
            unified[norm_comp]["avg_package"] = float(meta.get("avg_package") or 0.0) if meta.get("avg_package") else None
            unified[norm_comp]["avg_cgpa_cutoff"] = float(meta.get("avg_cgpa_cutoff") or 0.0) if meta.get("avg_cgpa_cutoff") else None
            unified[norm_comp]["bond_free"] = meta.get("bond-free?", "").strip().lower() == "yes"

        for meta in temporal_metas:
            comp = meta.get("company", "")
            if not comp:
                continue
            norm_comp = normalize_company_name(comp)
            if norm_comp not in unified:
                unified[norm_comp] = {
                    "company": comp, "norm_company": norm_comp,
                    "min_cgpa": None, "max_backlogs": None,
                    "package": None, "bond": None, "tech_focus": set()
                }
            year_keys = []
            for key in meta.keys():
                match = re.search(r"(\d{4})", key)
                if match:
                    try:
                        year_val = int(match.group(1))
                        year_keys.append((year_val, key))
                    except ValueError:
                        continue
            if len(year_keys) >= 2:
                year_keys.sort()
                _, earliest_key = year_keys[0]
                _, latest_key = year_keys[-1]
                unified[norm_comp]["earliest_pkg"] = meta.get(earliest_key)
                unified[norm_comp]["latest_pkg"] = meta.get(latest_key)
                unified[norm_comp]["earliest_year"] = year_keys[0][0]
                unified[norm_comp]["latest_year"] = year_keys[-1][0]

        for comp_norm, focuses in company_tech_focuses.items():
            if comp_norm in unified:
                unified[comp_norm]["tech_focus"].update(focuses)

        return list(unified.values())

    @staticmethod
    def resolve_query(query: str) -> Optional[Document]:
        from .company_utils import get_canonical_companies
        query_lower = query.lower()
        
        # Extract entities mentioned in query for comparison fallback
        entities = []
        canonical_companies = get_canonical_companies()
        for c in canonical_companies:
            clean_c = c.replace(";", "")
            if clean_c.lower() in query_lower:
                entities.append(c)
                
        is_compare = "compare" in query_lower and len(entities) >= 2

        params = MultiHopEngine.resolver.parse(query)
        params["is_compare"] = is_compare
        
        active_params = 0
        if params["cgpa"] is not None or params["backlogs"] is not None:
            active_params += 1
        if params["min_package"] is not None or params["sort_by_package"]:
            active_params += 1
        if params["role"] is not None:
            active_params += 1
        if params["zero_bond"]:
            active_params += 1
        if params["tech_focus"] is not None:
            active_params += 1

        if active_params < 2 and not is_compare:
            return None

        profiles = MultiHopEngine.get_unified_profiles()
        
        if is_compare:
            from .company_utils import normalize_company_name
            norm_entities = [normalize_company_name(e) for e in entities]
            filtered = [p for p in profiles if p["norm_company"] in norm_entities]
        else:
            filtered = MultiHopEngine.filter_engine.filter(profiles, params)
            
        ranked = MultiHopEngine.ranker.rank(filtered, params)
        trace = MultiHopEngine.ranker.build_trace(ranked, params)

        return Document(
            page_content=trace,
            metadata={"section": "multi_hop_reasoning", "type": "python_summary"}
        )
