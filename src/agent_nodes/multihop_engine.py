import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .company_utils import normalize_company_name, get_chroma_store
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

        elig_res = store.collection.get(where={"section": "section_1:_company_eligibility_profiles"})
        elig_metas = elig_res.get("metadatas", [])

        hiring_res = store.collection.get(where={"section": "hiring_distribution_data_table_(text_representation_of_all_charts_above)"})
        hiring_metas = hiring_res.get("metadatas", [])

        stats_res = store.collection.get(where={"section": "section_7:_overall_placement_statistics"})
        stats_metas = stats_res.get("metadatas", [])

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

        for comp_norm, focuses in company_tech_focuses.items():
            if comp_norm in unified:
                unified[comp_norm]["tech_focus"].update(focuses)

        return list(unified.values())

    @staticmethod
    def resolve_query(query: str) -> Optional[Document]:
        params = MultiHopEngine.resolver.parse(query)
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

        if active_params < 2:
            return None

        profiles = MultiHopEngine.get_unified_profiles()
        filtered = MultiHopEngine.filter_engine.filter(profiles, params)
        ranked = MultiHopEngine.ranker.rank(filtered, params)
        trace = MultiHopEngine.ranker.build_trace(ranked, params)

        return Document(
            page_content=trace,
            metadata={"section": "multi_hop_reasoning", "type": "python_summary"}
        )
