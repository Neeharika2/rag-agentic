from typing import Dict, Any
import re
from .company_utils import parse_cgpa_from_text, parse_backlogs_from_text

class QueryResolver:
    def parse(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        params = {
            "cgpa": None,
            "backlogs": None,
            "min_package": None,
            "sort_by_package": False,
            "role": None,
            "zero_bond": False,
            "tech_focus": None
        }

        params["cgpa"] = parse_cgpa_from_text(query_lower)
        params["backlogs"] = parse_backlogs_from_text(query_lower)


        pkg_match = re.search(r"(?:above|more than|greater than|>\s*)\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakh|lakhs)?", query_lower)
        if pkg_match:
            val = float(pkg_match.group(1))
            start_idx = pkg_match.start()
            pre_context = query_lower[max(0, start_idx - 15):start_idx]
            if "cgpa" in pre_context or "gpa" in pre_context or (params["cgpa"] is not None and abs(val - params["cgpa"]) < 0.01):
                pass
            else:
                params["min_package"] = val
        else:
            pkg_lpa = re.search(r"(\d+(?:\.\d+)?)\s*(?:lpa|lakh|lakhs)", query_lower)
            if pkg_lpa and any(x in query_lower for x in ["above", "more than", "greater than", ">", "offer"]):
                params["min_package"] = float(pkg_lpa.group(1))

        if any(x in query_lower for x in ["highest-paying", "maximum pay", "highest package", "highest pay", "highest-paid", "max pay", "highest salary", "best package"]):
            params["sort_by_package"] = True

        if any(x in query_lower for x in ["zero-bond", "0-bond", "zero bond", "0 bond", "bond-free", "bond free", "no bond"]):
            params["zero_bond"] = True

        for lang in ["python", "java", "c++", "cloud", "system design", "dsa"]:
            if lang in query_lower:
                params["tech_focus"] = lang
                break

        for r in ["sde", "analyst", "officer", "intern"]:
            if r in query_lower:
                params["role"] = r
                break
        if not params["role"]:
            if "software" in query_lower or "developer" in query_lower:
                params["role"] = "sde"
            elif "internship" in query_lower:
                params["role"] = "intern"

        return params
