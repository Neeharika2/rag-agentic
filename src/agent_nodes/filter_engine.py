import re
from typing import Dict, Any, List


class FilterEngine:
    def filter(self, profiles: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        filtered = list(profiles)

        if params["cgpa"] is not None or params["backlogs"] is not None:
            cgpa_val = params["cgpa"] if params["cgpa"] is not None else 10.0
            backlogs_val = params["backlogs"] if params["backlogs"] is not None else 0
            qualifying = []
            for p in filtered:
                p_cgpa = p.get("min_cgpa")
                p_backlogs = p.get("max_backlogs")
                if p_cgpa is not None and p_backlogs is not None:
                    if p_cgpa <= cgpa_val and p_backlogs >= backlogs_val:
                        qualifying.append(p)
            filtered = qualifying

        if params["tech_focus"] is not None:
            tf = params["tech_focus"]
            qualifying = []
            for p in filtered:
                if tf in p.get("tech_focus", set()):
                    qualifying.append(p)
            filtered = qualifying

        if params["zero_bond"]:
            qualifying = []
            for p in filtered:
                if p.get("bond") == 0 or p.get("bond_free") is True:
                    qualifying.append(p)
            filtered = qualifying

        if params["role"] is not None:
            role = params["role"]
            threshold = self._infer_threshold(role, filtered)
            qualifying = []
            for p in filtered:
                try:
                    hires_val = int(p.get(role, 0))
                except (ValueError, TypeError):
                    hires_val = 0
                if hires_val >= threshold:
                    qualifying.append(p)
            filtered = qualifying

        if params["min_package"] is not None:
            min_pkg = params["min_package"]
            qualifying = []
            for p in filtered:
                pkg = p.get("package")
                if pkg is not None and pkg >= min_pkg:
                    qualifying.append(p)
            filtered = qualifying

        return filtered

    def _infer_threshold(self, role: str, profiles: List[Dict]) -> int:
        return 40
