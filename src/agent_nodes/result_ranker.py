from typing import Dict, Any, List


class ResultRanker:
    def rank(self, filtered: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        if params["sort_by_package"]:
            filtered.sort(key=lambda x: x.get("package", 0.0) or 0.0, reverse=True)
        return filtered

    def build_trace(self, filtered: List[Dict], params: Dict[str, Any]) -> str:
        if params.get("is_compare"):
            trace = "Multi-Hop Reasoning Trace for Comparison:\n"
            for p in filtered:
                tf_str = ", ".join(p.get("tech_focus", [])) if p.get("tech_focus") else "None"
                trace += f"Company: {p['company']}\n"
                trace += f"  - Min CGPA: {p.get('min_cgpa')}\n"
                trace += f"  - Max Backlogs: {p.get('max_backlogs')}\n"
                trace += f"  - Package: {p.get('package')} LPA\n"
                trace += f"  - Bond: {p.get('bond')} Yrs\n"
                trace += f"  - Tech Focus: {tf_str}\n"
                if "sde" in p:
                    trace += f"  - Hiring Details: SDE={p.get('sde', 0)}, Analyst={p.get('analyst', 0)}, Officer={p.get('officer', 0)}, Intern={p.get('intern', 0)}, Total={p.get('total', 0)}\n"
                if "earliest_pkg" in p:
                    trace += f"  - Package Trend: {p.get('earliest_year')}: {p.get('earliest_pkg')} LPA -> {p.get('latest_year')}: {p.get('latest_pkg')} LPA\n"
                trace += "\n"
            return trace

        trace = "Multi-Hop Reasoning Trace:\n"
        step = 1

        if params["cgpa"] is not None or params["backlogs"] is not None:
            cgpa_val = params["cgpa"] if params["cgpa"] is not None else 10.0
            backlogs_val = params["backlogs"] if params["backlogs"] is not None else 0
            names = ", ".join([x["company"] for x in filtered]) if filtered else "None"
            trace += f"Step {step}: Filter companies by eligibility (Cutoff CGPA <= {cgpa_val} and Backlogs allowed >= {backlogs_val}).\n"
            trace += f"  Qualifying companies: {names}.\n"
            step += 1

        if params["tech_focus"] is not None:
            names = ", ".join([x["company"] for x in filtered]) if filtered else "None"
            trace += f"Step {step}: Filter companies with tech focus '{params['tech_focus']}' in interviews.\n"
            trace += f"  Qualifying companies: {names}.\n"
            step += 1

        if params["zero_bond"]:
            names = ", ".join([x["company"] for x in filtered]) if filtered else "None"
            trace += f"Step {step}: Filter companies with Zero Bond requirement.\n"
            trace += f"  Qualifying companies: {names}.\n"
            step += 1

        if params["role"] is not None:
            role = params["role"]
            trace += f"Step {step}: Cross-reference hiring data for role '{role.upper()}':\n"
            for p in filtered:
                trace += f"  - {p['company']}: {p.get(role, '0')} {role.upper()} hires\n"
            step += 1

        if params["min_package"] is not None:
            min_pkg = params["min_package"]
            names = ", ".join([x["company"] for x in filtered]) if filtered else "None"
            trace += f"Step {step}: Filter companies offering package >= {min_pkg} LPA.\n"
            trace += f"  Qualifying companies: {names}.\n"
            step += 1

        if params["sort_by_package"]:
            trace += f"Step {step}: Sort qualifying companies by package (LPA) descending:\n"
            for p in filtered:
                trace += f"  - {p['company']}: {p.get('package')} LPA\n"
            step += 1

        trace += "\nAnswer: "
        if not filtered:
            trace += "No companies found matching the specified criteria."
        else:
            if params["sort_by_package"]:
                highest_pkg = filtered[0].get("package")
                highest_companies = [p for p in filtered if p.get("package") == highest_pkg]
                names_str = " and ".join([f"{c['company']} at {c['package']} LPA" for c in highest_companies])
                trace += f"The highest-paying company is {names_str}."
            else:
                names_str = ", ".join([f"{c['company']} (Package: {c.get('package')} LPA)" for c in filtered])
                trace += f"The qualifying companies are: {names_str}."

        return trace
