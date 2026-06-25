import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .llm_utils import get_structured_llm
from .company_utils import get_chroma_store, get_section_all, retrieve_semantic, normalize_company_name

class CalibrationOutput(BaseModel):
    delta: int = Field(description="Structured calibration delta, must be exactly -10, 0, or 10")
    reason: str = Field(description="Short reason explaining the subjective alignment assessment")

def is_skill_match(skill1: str, skill2: str) -> bool:
    s1 = skill1.lower().strip()
    s2 = skill2.lower().strip()
    if s1 == s2:
        return True
    
    # Substring checks
    if len(s1) > 2 and len(s2) > 2:
        if s1 in s2 or s2 in s1:
            return True
            
    # Synonym lists / Abbreviations
    synonyms = {
        "dsa": ["dsa", "algorithms", "data structures", "ds", "algo"],
        "dbms": ["dbms", "database", "databases", "sql", "mysql", "mongodb", "postgresql"],
        "os": ["os", "operating systems", "operating system"],
        "cn": ["cn", "computer networks", "computer networking", "networking", "network"],
        "oops": ["oops", "object oriented programming", "oop", "object-oriented"],
        "web dev": ["web dev", "web development", "frontend", "backend", "fullstack", "html", "css", "javascript", "react", "node"],
    }
    
    for key, values in synonyms.items():
        s1_matches = (s1 == key or s1 in values)
        s2_matches = (s2 == key or s2 in values)
        if s1_matches and s2_matches:
            return True
            
    return False

def get_company_details(company_name: str, store) -> str:
    company_docs = []
    try:
        # Use semantic search to find company-specific round details
        semantic_docs = retrieve_semantic(company_name, store, limit=10)
        for doc in semantic_docs:
            sec = doc.metadata.get("section", "")
            if sec.startswith("n_") and company_name.lower().replace(";", "").replace(" r&d", "") in sec.lower():
                company_docs.append(doc.page_content)
    except Exception as e:
        print(f"[*] Info: Could not retrieve detailed sections for {company_name}: {e}")
        
    if not company_docs:
        return "No additional round details or job description available in the database."
    return "\n".join(company_docs)[:1000]

def get_company_tech_focus_keywords(company_name: str, store) -> List[str]:
    keywords = set()
    try:
        norm_company = normalize_company_name(company_name)
        sec1_docs = get_section_all(store, "section_1:_company_eligibility_profiles")
        
        for doc in sec1_docs:
            meta = doc.metadata
            if meta and normalize_company_name(meta.get("company", "")) == norm_company:
                tf = meta.get("tech_focus") or ""
                kt = meta.get("key_topics") or ""
                for item in re.split(r'[/,;\s]', f"{tf} {kt}"):
                    item_clean = item.strip().lower()
                    if item_clean:
                        keywords.add(item_clean)
    except Exception as e:
        print(f"[*] Info: Could not get tech focus keywords from DB: {e}")
        
    return list(keywords)

def score_profile_for_company(prof: Dict[str, Any], opp: Dict[str, Any], store, structured_llm) -> Dict[str, Any]:
    student_cgpa = prof.get("cgpa")
    student_skills = [s.lower().strip() for s in prof.get("skills", [])]
    student_weaknesses = [w.lower().strip() for w in prof.get("weaknesses", [])]
    student_interests = [i.lower().strip() for i in prof.get("interests", [])]
    projects_count = prof.get("projects_count") or 0
    
    company_name = opp.get("company", "Unknown")
    min_cgpa = opp.get("min_cgpa", 0.0)
    
    # 1. Academic Fit (Max 30 Points)
    if student_cgpa is None:
        academic_score = 0.0
    elif student_cgpa < min_cgpa:
        academic_score = 0.0
    else:
        base_points = 20.0
        if 10.0 - min_cgpa > 0:
            scaling_points = min(10.0, ((student_cgpa - min_cgpa) / (10.0 - min_cgpa)) * 10.0)
        else:
            scaling_points = 10.0
        academic_score = base_points + scaling_points
        
    # 2. Skills Compatibility (Max 40 Points)
    company_keywords = get_company_tech_focus_keywords(company_name, store)
    if not company_keywords:
        tf_field = opp.get("tech_focus", "")
        company_keywords = [w.strip().lower() for w in re.split(r'[/,;\s]', tf_field) if w.strip()]
        
    matching_skills = []
    for skill in student_skills:
        for tf in company_keywords:
            if is_skill_match(skill, tf):
                matching_skills.append(skill)
                break
                
    added_points = len(matching_skills) * 10
    skills_score_base = min(40, added_points)
    
    matching_weaknesses = []
    for weakness in student_weaknesses:
        for tf in company_keywords:
            if is_skill_match(weakness, tf):
                matching_weaknesses.append(weakness)
                break
                
    subtracted_points = len(matching_weaknesses) * 10
    skills_score = max(0, skills_score_base - subtracted_points)
    
    # 3. Projects & Experience (Max 30 Points)
    if projects_count == 0:
        projects_score = 0.0
    elif projects_count == 1:
        projects_score = 15.0
    else:
        projects_score = 30.0
        
    base_score = academic_score + skills_score + projects_score
    
    # 4. LLM Semantic Calibration (±10% Delta)
    calibration_delta = 0
    reason = "LLM calibration bypassed."
    
    if structured_llm and base_score > 0:
        company_desc = get_company_details(company_name, store)
        prompt = (
            "You are an expert Placement Assistant calibrating alignment scoring for a student's readiness at a company.\n"
            "We have a student profile and a target company's placement info.\n"
            "Analyze the subjective alignment between the student's interests/skills and the company's job description/focus.\n"
            "For example, check: 'Does the student's interest in Frontend align with this backend-heavy role?' or 'Do the projects and tech stack show a strong synergy with the company's domain?'\n\n"
            "Student Profile:\n"
            f"- Skills: {student_skills}\n"
            f"- Weaknesses: {student_weaknesses}\n"
            f"- Interests: {student_interests}\n"
            f"- Projects Count: {projects_count}\n\n"
            "Target Company Info:\n"
            f"- Company: {company_name}\n"
            f"- Tech Focus: {company_keywords}\n"
            f"- Details & Round Info: {company_desc}\n\n"
            "Determine a calibration delta of either -10, 0, or 10 points to adjust the baseline heuristic score. "
            "Provide a short, clear reason for the adjustment."
        )
        try:
            res = structured_llm.invoke(prompt)
            calibration_delta = res.delta
            reason = res.reason
            if calibration_delta not in [-10, 0, 10]:
                if calibration_delta < -5:
                    calibration_delta = -10
                elif calibration_delta > 5:
                    calibration_delta = 10
                else:
                    calibration_delta = 0
        except Exception as e:
            print(f"[*] Warning: Calibration LLM call failed for {company_name}: {e}")
            reason = "Failed to run LLM calibration: using default heuristic score."
            
    final_score = max(0.0, min(100.0, base_score + calibration_delta))
    return {
        "final_score": round(final_score, 2),
        "base_heuristic_score": round(base_score, 2),
        "calibration_reason": reason
    }

def probability_estimator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    probability_estimator_node: Calculates a readiness score for each opportunity.
    1. Read student_profile, baseline_profile, and opportunities from the graph state.
    2. Compute Academic Fit (Max 30), Skills Compatibility (Max 40), Projects/Experience Fit (Max 30).
    3. Apply calibration and deltas for baseline and simulated profiles if is_simulation is active.
    4. Save the runs history to the session logs.
    """
    profile = state.get("student_profile") or {}
    baseline_profile = state.get("baseline_profile")
    is_simulation = state.get("is_simulation", False)
    opportunities = state.get("opportunities") or []
    
    store = get_chroma_store()
    
    probability_scores = {}
    updated_opportunities = []
    
    # Initialize LLM for calibration
    try:
        structured_llm = get_structured_llm(CalibrationOutput, temperature=0)
    except Exception as e:
        print(f"[*] Warning: Could not instantiate LLM for calibration: {e}")
        structured_llm = None
        
    for opp in opportunities:
        score_res = score_profile_for_company(profile, opp, store, structured_llm)
        final_score = score_res["final_score"]
        reason = score_res["calibration_reason"]
        base_score = score_res["base_heuristic_score"]
        
        opp_copy = dict(opp)
        opp_copy["readiness_score"] = final_score
        opp_copy["calibration_reason"] = reason
        opp_copy["base_heuristic_score"] = base_score
        
        if is_simulation and baseline_profile:
            # Calculate baseline score as well for what-if comparison
            base_score_res = score_profile_for_company(baseline_profile, opp, store, structured_llm)
            baseline_score = base_score_res["final_score"]
            opp_copy["baseline_readiness_score"] = baseline_score
            opp_copy["readiness_increase"] = round(final_score - baseline_score, 2)
        else:
            opp_copy["baseline_readiness_score"] = final_score
            opp_copy["readiness_increase"] = 0.0
            
        probability_scores[opp["company"]] = final_score
        updated_opportunities.append(opp_copy)
        
    # Sort updated opportunities by readiness_score descending, then package descending
    updated_opportunities.sort(key=lambda x: (x.get("readiness_score", 0.0), x.get("package", 0.0)), reverse=True)
    
    # Save score to progress logs history
    try:
        from .history_tracker import append_history, get_history
        
        # If this is a simulation and history is empty, save the baseline first!
        if is_simulation and baseline_profile and len(get_history()) == 0:
            baseline_scores = {}
            for opp_copy in updated_opportunities:
                baseline_scores[opp_copy["company"]] = opp_copy.get("baseline_readiness_score", 0.0)
            append_history(baseline_profile, baseline_scores)
            
        # Append current simulated or normal run
        append_history(profile, probability_scores)
    except Exception as e:
        print(f"[*] Warning: Could not save run to history tracker: {e}")
        
    return {
        "opportunities": updated_opportunities,
        "probability_scores": probability_scores
    }
