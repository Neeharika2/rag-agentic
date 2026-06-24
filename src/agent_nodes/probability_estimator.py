import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from .company_utils import get_chroma_store, normalize_company_name

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
        results = store.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        all_sections = list(set([m["section"] for m in metadatas if m and "section" in m]))
        
        company_clean = company_name.lower().replace(";", "").replace(" r&d", "")
        matching_sections = [
            sec for sec in all_sections 
            if sec.startswith("n_") and company_clean in sec.lower()
        ]
        
        for sec in matching_sections:
            sec_results = store.collection.get(where={"section": sec})
            docs = sec_results.get("documents") or []
            company_docs.extend(docs)
    except Exception as e:
        print(f"[*] Info: Could not retrieve detailed sections for {company_name}: {e}")
        
    if not company_docs:
        return "No additional round details or job description available in the database."
    return "\n".join(company_docs)[:1000]

def get_company_tech_focus_keywords(company_name: str, store) -> List[str]:
    keywords = set()
    try:
        norm_company = normalize_company_name(company_name)
        results = store.collection.get(where={"section": "section_1:_company_eligibility_profiles"})
        metas = results.get("metadatas", [])
        
        for meta in metas:
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

def probability_estimator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    probability_estimator_node: Calculates a readiness score for each opportunity.
    1. Read student_profile and opportunities from the graph state.
    2. Compute Academic Fit (Max 30), Skills Compatibility (Max 40), Projects/Experience Fit (Max 30).
    3. Apply Gemini-based calibration delta (±10%).
    4. Set probability_scores and update opportunities.
    """
    profile = state.get("student_profile") or {}
    opportunities = state.get("opportunities") or []
    
    student_cgpa = profile.get("cgpa")
    student_skills = [s.lower().strip() for s in profile.get("skills", [])]
    student_weaknesses = [w.lower().strip() for w in profile.get("weaknesses", [])]
    student_interests = [i.lower().strip() for i in profile.get("interests", [])]
    projects_count = profile.get("projects_count") or 0
    
    store = get_chroma_store()
    
    probability_scores = {}
    updated_opportunities = []
    
    # Initialize LLM for calibration
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0)
        structured_llm = llm.with_structured_output(CalibrationOutput)
    except Exception as e:
        print(f"[*] Warning: Could not instantiate LLM for calibration: {e}")
        structured_llm = None
        
    for opp in opportunities:
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
            # Fallback split of opportunity tech_focus field
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
        
        # Round score to two decimal places
        final_score = round(final_score, 2)
        
        # Save score in probability_scores dictionary
        probability_scores[company_name] = final_score
        
        # Append readiness details to opportunity
        opp_copy = dict(opp)
        opp_copy["readiness_score"] = final_score
        opp_copy["calibration_reason"] = reason
        opp_copy["base_heuristic_score"] = round(base_score, 2)
        updated_opportunities.append(opp_copy)
        
    # Sort updated opportunities by readiness_score descending, then package descending
    updated_opportunities.sort(key=lambda x: (x.get("readiness_score", 0.0), x.get("package", 0.0)), reverse=True)
    
    return {
        "opportunities": updated_opportunities,
        "probability_scores": probability_scores
    }
