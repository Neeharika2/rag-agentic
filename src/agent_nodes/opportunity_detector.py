import re
from typing import Dict, Any, List
from langchain_core.documents import Document
from .company_utils import get_chroma_store, get_section_all, retrieve_semantic, normalize_company_name, check_academic_eligibility

def opportunity_detector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpportunityDetectorNode: Evaluates the student's profile against company criteria.
    1. Loads all company eligibility profiles from ChromaDB.
    2. Applies hard constraints (CGPA >= cutoff, backlogs <= max allowed).
    3. Calculates a skill alignment score based on tech focus overlap.
    4. Sorts by eligibility and package (LPA) descending.
    """
    profile = state.get("student_profile") or {}
    
    student_cgpa = profile.get("cgpa")
    student_backlogs = profile.get("backlogs") or 0
    student_skills = [s.lower().strip() for s in profile.get("skills", [])]
    
    store = get_chroma_store()
    
    # Retrieve all eligibility profiles from database
    try:
        # Use semantic search to find relevant eligibility profiles
        profile_docs = retrieve_semantic("company eligibility criteria cutoff package", store, section="section_1:_company_eligibility_profiles", limit=50)
        if not profile_docs:
            profile_docs = get_section_all(store, "section_1:_company_eligibility_profiles")
    except Exception as e:
        print(f"[*] Error fetching profiles from ChromaDB: {e}")
        profile_docs = []
        
    eligible_companies = []
    
    for doc in profile_docs:
        meta = doc.metadata
        if not meta or meta.get("type") != "tabular":
            continue
            
        comp_name = meta.get("company", "Unknown")
        
        # Parse cutoffs safely
        try:
            min_cgpa = float(meta.get("min_cgpa") or 0.0)
            max_backlogs = int(meta.get("max_backlogs") or 0)
            package = float(meta.get("package_(lpa)") or 0.0)
            bond = int(meta.get("bond_(yrs)") or 0)
            tech_focus = meta.get("tech_focus") or meta.get("key_topics") or ""
        except (ValueError, TypeError):
            continue
            
        # Hard Academic Filters
        if not check_academic_eligibility(student_cgpa, student_backlogs, min_cgpa, max_backlogs):
            continue
                
        # Calculate Skill Match Score
        skill_score = 0
        tech_words = re.split(r'[/,;\s]', tech_focus.lower())
        tech_words = {w.strip() for w in tech_words if w.strip()}
        
        for skill in student_skills:
            if skill in tech_words:
                skill_score += 1
            else:
                # check for substring matches, e.g. "python" in "python/oops"
                for tw in tech_words:
                    if skill in tw or tw in skill:
                        skill_score += 0.5
                        break
                        
        eligible_companies.append({
            "company": comp_name,
            "package": package,
            "min_cgpa": min_cgpa,
            "max_backlogs": max_backlogs,
            "tech_focus": tech_focus,
            "bond": bond,
            "skill_score": skill_score
        })
        
    # Sort eligible companies by skill score descending, then starting package (LPA) descending
    eligible_companies.sort(key=lambda x: (x["skill_score"], x["package"]), reverse=True)
    
    print(f"[*] OpportunityDetector: Matched {len(eligible_companies)} eligible companies.")
    for c in eligible_companies[:3]:
         print(f"    - {c['company']} ({c['package']} LPA, Skill Score: {c['skill_score']})")
         
    return {
        "opportunities": eligible_companies
    }
