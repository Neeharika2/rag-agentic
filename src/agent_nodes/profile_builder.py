from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import List
from .llm_utils import get_structured_llm
from langchain_core.prompts import ChatPromptTemplate
from .company_utils import clear_section_cache

class StudentProfile(BaseModel):
    cgpa: Optional[float] = Field(None, description="Student's CGPA, normalized between 0.0 and 10.0")
    skills: List[str] = Field(default_factory=list, description="Extracted tech stack, programming languages, or domain skills (e.g., Python, SQL, Web Dev)")
    weaknesses: List[str] = Field(default_factory=list, description="Explicitly mentioned areas of improvement or weaknesses (e.g., DSA, OS, DBMS)")
    interests: List[str] = Field(default_factory=list, description="Target job roles or fields (e.g., SDE, Analyst, Intern)")
    backlogs: int = Field(default=0, description="Number of active backlogs mentioned. Defaults to 0")
    projects_count: int = Field(default=0, description="Number of projects or project experience mentioned. Defaults to 0")
    is_strategy_query: bool = Field(
        description="True if the user is describing their profile, asking for placement suggestions, requesting a strategy plan, asking what-if simulations, or querying eligible opportunities based on their situation. False if it is a simple factual lookup question about a company."
    )
    is_simulation: bool = Field(
        default=False,
        description="True if the user is asking a 'what-if' profile simulation query (e.g. 'What if I learn C++?', 'How does adding DSA help?', 'What if my CGPA goes up to 8.5?'). False otherwise."
    )
    simulated_cgpa_delta: Optional[float] = Field(
        None,
        description="The absolute target CGPA to simulate (e.g., 8.5) or a positive change delta (e.g., 0.5 or +0.5) mentioned in a what-if query."
    )
    simulated_skills_to_add: List[str] = Field(
        default_factory=list,
        description="List of skills the user wants to simulate learning or adding (e.g. ['C++', 'DSA'])."
    )

def profile_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ProfileBuilderNode: Parses free-form queries into structured student profiles.
    If 'student_profile' was already supplied directly via the frontend form payload,
    it uses it directly and flags 'is_strategy_query' as True, unless it is a what-if query.
    """
    clear_section_cache()
    query = state.get("user_query") or state.get("query") or ""
    query_lower = query.lower()
    
    # Check if this is a what-if simulation query based on keywords
    sim_keywords = ["what if", "what-if", "simulate", "if i learn", "if i add", "if my cgpa"]
    is_sim_query = any(k in query_lower for k in sim_keywords)
    
    # 1. If student profile was passed in state directly and not a simulation query, use it
    input_profile = state.get("student_profile")
    if input_profile and isinstance(input_profile, dict) and input_profile.get("cgpa") is not None and not is_sim_query:
        print("[*] ProfileBuilder: Found pre-populated profile in state.")
        return {
            "student_profile": input_profile,
            "is_strategy_query": True,
            "is_simulation": False
        }
        
    # 2. Rule-based bypass for factual/lookup queries to save Gemini API quota
    strategy_keywords = ["strategy", "plan", "roadmap", "advisor", "suggest", "recommend", "prepare me", "pathway", "career path", "what should i do", "advice", "what to do"]
    has_strategy_keyword = any(k in query_lower for k in strategy_keywords)
    
    if not has_strategy_keyword and not is_sim_query:
        print("[*] ProfileBuilder: Bypassing LLM profile extraction for factual lookup query.")
        return {
            "student_profile": {
                "cgpa": None,
                "skills": [],
                "weaknesses": [],
                "interests": [],
                "backlogs": 0,
                "projects_count": 0
            },
            "is_strategy_query": False,
            "is_simulation": False
        }

    # 3. Extract structured profile/delta using LLM
    try:
        structured_llm = get_structured_llm(StudentProfile, temperature=0)
        
        prompt = (
            f"You are an expert SVECW Placement Assistant. Analyze the student's message and build a structured representation "
            f"of their profile, detect if their query is a career planning/strategy request, and extract any simulated "
            f"what-if changes if they are asking a simulation query (e.g. 'What if I learn Java?', 'How does adding DSA help?', 'What if my CGPA increases by 0.5?').\n\n"
            f"Student query: '{query}'"
        )
        
        result = structured_llm.invoke(prompt)
        
        profile_dict = {
            "cgpa": result.cgpa,
            "skills": result.skills,
            "weaknesses": result.weaknesses,
            "interests": result.interests,
            "backlogs": result.backlogs,
            "projects_count": result.projects_count
        }
        
        is_simulation = result.is_simulation or is_sim_query
        
        if is_simulation:
            # Load baseline profile from input_profile or history_tracker
            baseline_profile = None
            if input_profile and isinstance(input_profile, dict) and input_profile.get("cgpa") is not None:
                baseline_profile = input_profile
            else:
                from .history_tracker import get_latest_profile
                baseline_profile = get_latest_profile()
                
            if not baseline_profile:
                # Default baseline fallback using extracted values
                baseline_profile = {
                    "cgpa": profile_dict["cgpa"] if profile_dict["cgpa"] is not None else 7.0,
                    "skills": profile_dict["skills"] or [],
                    "weaknesses": profile_dict["weaknesses"] or [],
                    "interests": profile_dict["interests"] or [],
                    "backlogs": profile_dict["backlogs"] or 0,
                    "projects_count": profile_dict["projects_count"] or 0
                }
            
            # Create a copy for simulation merge
            simulated_profile = dict(baseline_profile)
            
            # Merge CGPA delta/absolute
            sim_cgpa_delta = result.simulated_cgpa_delta
            if sim_cgpa_delta is not None:
                if sim_cgpa_delta >= 4.0:
                    simulated_profile["cgpa"] = min(10.0, float(sim_cgpa_delta))
                else:
                    base_cgpa = float(baseline_profile.get("cgpa") or 7.0)
                    simulated_profile["cgpa"] = min(10.0, base_cgpa + float(sim_cgpa_delta))
            
            # Merge skills
            sim_skills_to_add = result.simulated_skills_to_add or []
            if sim_skills_to_add:
                # Deduplicate and merge
                existing_skills = [s.lower().strip() for s in baseline_profile.get("skills", [])]
                merged_skills = list(baseline_profile.get("skills", []))
                for s in sim_skills_to_add:
                    if s.lower().strip() not in existing_skills:
                        merged_skills.append(s)
                simulated_profile["skills"] = merged_skills
                
                # Remove learned skills from weaknesses
                weaknesses = baseline_profile.get("weaknesses", [])
                from .probability_estimator import is_skill_match
                simulated_profile["weaknesses"] = [
                    w for w in weaknesses 
                    if not any(is_skill_match(w, s) for s in sim_skills_to_add)
                ]
                
            print(f"[*] ProfileBuilder: Simulation detected. Baseline={baseline_profile}, Simulated={simulated_profile}")
            
            return {
                "student_profile": simulated_profile,
                "baseline_profile": baseline_profile,
                "is_strategy_query": True,
                "is_simulation": True
            }
        
        print(f"[*] ProfileBuilder: Parsed profile -> {profile_dict}")
        print(f"[*] ProfileBuilder: is_strategy_query -> {result.is_strategy_query}")
        
        return {
            "student_profile": profile_dict,
            "is_strategy_query": result.is_strategy_query,
            "is_simulation": False
        }
        
    except Exception as e:
        print(f"[*] Warning: Profile extraction failed, using fallback: {e}")
        # Return empty baseline fallback
        return {
            "student_profile": {
                "cgpa": None,
                "skills": [],
                "weaknesses": [],
                "interests": [],
                "backlogs": 0,
                "projects_count": 0
            },
            "is_strategy_query": False,
            "is_simulation": False
        }
