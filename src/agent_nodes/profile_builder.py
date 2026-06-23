from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class StudentProfile(BaseModel):
    cgpa: Optional[float] = Field(None, description="Student's CGPA, normalized between 0.0 and 10.0")
    skills: List[str] = Field(default_factory=list, description="Extracted tech stack, programming languages, or domain skills (e.g., Python, SQL, Web Dev)")
    weaknesses: List[str] = Field(default_factory=list, description="Explicitly mentioned areas of improvement or weaknesses (e.g., DSA, OS, DBMS)")
    interests: List[str] = Field(default_factory=list, description="Target job roles or fields (e.g., SDE, Analyst, Intern)")
    backlogs: int = Field(default=0, description="Number of active backlogs mentioned. Defaults to 0")
    projects_count: int = Field(default=0, description="Number of projects or project experience mentioned. Defaults to 0")
    is_strategy_query: bool = Field(
        description="True if the user is describing their profile, asking for placement suggestions, requesting a strategy plan, asking what-if simulations, or querying eligible opportunities based on their situation. False if it is a simple factual lookup question about a company (e.g., CGPA requirement for TCS, Google package, what rounds TCS conducts) or a general out-of-corpus query."
    )

def profile_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ProfileBuilderNode: Parses free-form queries into structured student profiles.
    If 'student_profile' was already supplied directly via the frontend form payload,
    it uses it directly and flags 'is_strategy_query' as True.
    """
    query = state.get("user_query") or state.get("query") or ""
    
    # 1. If student profile was passed in state directly, use it
    input_profile = state.get("student_profile")
    if input_profile and isinstance(input_profile, dict) and input_profile.get("cgpa") is not None:
        print("[*] ProfileBuilder: Found pre-populated profile in state.")
        return {
            "student_profile": input_profile,
            "is_strategy_query": True
        }
        
    # 2. Rule-based bypass for factual/lookup queries to save Gemini API quota
    query_lower = query.lower()
    strategy_keywords = ["strategy", "plan", "roadmap", "advisor", "suggest", "recommend", "prepare me", "pathway", "career path", "what should i do", "advice", "what to do"]
    has_strategy_keyword = any(k in query_lower for k in strategy_keywords)
    
    if not has_strategy_keyword:
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
            "is_strategy_query": False
        }

    # 3. Extract structured profile using Gemini
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0)
        structured_llm = llm.with_structured_output(StudentProfile)
        
        prompt = (
            f"You are an expert Placement Assistant. Analyze the student's message and build a structured representation "
            f"of their profile and detect if their query is a career planning/strategy request.\n\n"
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
        
        print(f"[*] ProfileBuilder: Parsed profile -> {profile_dict}")
        print(f"[*] ProfileBuilder: is_strategy_query -> {result.is_strategy_query}")
        
        return {
            "student_profile": profile_dict,
            "is_strategy_query": result.is_strategy_query
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
            "is_strategy_query": False
        }
