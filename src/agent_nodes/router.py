import re
from typing import Dict, Any, List
from .company_utils import get_canonical_companies, normalize_company_name

def rule_based_router(query: str) -> dict:
    query_lower = query.lower()
    q_type = "eligibility" # default
    
    # 1. Conflict detection (check for conflict keywords, or multiple numbers with 'or'/'vs'/'versus')
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", query_lower)
    has_conflict_keywords = any(x in query_lower for x in ["conflict", "contradict", "different data", "differing data", "discrepancy", "mismatch", "error", "correct"])
    has_multi_numbers = len(set(numbers)) >= 2 and any(x in query_lower for x in ["or", "vs", "versus", "between"])
    
    if has_conflict_keywords or (has_multi_numbers and any(x in query_lower for x in ["cgpa", "cutoff", "package", "lpa", "salary", "pay"])):
        q_type = "conflict"
    # 2. Trend detection (check for growth/trend keywords or two years mentioned)
    elif any(x in query_lower for x in ["grew", "growth", "trend", "temporal", "increase", "decrease", "change", "rise", "over time", "comparison"]) or len(re.findall(r"\b20\d{2}\b", query_lower)) >= 2:
        q_type = "trend"
    # 3. Statistics detection (check for mathematical/ratio terms)
    elif any(x in query_lower for x in ["ratio", "package-to-cgpa", "package to cgpa", "package/cgpa", "offers", "min_offers", "max_offers", "avg_package", "average", "statistics", "stats"]):
        q_type = "statistics"
    # 4. Hiring stats detection (check for hiring role terms)
    elif any(x in query_lower for x in ["intern", "sde", "analyst", "officer", "hiring", "hires", "hired", "recruit", "recruitment"]):
        q_type = "hiring"
    # 5. Interview preparation detection (check for interview/rounds/focus terms)
    elif any(x in query_lower for x in ["round", "rounds", "topic", "topics", "focus", "prepare", "interview", "preparation", "tech", "focus", "subject"]):
        q_type = "interview_prep"
    # 6. Fallback/Out of corpus detection (questions about date, stock, visit, world, career etc.)
    elif any(x in query_lower for x in ["date", "visit", "stock", "price", "world", "career", "when"]):
        q_type = "fallback"
    # 7. Eligibility (default or specific keywords)
    elif any(x in query_lower for x in ["eligibility", "cgpa", "cutoff", "backlog", "backlogs", "bond", "require", "criteria"]):
        q_type = "eligibility"
        
    # Dynamic rule-based entity extractor
    entities = []
    canonical_companies = get_canonical_companies()
    
    # Split query into words and clean them
    query_words = re.findall(r'[a-zA-Z0-9&]+', query_lower)
    
    for word in query_words:
        # Avoid matching short/common words that might clash or cause false matches
        if len(word) < 2 or word in [
            'in', 'of', 'for', 'to', 'is', 'it', 'or', 'and', 'the', 'who', 'how', 
            'any', 'all', 'a', 'an', 'at', 'with', 'about', 'what', 'which', 'where',
            'c++', 'java', 'dsa', 'os', 'dbms', 'oops', 'rounds', 'round', 'cutoff', 'cutoffs'
        ]:
            continue
            
        # Try to normalize this word to a canonical company name
        normalized = normalize_company_name(word)
        if normalized in canonical_companies:
            if normalized not in entities:
                entities.append(normalized)
                
    return {"query_type": q_type, "entities": entities}


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    RouterNode: Uses LLM with structured output to classify user query intents and extract entities.
    Falls back to a robust rule-based parser if LLM execution fails.
    """
    query = state.get("user_query") or state.get("query") or ""
    
    # Initialize result structure
    query_type = "eligibility"
    entities = []
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from pydantic import BaseModel, Field
        from typing import List
        
        class RouterOutput(BaseModel):
            query_type: str = Field(description="Intent class of the query. Must be one of: 'eligibility', 'interview_prep', 'hiring', 'statistics', 'trend', 'conflict', 'fallback'")
            entities: List[str] = Field(description="List of company names extracted and normalized from the query. Convert variation names to their normalized canonical forms.")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0)
        structured_llm = llm.with_structured_output(RouterOutput)
        
        canonical_companies = get_canonical_companies()
        canonical_list_str = ", ".join(canonical_companies)
        
        prompt = (
            f"Analyze the following user placement query and determine the intent classification and extract entities.\n\n"
            f"Query: '{query}'\n\n"
            f"Canonical Company List: [{canonical_list_str}]\n\n"
            f"Instructions:\n"
            f"1. Classify the query intent into one of the following classes:\n"
            f"   - 'eligibility': Questions about cutoffs, GPA, active backlogs, bonds, and package thresholds.\n"
            f"   - 'interview_prep': Questions about selection rounds, topics, technical focus, preparing.\n"
            f"   - 'hiring': Questions about counts of hires, software development roles, intern/analyst count rankings.\n"
            f"   - 'statistics': Placement statistics aggregations, averages, or ratios (e.g. package-to-CGPA).\n"
            f"   - 'trend': Temporal reasoning or package comparisons across multiple years (e.g. 2021 to 2024 growth).\n"
            f"   - 'conflict': Explicit conflict or discrepancy questions (e.g. 'Is Amazon cutoff 6.4 or 7.0?', 'conflicting data').\n"
            f"   - 'fallback': Out of corpus, stock prices, subjective opinions, or vague/unrelated queries.\n\n"
            f"2. Extract company names mentioned. Resolve aliases to match exactly the canonical companies in the list."
        )
        
        result = structured_llm.invoke(prompt)
        query_type = result.query_type
        entities = result.entities
    except Exception as e:
        print(f"[*] Info: Router LLM structured call failed: {e}")
        print("[*] Falling back to rule-based classification...")
        # Fallback to rule-based routing
        rule_res = rule_based_router(query)
        query_type = rule_res["query_type"]
        entities = rule_res["entities"]
        
    return {
        "query_type": query_type,
        "entities": entities
    }
