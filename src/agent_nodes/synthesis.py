import os
from typing import Dict, Any
from langchain_core.documents import Document
from .company_utils import _load_env_file
from .web_search import tavily_search

def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    SynthesisNode: Compiles the final fact-anchored answer using retrieved contexts
    and appropriate source citations, incorporating conflict notifications if present.
    """
    query = state.get("user_query") or state.get("query") or ""
    is_strategy = state.get("is_strategy_query", False)
    
    if is_strategy:
        opps = state.get("opportunities") or []
        opps_str = ""
        if opps:
            for idx, o in enumerate(opps):
                opps_str += f"{idx+1}. Company: {o['company']}, Package: {o['package']} LPA, Cutoff CGPA: {o['min_cgpa']}, Max Backlogs: {o['max_backlogs']}, Tech Focus: {o['tech_focus']}, Bond: {o['bond']} Yrs (Skill Overlap Score: {o['skill_score']})\n"
        else:
            opps_str = "No eligible companies found matching the CGPA and backlog criteria.\n"
            
        system_prompt = (
            "You are an expert SVECW Placement Assistant compiling a final career advice report.\n"
            "The student has submitted their profile details. Below is the list of eligible companies "
            "and their requirements parsed from the database.\n\n"
            "Eligible Companies List:\n"
            f"{opps_str}\n\n"
            "Instructions:\n"
            "1. Present the matching companies in a clear, formatted markdown report, sorted by skill overlap score descending (and package LPA descending as a tie-breaker).\n"
            "2. For each company, summarize the cutoff CGPA, backlog allowance, tech focus/topics to prepare, and bond details.\n"
            "3. If no eligible companies are found, state that clearly and advise the student on what CGPA/backlogs to target.\n"
            "4. Provide a supportive, brief summary concluding the response."
        )
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{query}")
            ])
            chain = prompt_template | llm
            response = chain.invoke({"query": query})
            answer = response.content
        except Exception as e:
            print(f"[*] Info: Strategy Synthesis LLM call failed: {e}")
            answer = f"Here are your eligible companies:\n\n{opps_str}"
            
        return {
            "final_answer": answer,
            "sources": ["section_1:_company_eligibility_profiles"],
            "confidence": 0.95
        }

    conflict_detected = state.get("conflict_detected", False)
    conflict_details = state.get("conflict_details")

    
    # Gather all active context documents
    all_docs = []
    for field in ["eligibility_context", "interview_context", "hiring_context", "stats_context", "trend_context", "websearch_context"]:
        if state.get(field):
            all_docs.extend(state[field])
            
    # If no contexts are found, check if we should query Tavily dynamically
    if not all_docs:
        _load_env_file()
        if os.getenv("TAVILY_API_KEY"):
            print(f"[*] Context is empty. Triggering dynamic Tavily Web Search fallback for: '{query}'")
            all_docs = tavily_search(query)
            
    # Check out-of-corpus fallback if we STILL have no context
    if not all_docs:
        query_lower = query.lower()
        if any(x in query_lower for x in ["date", "visit", "stock", "price", "world", "career", "join", "experience"]):
            final_answer = "I apologize, but this information is not available in the Placement RAG dataset."
            return {
                "final_answer": final_answer,
                "sources": [],
                "confidence": 0.2
            }
            
    context_str = "\n\n".join([
        f"Document {i+1} [Section: {doc.metadata.get('section', 'general')}]:\n{doc.page_content}"
        for i, doc in enumerate(all_docs)
    ])
    
    # Check if we have web search context
    has_web_context = any(
        doc.metadata and "tavily" in str(doc.metadata.get("section", "")).lower() 
        for doc in all_docs
    )
    
    # Select prompt strategy based on conflict status or web context
    if conflict_detected and conflict_details:
        system_prompt = (
            f"You are an expert Placement Assistant compiling a final answer.\n"
            f"A placement data conflict has been detected:\n"
            f"Company: {conflict_details.get('company')}\n"
            f"Metric: {conflict_details.get('metric')}\n"
            f"Official Value: {conflict_details.get('official_value')}\n"
            f"Portal Value: {conflict_details.get('portal_value')}\n\n"
            f"Instructions:\n"
            f"1. Cite both the official source and the placement portal source.\n"
            f"2. Present the official source as the primary authority.\n"
            f"3. Clearly notify the user of the discrepancy (e.g., 'There are conflicting records...') "
            f"and explicitly advise them to verify the criteria with the official placement cell."
        )
    elif has_web_context:
        system_prompt = (
            "You are an expert Placement Assistant. Answer the user query using the web search results provided in the context below. "
            "Synthesize a clear, accurate, and comprehensive response based on the search context, citing the sources or URLs if available.\n\n"
            "Contexts:\n"
            f"{context_str}"
        )
    else:
        system_prompt = (
            "You are an expert Placement Assistant. Answer the user query using the provided context below. "
            "Rely strictly on the facts present in the contexts. If the context does not contain the answer or if the query asks about out-of-corpus information, "
            "you MUST reply exactly with: 'I apologize, but this information is not available in the Placement RAG dataset.'\n\n"
            "Instructions:\n"
            "1. When answering eligibility queries (like cutoffs or GPA requirements) for any company, always state both the CGPA cutoff, the maximum backlogs allowed (even if 0), and other details if available.\n"
            "2. If the context contains a 'Multi-Hop Reasoning Trace', present the step-by-step reasoning steps and the ranked list of companies exactly as written in the trace in your final answer, preserving all details such as packages (e.g., Qualcomm at 41.3 LPA and Amazon at 28.6 LPA).\n\n"
            "Contexts:\n"
            f"{context_str}"
        )
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0)
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        chain = prompt_template | llm
        response = chain.invoke({"query": query})
        answer = response.content
    except Exception as e:
        print(f"[*] Info: Synthesis LLM call failed: {e}")
        print("[*] Falling back to offline rule-based synthesis...")
        
        # Offline rule-based fallback answer
        mh_docs = [d for d in all_docs if d.metadata and "multi_hop_reasoning" in d.metadata.get("section", "")]
        web_docs = [d for d in all_docs if d.metadata and "tavily" in str(d.metadata.get("section", "")).lower()]
        
        if web_docs:
            summary_bullets = []
            for doc in web_docs:
                src = doc.metadata.get("source", "Web Search")
                summary_bullets.append(f"- [{src}] {doc.page_content}")
            answer = (
                "**[Offline Fallback Web Answer]**\n"
                "Here are the web search results retrieved for your query:\n\n" +
                "\n".join(summary_bullets)
            )
        elif state.get("query_type") == "fallback" or any(x in query.lower() for x in ["date", "visit", "stock", "price", "world", "career", "join", "experience"]):
            answer = "I apologize, but this information is not available in the Placement RAG dataset."
        elif mh_docs:
            answer = mh_docs[0].page_content
        elif conflict_detected and conflict_details:
            answer = (
                f"**[Offline Fallback Answer]**\n"
                f"There are conflicting records. A placement data conflict has been detected for **{conflict_details['company']}**.\n"
                f"The official criteria states {conflict_details['official_value']}, while the placement portal lists {conflict_details['portal_value']}.\n"
                f"Please verify this discrepancy directly with the official placement cell."
            )
        else:
            summary_bullets = []
            for doc in all_docs:
                sect = doc.metadata.get("section", "general").upper()
                summary_bullets.append(f"- [{sect}] {doc.page_content}")
            answer = (
                f"**[Offline Fallback Answer]**\n"
                f"Here is the retrieved context related to your query '{query}':\n\n" +
                "\n".join(summary_bullets)
            )
            
    sources = list(set([doc.metadata.get("section", "general") for doc in all_docs if doc.metadata]))
    confidence = 0.5 if conflict_detected else (0.95 if all_docs else 0.2)
    
    return {
        "final_answer": answer,
        "sources": sources,
        "confidence": confidence
    }
