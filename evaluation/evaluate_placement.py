import sys
import os
import time
import json
import re

# Add workspace base directory to system path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.rag_pipeline import build_placement_graph
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Define LLM Judge Output Schema
class JudgeResult(BaseModel):
    score: int = Field(description="Factual accuracy score from 1 to 5.")
    reason: str = Field(description="Short one-sentence explanation of the score.")

def get_gemini_judge(query: str, generated_answer: str, ground_truth: str) -> dict:
    """
    Independent Gemini-based judge to evaluate output semantic and factual accuracy.
    Includes rate-limit handling and graceful fallback.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=1)
        structured_llm = llm.with_structured_output(JudgeResult)
        
        prompt = (
            f"You are an independent RAG evaluation judge.\n"
            f"Evaluate the generated answer against the ground truth reference facts for the user query.\n\n"
            f"User Query: '{query}'\n"
            f"Generated Answer: '{generated_answer}'\n"
            f"Ground Truth Reference: '{ground_truth}'\n\n"
            f"Criteria:\n"
            f"- 5: Completely accurate, contains all critical numbers/facts, resolves conflicts correctly (if any), and falls back gracefully (if out-of-corpus).\n"
            f"- 4: Factual and helpful, but missing minor context.\n"
            f"- 3: Partially correct, but misses some important facts or contains slight inaccuracies.\n"
            f"- 2: Mostly incorrect or contains major hallucinations/inaccuracies.\n"
            f"- 1: Completely wrong, irrelevant, or fails to perform fallback when information is missing.\n\n"
            f"Evaluate and output the score (1-5) and reason."
        )
        
        # Call model with basic retry handling for rate limits
        for delay in [3, 6, 12]:
            try:
                res = structured_llm.invoke(prompt)
                return {"score": res.score, "reason": res.reason}
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(delay)
                else:
                    raise e
        return {"score": 0, "reason": "Skipped due to persistent LLM judge rate limit (429)."}
    except Exception as e:
        return {"score": 0, "reason": f"Judge failed: {e}"}

def run_evaluation():
    graph = build_placement_graph()
    
    # 30-Query Evaluation Suite: ID -> (Query, Required Node, Expected Substrings, Ground Truth)
    test_cases = {
        # EASY QUERIES
        "E1": (
            "What is the CGPA requirement for TCS?",
            "eligibility",
            ["tcs", "7.5", "0"],
            "TCS requires a minimum CGPA of 7.5 and allows 0 backlogs."
        ),
        "E2": (
            "How many backlogs does Deloitte allow?",
            "eligibility",
            ["deloitte", "1"],
            "Deloitte allows a maximum of 1 backlog."
        ),
        "E3": (
            "What is the bond period for Amazon?",
            "eligibility",
            ["amazon", "2"],
            "Amazon has a bond period of 2 years."
        ),
        "E4": (
            "Which technology does Flipkart focus on in interviews?",
            "interview",
            ["flipkart", "python"],
            "Flipkart focuses on Python in its interviews."
        ),
        "E5": (
            "What is the package offered by Google?",
            "eligibility",
            ["google", "42"],
            "Google offers a package of 42.0 LPA."
        ),
        "E6": (
            "Does Microsoft allow backlogs?",
            "eligibility",
            ["microsoft", "1"],
            "Yes, Microsoft allows a maximum of 1 backlog."
        ),
        "E7": (
            "What rounds does TCS conduct?",
            "interview",
            ["tcs", "online", "technical", "managerial"],
            "TCS conducts 3 rounds: Online Assessment, Technical Interview, and Managerial/HR."
        ),
        "E8": (
            "Which programming language is tested at Amazon?",
            "interview",
            ["amazon", "c++"],
            "Amazon tests C++ as its technical focus."
        ),
        
        # MEDIUM QUERIES
        "M1": (
            "List all companies that allow at least 2 backlogs.",
            "eligibility",
            ["flipkart", "ibm", "tech mahindra", "qualcomm", "samsung"],
            "Companies allowing at least 2 backlogs are Flipkart, IBM, Tech Mahindra, Qualcomm, and Samsung R&D."
        ),
        "M2": (
            "Which companies require a CGPA above 8.0?",
            "eligibility",
            ["accenture", "cognizant", "sap", "hcl", "tech mahindra"],
            "Companies requiring CGPA above 8.0 are Accenture (8.2), Cognizant (8.4), SAP (8.4), HCL (8.4), and Tech Mahindra (8.1)."
        ),
        "M3": (
            "Which company has the highest package among IT service firms? Medium",
            "eligibility",
            ["infosys", "42.9"],
            "Infosys offers the highest package among IT service firms with 42.9 LPA."
        ),
        "M4": (
            "Which companies are bond-free?",
            "eligibility",
            ["tcs", "infosys", "microsoft", "ibm", "intel"],
            "The bond-free companies are TCS, Infosys, Microsoft, IBM, and Intel."
        ),
        "M5": (
            "Compare TCS and Infosys on all eligibility criteria.",
            "eligibility",
            ["tcs", "infosys", "7.5", "8.0", "4.1", "42.9", "0"],
            "TCS requires 7.5 CGPA, 0 backlogs, 4.1 LPA, 0 bond. Infosys requires 8.0 CGPA, 0 backlogs, 42.9 LPA, 0 bond."
        ),
        "M6": (
            "How many SDE roles does Amazon hire versus Google?",
            "hiring",
            ["amazon", "google", "42", "30"],
            "Amazon hires 42 SDE roles, while Google hires 30 SDE roles."
        ),
        "M7": (
            "Which company hires the most Interns?",
            "hiring",
            ["oracle", "95"],
            "Oracle hires the most interns with 95."
        ),
        "M8": (
            "What topics should I prepare for a Microsoft interview?",
            "interview",
            ["microsoft", "dsa", "os", "dbms"],
            "For Microsoft, prepare DSA (Trees, Graphs), OS (threading, deadlocks), and DBMS."
        ),
        "M9": (
            "Which company's package grew the most from 2021 to 2024?",
            "trend",
            ["infosys", "6.9"],
            "Infosys package grew the most from 2021 to 2024 (by 6.9 LPA, from 36.0 to 42.9)."
        ),
        "M10": (
            "Which companies use Python as the technical focus?",
            "interview",
            ["flipkart", "google", "oracle", "intel"],
            "Flipkart, Google, Oracle, and Intel use Python as their technical focus."
        ),
        
        # HARD QUERIES
        "H1": (
            "A student with CGPA 7.0, 1 backlog wants maximum pay with no bond. Hard",
            "eligibility",
            ["microsoft", "21.4"],
            "The student qualifies for Microsoft which offers 21.4 LPA with no bond."
        ),
        "H2": (
            "Which Python-focused company hires the most Interns?",
            "interview",
            ["oracle", "95"],
            "Oracle is the Python-focused company hiring the most Interns (95)."
        ),
        "H3": (
            "For CGPA 8.0+, zero backlog students, rank companies by package. Hard",
            "eligibility",
            ["infosys", "42.9", "cognizant", "42.3", "intel", "41.4", "qualcomm", "41.3"],
            "Ranked companies for CGPA 8.0+, zero backlogs are Infosys (42.9 LPA), Cognizant (42.3 LPA), Intel (41.4 LPA), Qualcomm (41.3 LPA), and Capgemini (38.3 LPA)."
        ),
        "H4": (
            "Which company had conflicting CGPA data across sources?",
            "eligibility",
            ["discrepancy", "conflict"],
            "TCS, Amazon, Google, Infosys, and Microsoft have conflicting CGPA data across sources."
        ),
        "H5": (
            "Is the Amazon CGPA cutoff 6.4 or 7.0? Explain.",
            "eligibility",
            ["6.4", "7.0", "conflict", "discrepancy"],
            "Amazon CGPA cutoff is conflicting: the official source states 6.4, while the portal lists 7.0."
        ),
        "H6": (
            "Which company offers the best package-to-CGPA ratio?",
            "statistics",
            ["qualcomm", "5.7"],
            "Qualcomm offers the best package-to-CGPA ratio of 5.74 (41.3 LPA / 7.2 CGPA)."
        ),
        "H7": (
            "Compare Google and Amazon on all dimensions: eligibility, package, hiring, trend. Hard Full synthesis",
            "eligibility",
            ["google", "amazon", "7.4", "6.4", "42.0", "28.6", "198", "200"],
            "Google (7.4 CGPA, 42.0 LPA, 198 hires, 38.0 to 42.0 LPA trend) vs Amazon (6.4 CGPA, 28.6 LPA, 200 hires, 22.0 to 28.6 LPA trend)."
        ),
        
        # EXPERT QUERIES (FALLBACKS)
        "X1": (
            "What is TCS's campus visit date at SVECW?",
            "websearch",
            ["available", "dataset"],
            "I apologize, but this information is not available in the Placement RAG dataset."
        ),
        "X2": (
            "Should I join Google or Microsoft? Which is better for my career?",
            "websearch",
            ["google", "microsoft"],
            "Career preference is subjective; Google offers 42.0 LPA and Microsoft offers 21.4 LPA."
        ),
        "X3": (
            "I have CGPA 5.0. Where can I apply?",
            "eligibility",
            ["no company", "5.0"],
            "No company in this dataset has a CGPA cutoff <= 5.0."
        ),
        "X4": (
            "What is Infosys's current stock price?",
            "websearch",
            ["available", "dataset"],
            "I apologize, but this information is not available in the Placement RAG dataset."
        ),
        "X5": (
            "Which company in this dataset pays the highest in the world?",
            "websearch",
            ["available", "dataset"],
            "I apologize, but this information is not available in the Placement RAG dataset."
        )
    }
    
    print("==================================================")
    print("[*] Running Complete 30-Query Evaluation Suite...")
    print("==================================================")
    
    results = []
    passed_asserts = 0
    passed_routing = 0
    total_cases = len(test_cases)
    
    for case_id, (query, key_node, expected_substrings, ground_truth) in test_cases.items():
        print(f"\n[{case_id}] Running Query: '{query}'")
        
        initial_state = {
            "user_query": query,
            "query": query
        }
        
        # Sleep for a small delay to avoid rate limit spikes on Gemini API
        time.sleep(3.0)
        
        try:
            # Trace nodes programmatically using stream
            node_history = []
            final_state = {}
            for event in graph.stream(initial_state):
                for node_name in event.keys():
                    node_history.append(node_name)
                    # Merge states
                    final_state.update(event[node_name])
            
            final_answer = final_state.get("final_answer", "")
            query_type = final_state.get("query_type", "unknown")
            entities = final_state.get("entities", [])
            
            # Check Routing
            routing_pass = False
            if key_node in node_history:
                routing_pass = True
                passed_routing += 1
            
            # Check Deterministic Keyword Assertions
            missing_substrings = []
            for sub in expected_substrings:
                if sub.lower() not in final_answer.lower():
                    missing_substrings.append(sub)
            
            assert_pass = len(missing_substrings) == 0
            if not assert_pass:
                # Fallback relaxed check for out-of-corpus messages
                if "apologize" in final_answer.lower() or "not available" in final_answer.lower() or "not in" in final_answer.lower():
                    if case_id in ["X1", "X3", "X4", "X5"]:
                        assert_pass = True
            
            if assert_pass:
                passed_asserts += 1
            
            # Run LLM-as-a-judge
            judge_res = get_gemini_judge(query, final_answer, ground_truth)
            judge_score = judge_res.get("score", 0)
            judge_reason = judge_res.get("reason", "N/A")
            
            print(f"  -> Path Traced: {' -> '.join(node_history)}")
            print(f"  -> Routing Status: {'PASS' if routing_pass else 'FAIL'} (Expected node: '{key_node}')")
            print(f"  -> Assertion Status: {'PASS' if assert_pass else 'FAIL'} (Missing: {missing_substrings})")
            print(f"  -> LLM Judge Score: {judge_score}/5 ({judge_reason})")
            
            results.append({
                "id": case_id,
                "query": query,
                "path": " -> ".join(node_history),
                "routing_status": "PASS" if routing_pass else "FAIL",
                "assert_status": "PASS" if assert_pass else "FAIL",
                "judge_score": judge_score,
                "judge_reason": judge_reason,
                "answer": final_answer
            })
            
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
            results.append({
                "id": case_id,
                "query": query,
                "path": "ERROR",
                "routing_status": "ERROR",
                "assert_status": "ERROR",
                "judge_score": 0,
                "judge_reason": f"Execution failed: {e}",
                "answer": ""
            })
            
    # Generate evaluation scorecard report in Markdown
    # Save the report in the evaluation directory as well as the brain folder
    os.makedirs(os.path.join(base_dir, "evaluation"), exist_ok=True)
    report_paths = [
        os.path.join(base_dir, "evaluation", "evaluation_report.md"),
        r"C:\Users\neeha\.gemini\antigravity-ide\brain\4b640ab1-0cfd-47d7-97b0-b4f9324ae298\evaluation_report.md"
    ]
    
    for report_path in report_paths:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# RAG Evaluation Suite Scorecard\n\n")
                f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Queries Evaluated**: {total_cases}\n")
                f.write(f"**Routing Accuracy**: {passed_routing}/{total_cases} ({passed_routing/total_cases*100:.1f}%)\n")
                f.write(f"**Assertion Success Rate**: {passed_asserts}/{total_cases} ({passed_asserts/total_cases*100:.1f}%)\n\n")
                
                f.write("## Detailed Results Table\n\n")
                f.write("| ID | Query | Traced Routing Path | Routing | Assertions | Judge Score | Judge Reason |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for r in results:
                    judge_score_str = f"{r['judge_score']}/5" if r['judge_score'] > 0 else "N/A"
                    f.write(f"| **{r['id']}** | {r['query']} | `{r['path']}` | {r['routing_status']} | {r['assert_status']} | **{judge_score_str}** | {r['judge_reason']} |\n")
                    
                f.write("\n## Sample Answers\n\n")
                for r in results[:5]:
                    f.write(f"### {r['id']}: {r['query']}\n")
                    f.write(f"**Answer**:\n{r['answer']}\n\n")
                    f.write("---\n\n")
        except Exception as e:
            print(f"[*] Warning: Could not write report to {report_path}: {e}")
            
    print("\n==================================================")
    print(f"[*] Evaluation Finished!")
    print(f"    - Routing Accuracy: {passed_routing}/{total_cases}")
    print(f"    - Assertions Passed: {passed_asserts}/{total_cases}")
    print("==================================================")
    
    if passed_routing >= 24 and passed_asserts >= 24:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()
