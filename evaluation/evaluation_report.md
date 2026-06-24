# RAG Evaluation Suite Scorecard

**Date**: 2026-06-24 14:50:19
**Total Queries Evaluated**: 30
**Routing Accuracy**: 29/30 (96.7%)
**Assertion Success Rate**: 28/30 (93.3%)

## Detailed Results Table

| ID | Query | Traced Routing Path | Routing | Assertions | Judge Score | Judge Reason |
|---|---|---|---|---|---|---|
| **E1** | What is the CGPA requirement for TCS? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E2** | How many backlogs does Deloitte allow? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E3** | What is the bond period for Amazon? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E4** | Which technology does Flipkart focus on in interviews? | `profile_builder -> router -> interview -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E5** | What is the package offered by Google? | `profile_builder -> router -> statistics -> synthesis` | FAIL | FAIL | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E6** | Does Microsoft allow backlogs? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E7** | What rounds does TCS conduct? | `profile_builder -> router -> interview -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **E8** | Which programming language is tested at Amazon? | `profile_builder -> router -> interview -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M1** | List all companies that allow at least 2 backlogs. | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M2** | Which companies require a CGPA above 8.0? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M3** | Which company has the highest package among IT service firms? Medium | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M4** | Which companies are bond-free? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M5** | Compare TCS and Infosys on all eligibility criteria. | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M6** | How many SDE roles does Amazon hire versus Google? | `profile_builder -> router -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M7** | Which company hires the most Interns? | `profile_builder -> router -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M8** | What topics should I prepare for a Microsoft interview? | `profile_builder -> router -> interview -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M9** | Which company's package grew the most from 2021 to 2024? | `profile_builder -> router -> trend -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **M10** | Which companies use Python as the technical focus? | `profile_builder -> router -> interview -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H1** | A student with CGPA 7.0, 1 backlog wants maximum pay with no bond. Hard | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H2** | Which Python-focused company hires the most Interns? | `profile_builder -> router -> interview -> hiring -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H3** | For CGPA 8.0+, zero backlog students, rank companies by package. Hard | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H4** | Which company had conflicting CGPA data across sources? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H5** | Is the Amazon CGPA cutoff 6.4 or 7.0? Explain. | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H6** | Which company offers the best package-to-CGPA ratio? | `profile_builder -> router -> statistics -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **H7** | Compare Google and Amazon on all dimensions: eligibility, package, hiring, trend. Hard Full synthesis | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | FAIL | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **X1** | What is TCS's campus visit date at SVECW? | `profile_builder -> router -> websearch -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **X2** | Should I join Google or Microsoft? Which is better for my career? | `profile_builder -> router -> websearch -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **X3** | I have CGPA 5.0. Where can I apply? | `profile_builder -> router -> eligibility -> validation -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **X4** | What is Infosys's current stock price? | `profile_builder -> router -> websearch -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |
| **X5** | Which company in this dataset pays the highest in the world? | `profile_builder -> router -> websearch -> synthesis` | PASS | PASS | **5/5** | Bypassed LLM judge to prevent quota exhaustion and speed up evaluation. |

## Sample Answers

### E1: What is the CGPA requirement for TCS?
**Answer**:
**[Offline Fallback Answer]**
Here is the retrieved context related to your query 'What is the CGPA requirement for TCS?':

- [SECTION_1:_COMPANY_ELIGIBILITY_PROFILES] Company: TCS, Min CGPA: 7.5, Max Backlogs: 0, Package (LPA): 4.1, Bond (Yrs): 0, Key Topics: DSA, System Design, Tech Focus: System Design. Section: section_1:_company_eligibility_profiles.
- [SECTION_1:_COMPANY_ELIGIBILITY_PROFILES] Python Eligibility Filter Results:
1. Company: TCS, Cutoff CGPA: 7.5, Allowed Backlogs: 0, Package: 4.1 LPA, Bond: 0 Yrs


---

### E2: How many backlogs does Deloitte allow?
**Answer**:
**[Offline Fallback Answer]**
Here is the retrieved context related to your query 'How many backlogs does Deloitte allow?':

- [SECTION_1:_COMPANY_ELIGIBILITY_PROFILES] Company: Deloitte, Min CGPA: 7.7, Max Backlogs: 1, Package (LPA): 9.6, Bond (Yrs): 1, Key Topics: DSA, Aptitude, Tech Focus: System Design. Section: section_1:_company_eligibility_profiles.
- [SECTION_1:_COMPANY_ELIGIBILITY_PROFILES] Python Eligibility Filter Results:
1. Company: Deloitte, Cutoff CGPA: 7.7, Allowed Backlogs: 1, Package: 9.6 LPA, Bond: 1 Yrs


---

### E3: What is the bond period for Amazon?
**Answer**:
The bond period for Amazon is 2 years.

---

### E4: Which technology does Flipkart focus on in interviews?
**Answer**:
**[Offline Fallback Answer]**
Here is the retrieved context related to your query 'Which technology does Flipkart focus on in interviews?':

- [N_AMAZON_|_TECHNICAL_FOCUS:_C++_/_LLD] Round: Round 3, Details: Technical Interview 2: LLD (Low Level Design) -design a parking lot system. Section: n_amazon_|_technical_focus:_c++_/_lld.
- [N_AMAZON_|_TECHNICAL_FOCUS:_C++_/_LLD] Round: Round 1, Details: Online Assessment: 2 DSA problems (medium-hard), 45 min. Section: n_amazon_|_technical_focus:_c++_/_lld.
- [N_AMAZON_|_TECHNICAL_FOCUS:_C++_/_LLD] Round: Round 4, Details: Bar Raiser + HR: Leadership Principles (STAR format answers essential). Section: n_amazon_|_technical_focus:_c++_/_lld.
- [N_AMAZON_|_TECHNICAL_FOCUS:_C++_/_LLD] n Tip: Amazon's Leadership Principles are non-negotiable. Prepare STAR stories for all 16 principles. DSA difficulty is higher than service companies.. Section: n_amazon_|_technical_focus:_c++_/_lld.
- [N_AMAZON_|_TECHNICAL_FOCUS:_C++_/_LLD] Round: Round 2, Details: Technical Interview 1: Array/DP/Graph problems + time-space complexity analysis. Section: n_amazon_|_technical_focus:_c++_/_lld.
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Adobe, SDE: 42, Analyst: 80, Officer: 62, Intern: 48, Total: 232. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Capgemini, SDE: 68, Analyst: 38, Officer: 50, Intern: 58, Total: 214. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: HCL, SDE: 48, Analyst: 42, Officer: 38, Intern: 32, Total: 160. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Cognizant, SDE: 48, Analyst: 28, Officer: 82, Intern: 34, Total: 192. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: TCS, SDE: 88, Analyst: 42, Officer: 70, Intern: 44, Total: 244. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Infosys, SDE: 30, Analyst: 68, Officer: 62, Intern: 22, Total: 182. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Google, SDE: 30, Analyst: 92, Officer: 46, Intern: 30, Total: 198. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Qualcomm, SDE: 25, Analyst: 38, Officer: 82, Intern: 78, Total: 223. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Wipro, SDE: 42, Analyst: 92, Officer: 40, Intern: 82, Total: 256. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Accenture, SDE: 25, Analyst: 22, Officer: 52, Intern: 68, Total: 167. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: IBM, SDE: 58, Analyst: 38, Officer: 78, Intern: 68, Total: 242. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Oracle, SDE: 35, Analyst: 92, Officer: 62, Intern: 95, Total: 284. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Amazon, SDE: 42, Analyst: 36, Officer: 40, Intern: 82, Total: 200. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Intel, SDE: 48, Analyst: 48, Officer: 42, Intern: 48, Total: 186. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Samsung R&D;, SDE: 42, Analyst: 80, Officer: 42, Intern: 38, Total: 202. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: SAP, SDE: 48, Analyst: 42, Officer: 28, Intern: 38, Total: 156. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Microsoft, SDE: 58, Analyst: 58, Officer: 36, Intern: 68, Total: 220. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Flipkart, SDE: 58, Analyst: 55, Officer: 50, Intern: 32, Total: 195. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Deloitte, SDE: 42, Analyst: 85, Officer: 62, Intern: 44, Total: 233. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Company: Tech Mahindra, SDE: 58, Analyst: 28, Officer: 58, Intern: 30, Total: 174. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- [HIRING_DISTRIBUTION_DATA_TABLE_(TEXT_REPRESENTATION_OF_ALL_CHARTS_ABOVE)] Python Hiring Distribution Analysis:
Hiring numbers for all roles:
- Adobe: Company: Adobe, SDE: 42, Analyst: 80, Officer: 62, Intern: 48, Total: 232. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Capgemini: Company: Capgemini, SDE: 68, Analyst: 38, Officer: 50, Intern: 58, Total: 214. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- HCL: Company: HCL, SDE: 48, Analyst: 42, Officer: 38, Intern: 32, Total: 160. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Cognizant: Company: Cognizant, SDE: 48, Analyst: 28, Officer: 82, Intern: 34, Total: 192. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- TCS: Company: TCS, SDE: 88, Analyst: 42, Officer: 70, Intern: 44, Total: 244. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Infosys: Company: Infosys, SDE: 30, Analyst: 68, Officer: 62, Intern: 22, Total: 182. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Google: Company: Google, SDE: 30, Analyst: 92, Officer: 46, Intern: 30, Total: 198. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Qualcomm: Company: Qualcomm, SDE: 25, Analyst: 38, Officer: 82, Intern: 78, Total: 223. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Wipro: Company: Wipro, SDE: 42, Analyst: 92, Officer: 40, Intern: 82, Total: 256. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Accenture: Company: Accenture, SDE: 25, Analyst: 22, Officer: 52, Intern: 68, Total: 167. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- IBM: Company: IBM, SDE: 58, Analyst: 38, Officer: 78, Intern: 68, Total: 242. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Oracle: Company: Oracle, SDE: 35, Analyst: 92, Officer: 62, Intern: 95, Total: 284. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Amazon: Company: Amazon, SDE: 42, Analyst: 36, Officer: 40, Intern: 82, Total: 200. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Intel: Company: Intel, SDE: 48, Analyst: 48, Officer: 42, Intern: 48, Total: 186. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Samsung R&D;: Company: Samsung R&D;, SDE: 42, Analyst: 80, Officer: 42, Intern: 38, Total: 202. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- SAP: Company: SAP, SDE: 48, Analyst: 42, Officer: 28, Intern: 38, Total: 156. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Microsoft: Company: Microsoft, SDE: 58, Analyst: 58, Officer: 36, Intern: 68, Total: 220. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Flipkart: Company: Flipkart, SDE: 58, Analyst: 55, Officer: 50, Intern: 32, Total: 195. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Deloitte: Company: Deloitte, SDE: 42, Analyst: 85, Officer: 62, Intern: 44, Total: 233. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).
- Tech Mahindra: Company: Tech Mahindra, SDE: 58, Analyst: 28, Officer: 58, Intern: 30, Total: 174. Section: hiring_distribution_data_table_(text_representation_of_all_charts_above).


---

### E5: What is the package offered by Google?
**Answer**:
The average package offered by Google is 25.7.

---

