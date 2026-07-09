# Healthcare Claims — HITL Review Agent with Token Optimisation

A realistic example of where LLM token optimisation actually matters:
**preparing a complete analyst package** for a human reviewer before they decide on a high-cost insurance claim.

---

## The problem

A $67,000 spine surgery claim lands on a human analyst''s desk.
Before they can make a decision they need:
- Patient profile from the patient DB
- Summaries of 3 medical PDFs (MRI report, surgeon consult, physio discharge note)
- Historical claims pattern for this patient
- Treatment guideline compliance check

Without automation this takes 30-45 minutes of manual reading.
With this agent it takes ~15 seconds and costs **$0.0004**.

---

## Architecture

```
Claim
  |
  v
ML Risk Model (<1ms, $0 LLM)
  |-- AUTO_APPROVE / AUTO_DENY  -->  done
  +-- NEEDS_REVIEW              -->  LangGraph Agent (7 nodes)
                                        |
                              fetch_patient_profile    (with DB retry)
                                        |
                              fetch_medical_records    (with DB retry)
                                        |
                              process_documents        <- LLM (token-optimised)
                                        |
                              fetch_claims_history     (with DB retry)
                                        |
                              analyze_claims_pattern   <- LLM (token-optimised)
                                        |
                              fetch_guidelines         (with loop detection)
                                        |
                              compile_analyst_brief    <- LLM (token-optimised)
                                        |
                              Human analyst reviews + decides
```

---

## Token optimisation techniques

| Technique | Where applied | Saving |
|-----------|--------------|--------|
| **Pre-extract relevant sections** | PDF text -> strip to FINDINGS / ASSESSMENT / PLAN before prompt | 54-60% |
| **Structured JSON prompts** | Ask for specific fields, not prose narrative | 56-62% |
| **Content-hash caching** | Same document seen again -> 0 tokens | 100% on hit |
| **Retry - DB only** | Failures loop back at the DB fetch node, not at the LLM | 0 extra tokens |
| **Loop detection** | node_visit_counts >= MAX_NODE_VISITS -> skip with partial data | prevents infinite spend |

See [RESULTS.md](RESULTS.md) for real numbers.

---

## 5 Scenarios (`python demo.py`)

| # | Claim | What fires | Key result |
|---|-------|-----------|------------|
| 1 | C001 - knee surgery | Full pipeline | 62% token saving vs verbose |
| 2 | C003 - chemo cycle | Patient fetch fails x2, history fails x1 | Recovers, 4/4 data complete |
| 3 | C004 - spine fusion (3 PDFs) | Full pipeline | 56% saving, 54% PDF extraction |
| 4 | C006 - same patient (C001 return) | PDF + history cache | 73% saving vs first visit |
| 5 | C001 - loop simulation | Guidelines returns empty x4 | Loop detected, brief compiled with 3/4 data |

---

## Setup

```bash
git clone https://github.com/aipankajmishra/agents_token_optimization.git
cd agents_token_optimization
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # add OPENAI_API_KEY
python demo.py
```

---

## Project structure

```
|- demo.py                      entry point
|- core/
|  |- state.py                  LangGraph TypedDict state
|  +- llm_engine.py             OpenAI wrapper + token tracker
|- data/
|  |- claims_data.py            6 mock claims
|  +- medical_records.py        patients, PDFs, history, guidelines
|- ml_models/
|  +- risk_model.py             XGBoost-style risk scorer
|- tools/
|  |- data_fetcher.py           DB fetch functions with retry simulation
|  +- document_processor.py    PDF extraction + LLM summarisation + cache
|- agents/
|  +- claims_review_agent.py    LangGraph StateGraph (7 nodes)
|- scenarios/
|  +- run_scenarios.py          5 scenarios
+- RESULTS.md                   real token/cost numbers
```
