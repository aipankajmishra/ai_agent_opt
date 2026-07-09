# Healthcare Claims Review — Complete Flow Walkthrough

This document explains **every single step** of the stress test demo, including where data is fetched, where the LLM is called, where caching happens, and how the optimized route differs from the unoptimized route.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point: demo.py](#2-entry-point-demopy)
3. [Stress Test Orchestrator: stress_test.py](#3-stress-test-orchestrator-stress_testpy)
4. [ML Risk Model: risk_model.py](#4-ml-risk-model-risk_modelpy)
5. [Optimized Route: optimized_agent.py](#5-optimized-route-optimized_agentpy)
6. [Unoptimized Route: unoptimized_agent.py](#6-unoptimized-route-unoptimized_agentpy)
7. [Document Processor: document_processor.py](#7-document-processor-document_processorpy)
8. [LLM Engine: llm_engine.py](#8-llm-engine-llm_enginepy)
9. [Semantic Cache: semantic_cache.py](#9-semantic-cache-semantic_cachepy)
10. [Context Cache: context_cache.py](#10-context-cache-context_cachepy)
11. [Data Fetcher: data_fetcher.py](#11-data-fetcher-data_fetcherpy)
12. [End-to-End Flow for a Single Claim](#12-end-to-end-flow-for-a-single-claim)
13. [The 5 Optimization Layers](#13-the-5-optimization-layers)
14. [Part B: Failure Scenarios](#14-part-b-failure-scenarios)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            demo.py (entry point)                              │
│                                 │                                              │
│                          run_stress_test()                                     │
│                            /          \                                        │
│                   Part A (8 rounds)   Part B (3 failure scenarios)             │
│                    /              \                                             │
│          run_optimized()      run_unoptimized()                                │
│                │                    │                                          │
│    5 optimization layers    ZERO optimizations                                 │
│    parallel subagents       sequential execution                               │
│    retry + loop detection   no retry, no loops                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Concept:** Every claim goes through BOTH routes with the SAME risk score, same data, same everything. The output compares them side-by-side so we can measure exactly how much the optimizations save.

---

## 2. Entry Point: demo.py

**File:** `demo.py`  
**Role:** Prints a header explaining the test, loads the OpenAI API key from `.env`, then calls `run_stress_test()`.

```
python demo.py
```

**What it does step-by-step:**

1. **Prints a banner** showing the two routes and the 5 optimization layers
2. **Reads `OPENAI_API_KEY`** from environment variables (via `.env` file using `python-dotenv`)
3. **Warns if the API key is missing** or is the placeholder `sk-your...`. Without a real key:
   - Input token counts are still **accurate** (measured locally with `tiktoken`)
   - Output token counts will be **wrong** (the LLM returns an error string instead of a real response)
   - Token-saving **ratios** between routes are still meaningful (driven by prompt size differences, not output size)
4. **Calls `run_stress_test()`** from `scenarios/stress_test.py`

---

## 3. Stress Test Orchestrator: stress_test.py

**File:** `scenarios/stress_test.py`  
**Role:** The main orchestrator. Runs all claims through both routes, collects results, prints comparisons, and writes to `RESULTS.md`.

### Part A: Random-Score Rounds

For each of 4 claims (`C001`, `C003`, `C004`, `C006`), 2 rounds are run:

```
For each claim_id in ["C001", "C003", "C004", "C006"]:
    claim = CLAIMS[claim_id]           # Get claim data from data/claims_data.py
    
    # Round 1
    r1 = random.random()               # Random score 0.0–1.0
    decision = _risk_model.get_decision(r1)    # Determine decision band
    _, features = _risk_model.predict(claim, force_score=r1)
    
    opt_result   = run_optimized(claim, r1, decision, features)
    unopt_result = run_unoptimized(claim, r1, decision, features)
    
    # Print side-by-side comparison table
    
    # Round 2
    r2 = random.random()
    while abs(r2 - r1) < 0.01:         # Ensure r2 is different from r1
        r2 = random.random()
    
    opt_result2   = run_optimized(claim, r2, decision2, features2)
    unopt_result2 = run_unoptimized(claim, r2, decision2, features2)
    
    # Print Round 2 comparison + cross-round delta
```

The `stress_test.py` file also handles:

- **Printing comparison tables** via `print_round_comparison()` — shows token counts, costs, API calls, cache hits, and savings percentages side by side
- **Cross-round delta** via `print_cross_round_delta()` — shows how the decision changed when the risk score changed
- **Markdown generation** via `_findings_to_markdown()` — writes the full report to `RESULTS.md`
- **Aggregation** — accumulates all token and cost totals for the Part A summary

### Part B: Failure Scenarios

After Part A, `run_failure_scenarios()` is called, which runs 3 injected-failure cases:

| Scenario | What Fails | What It Tests |
|---|---|---|
| **F1** | Patient DB fails 2 times | Retry resilience |
| **F2** | Guidelines DB always returns empty | Loop detection |
| **F3** | Records + History both fail once | Multi-resource retry |

All failure scenarios use a forced `risk_score=0.45` (NEEDS_REVIEW band) so the LLM pipeline actually runs.

---

## 4. ML Risk Model: risk_model.py

**File:** `ml_models/risk_model.py`  
**Role:** Simulates an XGBoost model that assigns a risk score to each claim.

**In stress-test mode (Part A):**
- `force_score` is a random number between 0 and 1 (passed from the stress test)
- The model does NOT compute features from claim data — it just returns the injected random score
- This simulates the "XGBoost returns random.random()" behavior described in the header

**In production mode (no force_score):**
- Computes actual features: `amount_vs_avg`, `diagnosis_risk`, `provider_history`, `member_history`, `procedure_match`
- Uses weighted combination to produce a real risk score

**Decision Bands (`get_decision()`):**

| Risk Score | Decision | What Happens |
|---|---|---|
| **< 0.3** | `AUTO_APPROVE` | No LLM called at all. Both routes return immediately with 0 tokens. |
| **0.3–0.7** | `NEEDS_REVIEW` | FULL pipeline executes. All LLM calls are made. This is where savings are measured. |
| **> 0.7** | `AUTO_DENY` | No LLM called. Claim is auto-rejected by ML. |

**Where LLM is used here:** NOWHERE. This is a pure Python/XGBoost simulator. No API calls.

---

## 5. Optimized Route: optimized_agent.py

**File:** `agents/optimized_agent.py`  
**Role:** Runs the claims review pipeline with ALL 5 optimization layers enabled.

### The 3-Phase Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  ML TRIAGE GATE: If decision ≠ NEEDS_REVIEW, return immediately  │
│  (0 tokens, 0 ms, no LLM calls)                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (only if NEEDS_REVIEW)
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1 — 3 Parallel Fetch Subagents (ThreadPoolExecutor)       │
│                                                                  │
│  ┌─ SubAgent 1: fetch_patient(patient_id)  ──► profile          │
│  │  • Retries up to MAX_RETRIES=3 on transient failure          │
│  │  • Returns (data, log, retry_count)                          │
│  │                                                              │
│  ├─ SubAgent 2: fetch_records(patient_id)  ──► records          │
│  │  • Same retry logic                                          │
│  │                                                              │
│  └─ SubAgent 3: fetch_history(patient_id)  ──► history          │
│     • Same retry logic                                          │
│                                                                  │
│  ⚡ ALL THREE RUN IN PARALLEL (not sequential)                   │
│  ⚡ Each has built-in retry (up to 3 attempts per resource)      │
│                                                                  │
│  Is LLM used? NO. These are pure database fetches via           │
│  data_fetcher.py. No API calls here.                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2 — 3 Parallel Analysis Subagents (ThreadPoolExecutor)    │
│                                                                  │
│  ┌─ SubAgent 1: process_all_pdfs(records, name, age)            │
│  │  → calls document_processor.process_all_pdfs()               │
│  │  → uses LLM (one call per document)                          │
│  │  → Layer 1: regex pre-extraction (0 tokens)                  │
│  │  → Layer 2: concise JSON prompts (fewer tokens)              │
│  │  → Layer 3: content-hash LRU cache (0 tokens on hit)         │
│  │  → Returns (pdf_summaries, raw_tokens, actual_tokens,        │
│  │             lru_cache_hits)                                   │
│  │                                                              │
│  ├─ SubAgent 2: analyze_claims_pattern(patient_id, history,     │
│  │                                       claim)                  │
│  │  → calls document_processor.analyze_claims_pattern()         │
│  │  → uses LLM (one call)                                       │
│  │  → Concise: pre-aggregates stats, sends only numbers         │
│  │  → Layer 3: content-hash cache (0 tokens on hit)             │
│  │  → Returns (pattern_dict, raw_tokens, actual_tokens,         │
│  │             was_cached)                                       │
│  │                                                              │
│  └─ SubAgent 3: fetch_guidelines_with_loop_detection(           │
│  │                    proc_key, fail_times, simulate_loop)       │
│  │  → calls data_fetcher.fetch_treatment_guidelines()           │
│  │  → Loop detection: if DB returns empty/rejects ≥ 3 times    │
│  │    in a row, breaks out (graceful degradation)               │
│  │  → Returns (guidelines, loop_detected, log)                  │
│  │                                                              │
│  │  Is LLM used? NO. This is a DB fetch.                        │
│  │                                                              │
│  ⚡ ALL THREE RUN IN PARALLEL                                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Semantic Cache → Compile Brief → Context Cache        │
│                                                                  │
│  Step 1: Build a semantic cache key from the claim signature     │
│          (patient name, age, procedure, doc count, history count,│
│           billed amount)                                          │
│                                                                  │
│  Step 2: Check Layer 4 (semantic cache)                         │
│          → TF-IDF cosine similarity ≥ 0.82 threshold             │
│          → If HIT:  0 tokens, reuse cached brief                 │
│          → If MISS: proceed to Step 3                            │
│                                                                  │
│  Step 3: Check Layer 5 (context prefix cache)                   │
│          → Patient context prefix (name, age, plan) is hashed   │
│          → First time: full token cost charged                   │
│          → Repeat time: only 10% of normal cost (90% discount)  │
│                                                                  │
│  Step 4: compile_analyst_brief(claim, profile, pdf_summaries,   │
│           claims_pattern, guidelines, risk_score, decision,      │
│           features, optimization_mode="concise")                  │
│          → calls document_processor.compile_analyst_brief()     │
│          → uses LLM (ONE call)                                   │
│          → Concise prompt: structured, ~150 words, key facts    │
│          → Returns (brief_text, raw_estimate, actual_tokens)     │
│                                                                  │
│  Step 5: Store result in semantic cache for future rounds        │
│          semantic_cache.store(sem_key, brief_text)               │
│                                                                  │
│  ⚡ Semantic cache hit = 0 tokens for compile_analyst_brief      │
│  ⚡ Context cache = 90% token savings on the patient prefix      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  FINAL: Calculate metrics via snapshot delta                     │
│                                                                  │
│  llm_api_calls       = new API calls since snapshot              │
│  total_input_tokens  = input tokens since snapshot               │
│  total_output_tokens = output tokens since snapshot              │
│  total_tokens        = input + output                            │
│  estimated_cost_usd  = cost since snapshot                       │
│  wall_time_ms        = total elapsed time                        │
└──────────────────────────────────────────────────────────────────┘
```

### Where the LLM is Called in the Optimized Route

| Phase | Function | What it does | LLM API Calls |
|---|---|---|---|
| Phase 2 | `process_all_pdfs()` → `summarize_medical_pdf()` | Summarizes each medical document | **1 per document** (or 0 if cache hit) |
| Phase 2 | `analyze_claims_pattern()` | Analyzes historical claims pattern | **1** (or 0 if cache hit) |
| Phase 3 | `compile_analyst_brief()` | Compiles the final analyst brief | **1** (or 0 if semantic cache hit) |

**Typical:** 4–5 LLM API calls total (for C001/C003/C004 with 3 documents each: 3 doc summaries + 1 claims pattern + 1 brief = 5 calls).

### Retry Logic (`_fetch_with_retry`)

```
For each fetch (patient, records, history):
    Try attempt 1:
        If success → return data (0 retries)
        If failure → log, go to attempt 2
    Try attempt 2:
        If success → return data, log "recovered after 1 retry"
        If failure → log, go to attempt 3
    Try attempt 3:
        If success → return data, log "recovered after 2 retries"
        If failure → log, go to final attempt
    Try attempt MAX_RETRIES+1 (4th attempt):
        If success → return data
        If failure → return None, log "max retries exhausted"
```

### Guidelines Loop Detection (`_sa_fetch_guidelines_with_loop_detection`)

```
For visit in 1, 2, 3 (MAX_GUIDELINES_VISITS):
    Try to fetch guidelines:
        If success AND data is non-empty:
            Return data (no loop detected)
        If failure OR empty data:
            Log "visit N/3 – empty response (loop simulation)"
            Continue to next visit
After 3 empty visits:
    Log "⛔ loop detected after 3 visits – compiling brief without guidelines"
    Return (None, loop_detected=True)
```

---

## 6. Unoptimized Route: unoptimized_agent.py

**File:** `agents/unoptimized_agent.py`  
**Role:** Runs the claims review pipeline with ZERO optimizations — the naive baseline.

### The Sequential Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  ML TRIAGE GATE: Same as optimized — skip if not NEEDS_REVIEW     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Fetch patient profile (ONE attempt, NO retry)           │
│                                                                  │
│  db.fetch_patient_profile(patient_id, attempt=1, fail_times=X)   │
│  → If fail_times ≥ 1, this call WILL FAIL                       │
│  → Logs "[patient] ❌ fetch failed – no retry (unoptimized)"     │
│  → profile = None (proceeds anyway with incomplete data)         │
│                                                                  │
│  Is LLM used? NO. DB fetch.                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: Fetch medical records (ONE attempt, NO retry)           │
│                                                                  │
│  db.fetch_medical_records(patient_id, attempt=1, fail_times=X)   │
│  → Same single-attempt pattern                                   │
│  → records = [] on failure                                       │
│                                                                  │
│  Is LLM used? NO. DB fetch.                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Fetch claims history (ONE attempt, NO retry)            │
│                                                                  │
│  db.fetch_claims_history(patient_id, attempt=1, fail_times=X)    │
│  → Same single-attempt pattern                                   │
│  → history = [] on failure                                       │
│                                                                  │
│  Is LLM used? NO. DB fetch.                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: Fetch treatment guidelines (ONE attempt, NO loop check) │
│                                                                  │
│  db.fetch_treatment_guidelines(proc_key, attempt=1, ...)         │
│  → Single attempt, no retry                                      │
│  → If simulate_loop=True → returns empty → proceeds anyway       │
│  → No loop detection at all                                      │
│                                                                  │
│  Is LLM used? NO. DB fetch.                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: Summarize each document with LLM                        │
│                                                                  │
│  For EACH document in records:                                   │
│    prompt = _VERBOSE_DOC.format(                                  │
│        name=name, age=age,                                        │
│        doc_type=doc.get("type"),                                  │
│        filename=doc.get("filename"),                              │
│        content=doc.get("content")   ← FULL RAW CONTENT!          │
│    )                                                              │
│    response, inp, out = _raw_llm(prompt, max_tokens=600)         │
│                                                                  │
│  ⚠️ NO pre-extraction (no regex stripping)                      │
│  ⚠️ VERBOSE prompt (open-ended, asks for 6 analysis sections)    │
│  ⚠️ FULL document content in prompt (thousands of chars each)    │
│  ⚠️ NO caching — fresh API call every time                       │
│  ⚠️ _raw_llm() bypasses ALL caches in llm_engine.py              │
│                                                                  │
│  Where LLM is used: YES — 1 API call per document                │
│  (typically 3–4 documents → 3–4 API calls)                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 6: Analyze claims pattern with LLM                         │
│                                                                  │
│  full_hist_text = all history rows as text                       │
│  prompt = _VERBOSE_CLAIMS.format(                                 │
│      patient_id=patient_id,                                       │
│      procedure=procedure,                                         │
│      amount=billed_amount,                                        │
│      total_claims=len(history),                                   │
│      full_history=full_hist_text  ← FULL HISTORY TEXT!           │
│  )                                                                │
│  response, inp, out = _raw_llm(prompt, max_tokens=400)           │
│                                                                  │
│  ⚠️ VERBOSE prompt (open-ended, full narrative request)         │
│  ⚠️ FULL claims history text sent to LLM                        │
│  ⚠️ NO pre-aggregation of stats                                 │
│  ⚠️ NO caching                                                  │
│                                                                  │
│  Where LLM is used: YES — 1 API call                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 7: Compile analyst brief with LLM                          │
│                                                                  │
│  all_doc_content = ""                                            │
│  For EACH document:                                               │
│      all_doc_content += FULL RAW CONTENT AGAIN                   │
│                                                                  │
│  prompt = _VERBOSE_BRIEF.format(                                  │
│      claim_id, procedure, billed, ...                             │
│      all_doc_content=all_doc_content,  ← ALL RAW CONTENT RE-SENT!│
│      full_history=full_hist_text,       ← FULL HISTORY RE-SENT!  │
│      guidelines=str(guidelines)         ← FULL GUIDELINES TEXT   │
│  )                                                                │
│  response, inp, out = _raw_llm(prompt, max_tokens=800)           │
│                                                                  │
│  ⚠️ This is the WORST offender:                                  │
│     - Re-sends ALL raw document content (already sent in Step 5) │
│     - Re-sends ALL claims history (already sent in Step 6)       │
│     - Verbose narrative prompt asks for 5 detailed sections      │
│     - Monolithic: everything in one giant prompt                 │
│  ⚠️ NO caching                                                   │
│                                                                  │
│  Where LLM is used: YES — 1 API call                             │
└──────────────────────────────────────────────────────────────────┘
```

### Where the LLM is Called in the Unoptimized Route

| Step | What it does | LLM API Calls |
|---|---|---|
| Step 5 | Summarize each document (verbose, full raw content) | **1 per document** |
| Step 6 | Analyze claims pattern (verbose, full history text) | **1** |
| Step 7 | Compile final analyst brief (ALL raw content re-sent) | **1** |

**Typical:** 5–6 LLM API calls (3 docs + 1 patterns + 1 brief = 5 calls). Same number of calls as optimized, but each call sends **3–6× more tokens** because:
1. Full raw content instead of pre-extracted sections
2. Verbose open-ended prompts instead of concise JSON prompts
3. Step 7 re-sends ALL document content AGAIN (unoptimized sends 2× the doc content)

### The `_raw_llm()` Function

```
_raw_llm(prompt, max_tokens):
    # Create a fresh OpenAI client
    # Call chat.completions.create() directly
    # Bypasses ALL caches (response cache in llm_engine.py)
    # Bypasses content-hash cache in document_processor.py
    # Bypasses semantic cache in semantic_cache.py
    # Bypasses context cache in context_cache.py
    # Returns (response_text, input_tokens, output_tokens)
```

Every call goes directly to the OpenAI API. No caching. No shortcuts.

---

## 7. Document Processor: document_processor.py

**File:** `tools/document_processor.py`  
**Role:** The workhorse for document summarization and claims analysis. Contains 3 optimization layers for the optimized route.

### Layer 1: Pre-extraction (ZERO tokens, pure Python regex)

```python
extract_relevant_sections(content: str, doc_type: str) -> str:
    # For radiology_report: extract only FINDINGS + IMPRESSION sections
    # For consultation_note: extract only ASSESSMENT + PLAN sections
    # For lab_report: extract RESULTS + INTERPRETATION sections
    # For therapy_note: extract GOALS/OUTCOMES + RECOMMENDATION sections
    # Falls back to first 900 + last 400 chars if no sections found
    # Truncates each section to 900 chars max
```

**Effect:** Reduces document content by 50–70% before it ever reaches the LLM.  
**Where LLM is used:** NOWHERE. Pure string processing.

### Layer 2: Concise vs Verbose Prompts

#### Concise (optimized route):
```
"{doc_type} – insurance review. Patient: {name}, {age}y.
Respond ONLY with valid JSON (no markdown):
{{
    "diagnosis": "...",
    "severity": "...",
    "necessity_evidence": [...],
    "prior_treatment": "...",
    "physician_recommendation": "..."
}}
Document: {relevant_content}"    ← pre-extracted, not full content
```

#### Verbose (used in unoptimized route):
```
"You are a senior healthcare insurance specialist reviewing medical documentation
to support a claims adjudication decision.
PATIENT INFORMATION: Name: {name} Age: {age}
Document Type: {doc_type}
Please provide a thorough and comprehensive analysis...
[6 detailed sections requested]
MEDICAL DOCUMENT CONTENT: {content}"    ← FULL raw content
```

**Effect:** Concise prompt is ~80% shorter than verbose prompt. Forces JSON output which is also shorter.

### Layer 3: Content-Hash LRU Cache

```python
cache_key = SHA-256(relevant_content + optimization_mode)[:20]

if cache_key in _SUMMARY_CACHE:
    return cached_summary, raw_tokens_estimate, 0, was_cached=True
    #                    ↑                     ↑
    #               still estimate          0 actual tokens used
    #               what it WOULD cost
```

**Effect:** If the same document (same extracted content, same mode) is seen again, the cached summary is returned with **zero LLM tokens**. This is why C006 (same patient as C001) saw 3 LRU cache hits — the patient's documents were already summarized during C001's round 1.

### Claims Pattern Analysis (also in document_processor.py)

#### Concise mode (optimized route):
```
"Analyze insurance claims pattern. Respond ONLY with valid JSON.
Stats (last 24 months):
  Total claims: 8
  Total paid: $45,000
  Same-procedure claims: 2
  Average claim amount: $5,625
Flagged claims:
  - C002: Some Procedure $12,000
JSON output: {...}"
```
→ Sends only **aggregated statistics** (numbers), not the full history

#### Verbose mode (unoptimized route):
```
"You are a senior healthcare insurance fraud analyst...
COMPLETE CLAIMS HISTORY (last 24 months):
  2024-01-15  C001  Some Procedure                    $5,000
  2024-03-22  C002  Another Procedure                 $12,000
  ... [every single claim row listed]
Please perform a comprehensive analysis..."
```
→ Sends **every single claim row** as text

### Analyst Brief Compilation (also in document_processor.py)

#### Concise mode (optimized route):
```
"Compile analyst review brief. Max 200 words.
CLAIM: C001 | Knee Arthroscopy | $15,500 (network: $12,000)
PATIENT: John Doe, 52y | Plan: PPO Gold | In-network: Yes
ML RISK: 0.47 (NEEDS_REVIEW) | Top factors: ...
MEDICAL FINDINGS (from 3 documents):
  [Radiology Report] Dx: Meniscal tear | Severity: medium | Rec: ...
  [Consultation Note] Dx: Osteoarthritis | Severity: high | Rec: ...
CLAIMS HISTORY: medium risk | 8 prior claims | $45,000 total
  Pattern: Normal utilization pattern
GUIDELINES: 4 criteria defined
  Avg procedure cost: $12,500 | Billed is +24% vs network avg
Output brief with: KEY FINDINGS, RISK ASSESSMENT, RECOMMENDATION"
```
→ Sends **summaries, not raw content**

#### Verbose mode (unoptimized route):
```
"You are a senior healthcare insurance claims analyst...
CLAIM DETAILS: [all claim fields]
PATIENT INFORMATION: [all patient fields]
ML RISK ASSESSMENT: [all risk fields]
MEDICAL DOCUMENTATION REVIEW (3 documents):
  [FULL RAW DOCUMENT 1 CONTENT]
  [FULL RAW DOCUMENT 2 CONTENT]
  [FULL RAW DOCUMENT 3 CONTENT]
CLAIMS HISTORY ANALYSIS: [every claim row]
TREATMENT GUIDELINE COMPLIANCE: [full guidelines text]
Please compile a comprehensive analyst brief that:
[5 detailed sections requested]"
```
→ Re-sends **everything** — all raw document content, all history, all guidelines

---

## 8. LLM Engine: llm_engine.py

**File:** `core/llm_engine.py`  
**Role:** Centralized OpenAI API wrapper with a global token tracker and response cache.

### The `llm_call()` Function (used by optimized route)

```python
llm_call(prompt, step, model="gpt-4o-mini", temperature=0.0, 
         max_tokens=600, use_cache=True):
    
    # Step 1: Hash the prompt
    prompt_hash = SHA-256(prompt + model + temperature)[:16]
    
    # Step 2: Check response cache
    if use_cache and prompt_hash in _response_cache:
        # Return cached response (0 API calls, 0 cost)
        return _response_cache[prompt_hash], cached_record
    
    # Step 3: Call OpenAI API
    client = openai.OpenAI(api_key=...)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    response = resp.choices[0].message.content
    
    # Step 4: Cache the response
    if use_cache:
        _response_cache[prompt_hash] = response
    
    # Step 5: Record in global tracker
    tracker.record(step, prompt, response, cached=False)
    
    return response, token_record
```

### The `_raw_llm()` Function (used ONLY by the unoptimized route)

```python
_raw_llm(prompt, max_tokens=800):
    # Directly calls OpenAI API
    # BYPASSES _response_cache completely
    # BYPASSES tracker recording
    # Returns raw (response_text, input_tokens, output_tokens)
```

### Global Token Tracker

All LLM calls from the optimized route go through the global `tracker` instance. When the optimized route starts, it takes a `snapshot` of the tracker. After all calls, it computes the `delta` — this gives us the exact token usage for just that route's execution.

The unoptimized route does NOT use the tracker — it tracks its own tokens manually in local variables `total_inp` and `total_out`.

### Token Counting

Tokens are counted using `tiktoken` (OpenAI's tokenizer library), which is a **local, offline** operation:

```python
_enc = tiktoken.encoding_for_model("gpt-4o-mini")
token_count = len(_enc.encode(text))
```

This is why input token counts are accurate even without a valid API key — the counting happens locally.

### Pricing

| Token Type | Price per 1K tokens |
|---|---|
| Input (prompt) | $0.00015 |
| Output (completion) | $0.00060 |

These are the pricing rates for `gpt-4o-mini`.

---

## 9. Semantic Cache: semantic_cache.py

**File:** `core/semantic_cache.py`  
**Role:** Layer 4 of optimizations. Caches LLM responses based on **meaning**, not exact text match.

### How It Works

1. **Store:** When a brief is compiled, both the prompt text and the brief response are stored. The prompt is converted to a **bag-of-words** (lowercase alpha tokens ≥ 2 chars).

2. **Lookup:** When a new brief is needed, the new prompt is also converted to a bag-of-words. **TF-IDF cosine similarity** is computed against all stored entries.

3. **Threshold:** If similarity ≥ **0.82**, the cached brief is returned (0 tokens). If < 0.82, it's a miss and the LLM is called.

### Why This Matters for Healthcare Claims

- Same patient (P001) submits C001 then C006 → prompt is ~90% similar text → semantic hit
- Round 2 of stress test changes `risk_score` by a few decimal places → rest of prompt is identical → semantic hit

### Example

```
Stored prompt: "brief|John Doe|52|Knee Arthroscopy|3docs|8hist|15500"
New prompt:    "brief|John Doe|52|Post Op Visit|3docs|8hist|2400"
Similarity:    ~85% (same patient, same doc count, same history count)
Result:        CACHE HIT — 0 tokens for this LLM call
```

### In the Demo Output

The semantic cache had **0 hits / 4 entries (0.0%)** in this run. This is because:
- C006's brief prompt was different enough from C001's (different procedure, different billed amount)
- Round 2 for all claims was AUTO_APPROVE (no brief compiled, so nothing to cache or look up)
- The 4 entries were stored from the 4 round-1 NEEDS_REVIEW cases, but none were similar enough to reuse

---

## 10. Context Cache: context_cache.py

**File:** `core/context_cache.py`  
**Role:** Layer 5 of optimizations. Simulates **provider-level prompt prefix caching** (Anthropic's model: cached prefix tokens cost 10% of normal).

### How It Works

1. When `compile_analyst_brief()` is called, the patient context prefix is extracted:
   ```
   "Patient: John Doe, age 52, plan: PPO Gold"
   ```

2. This prefix is hashed (SHA-256) and checked against previously seen prefixes.

3. **First time:** Full token cost is charged. The prefix is stored.

4. **Repeat time:** Only **10%** of the token cost is charged. The other 90% is tracked as "tokens saved."

### In the Demo Output

From C006 Round 1:
```
Context tok. saved: 11
```

This means the patient context prefix "Patient: John Doe, age 52, plan: PPO Gold" was seen during C001 (same patient), and when C006 compiled its brief, the prefix was already in the context cache → 90% savings on that prefix chunk. Since the prefix was ~12 tokens:
- Full cost would be 12 tokens
- Cached cost is 12 × 0.10 = ~1 token
- Tokens saved = 12 - 1 = 11

### The Anthropic/OpenAI Pricing Model

In production APIs:
- **Anthropic context caching:** Cached prompt prefixes cost ~10% of normal input price
- **OpenAI prompt caching:** Automatic for prompts > 1024 tokens, ~50% discount
- **Google Gemini:** Explicit context caching with TTL

We simulate the Anthropic model (90% savings) here.

---

## 11. Data Fetcher: data_fetcher.py

**File:** `tools/data_fetcher.py`  
**Role:** Simulates database calls for patient data, medical records, claims history, and treatment guidelines. All data is synthetic/mock.

### What It Returns

| Function | Returns | Data Source |
|---|---|---|
| `fetch_patient_profile(patient_id)` | `{name, age, plan, ...}` | Synthetic mock |
| `fetch_medical_records(patient_id)` | `[{doc_id, type, filename, content, ...}]` | Synthetic mock |
| `fetch_claims_history(patient_id)` | `[{claim_id, date, procedure, amount, ...}]` | Synthetic mock |
| `fetch_treatment_guidelines(proc_key)` | `{approval_criteria, avg_cost_network, ...}` | Synthetic mock |

### Failure Simulation

The data fetcher supports failure injection through the `fail_times` parameter:
- `fail_times=2` → the first 2 calls fail (simulating transient DB outage), the 3rd succeeds
- `simulate_loop=True` → always returns empty data (simulating a stuck guidelines node)

**Where LLM is used:** NOWHERE. Pure data retrieval (simulated locally).

---

## 12. End-to-End Flow for a Single Claim

Let's trace through **C001 (Knee Arthroscopy, Round 1)** end-to-end to see exactly what happens:

### Input Data

```
claim_id:    C001
patient_id:  P001 (John Doe)
procedure:   Knee Arthroscopy with Meniscectomy
billed:      $15,500
risk_score:  0.4724 (random, from random.random())
decision:    NEEDS_REVIEW (0.4724 is in the 0.3–0.7 band)
```

### Optimized Route (1,837 tokens, 11,006 ms, 4 LLM calls)

```
Timeline:
│
├─ [0ms]      ML Triage: 0.4724 → NEEDS_REVIEW → continue pipeline
│
├─ [0ms]      PHASE 1 START (3 threads in parallel)
│   Thread 1: fetch_patient("P001")     → profile = {name:"John Doe", age:52, plan:"PPO Gold"}
│   Thread 2: fetch_records("P001")     → records = [radiology_report, consultation_note, 
│                                                        lab_report]  (3 documents)
│   Thread 3: fetch_history("P001")     → history = [C001_prior, C002_prior, C003_prior, ...] 
│                                        (8 prior claims)
│   ⏱ ALL finish in ~700ms (parallel, not sequential)
│   ✅ 0 retries (all succeeded on first attempt)
│
├─ [~700ms]    PHASE 2 START (3 threads in parallel)
│   Thread 1: process_all_pdfs(records, "John Doe", 52)
│     │
│     ├─ Doc 1: radiology_report "knee_mri_2024.pdf"
│     │   Layer 1: extract_relevant_sections() → FINDINGS + IMPRESSION only
│     │            (reduces 3000 chars → ~1100 chars, ~63% reduction)
│     │   Layer 3: cache check → MISS (first time seeing this doc)
│     │   Layer 2: build CONCISE prompt with pre-extracted text
│     │   LLM CALL #1: llm_call(prompt, step="summarize_pdf:knee_mri_2024.pdf")
│     │   → Returns JSON: {diagnosis:"Meniscal tear", severity:"medium", ...}
│     │   → Input tokens: ~350, Output tokens: ~120
│     │   ✅ Cache stored
│     │
│     ├─ Doc 2: consultation_note "ortho_consult_2024.pdf"
│     │   Layer 1: extract → ASSESSMENT + PLAN only (~800 chars)
│     │   Layer 3: cache check → MISS
│     │   LLM CALL #2: llm_call(prompt, step="summarize_pdf:ortho_consult_2024.pdf")
│     │   → Returns JSON: {diagnosis:"Osteoarthritis grade III", severity:"high", ...}
│     │   → Input tokens: ~280, Output tokens: ~130
│     │
│     ├─ Doc 3: lab_report "preop_labs_2024.pdf"
│     │   Layer 1: extract → RESULTS + INTERPRETATION (~900 chars)
│     │   Layer 3: cache check → MISS
│     │   LLM CALL #3: llm_call(prompt, step="summarize_pdf:preop_labs_2024.pdf")
│     │   → Returns JSON: {diagnosis:"Normal lab values", severity:"low", ...}
│     │   → Input tokens: ~260, Output tokens: ~100
│     │
│     └─ Returns: (pdf_summaries, raw_tokens=~7200, actual_tokens=~1240, lru_hits=0)
│
│   Thread 2: analyze_claims_pattern("P001", history, claim)
│     │   Layer 3: cache check → MISS (first time for this patient+amount combo)
│     │   Pre-aggregate stats (no LLM): total=8, paid=$45K, avg=$5.6K, similar=2
│     │   Layer 2: build CONCISE prompt with only stats, not full history
│     │   LLM CALL #4: llm_call(prompt, step="analyze_claims:P001")
│     │   → Returns JSON: {pattern:"normal", risk_level:"low", ...}
│     │   → Input tokens: ~200, Output tokens: ~100
│     │   ✅ Cache stored
│
│   Thread 3: fetch_guidelines_with_loop_detection("knee_arthroscopy")
│     │   Visit 1/3: fetch_treatment_guidelines() → SUCCESS, data present
│     │   → Returns ({approval_criteria: [...], avg_cost: $12,500}, loop=False, log=[])
│     │   ⚡ No loop detected
│     │
│     ⏱ ALL finish in ~5000ms (parallel, but LLM calls dominate)
│
├─ [~5700ms]   PHASE 3 START
│   Step 1: Build semantic key
│     "brief|John Doe|52|Knee Arthroscopy with Meniscectomy|3docs|8hist|15500"
│   Step 2: Semantic cache lookup → MISS (first time)
│   Step 3: Context cache 
│     prefix = "Patient: John Doe, age 52, plan: PPO Gold" (12 tokens)
│     Hash check → FIRST TIME (never seen before)
│     → Full 12 tokens charged
│   Step 4: compile_analyst_brief(...)
│     Layer 2: build CONCISE brief prompt (summaries only, not raw content)
│     LLM CALL #5: llm_call(prompt, step="compile_brief:C001")
│     → Returns: brief text (~200 words)
│     → Input tokens: ~300, Output tokens: ~150
│   Step 5: Store in semantic cache
│     semantic_cache.store(sem_key, brief_text)
│
└─ [~11006ms]  DONE
    Total LLM API calls: 5
    Total input tokens: ~1,311
    Total output tokens: ~526
    Total tokens: 1,837
    LRU cache hits: 0
    Semantic cache hits: 0
    Context tokens saved: 0 (first time seeing this patient)
```

### Unoptimized Route (6,925 tokens, 34,908 ms, 5 LLM calls)

```
Timeline:
│
├─ [0ms]      ML Triage: 0.4724 → NEEDS_REVIEW → continue pipeline
│
├─ [0ms]      Step 1: fetch_patient("P001") → profile
│   ⏱ ~500ms (sequential, blocks everything else)
│
├─ [~500ms]    Step 2: fetch_records("P001") → records (3 docs)
│   ⏱ ~500ms
│
├─ [~1000ms]   Step 3: fetch_history("P001") → history (8 claims)
│   ⏱ ~500ms
│
├─ [~1500ms]   Step 4: fetch_guidelines("knee_arthroscopy") → guidelines
│   ⏱ ~500ms
│
├─ [~2000ms]   Step 5: Summarize EACH document (SEQUENTIAL)
│   │
│   ├─ Doc 1: radiology_report "knee_mri_2024.pdf"
│   │   prompt = _VERBOSE_DOC (full raw 3000 chars, open-ended, 6 sections)
│   │   LLM CALL #1: _raw_llm(prompt, max_tokens=600)
│   │   → Input tokens: ~1,200, Output tokens: ~500
│   │   ⏱ ~8000ms (LLM processing large prompt)
│   │
│   ├─ Doc 2: consultation_note "ortho_consult_2024.pdf"
│   │   LLM CALL #2: _raw_llm(prompt, max_tokens=600)
│   │   → Input tokens: ~1,000, Output tokens: ~500
│   │   ⏱ ~7000ms
│   │
│   ├─ Doc 3: lab_report "preop_labs_2024.pdf"
│   │   LLM CALL #3: _raw_llm(prompt, max_tokens=600)
│   │   → Input tokens: ~900, Output tokens: ~400
│   │   ⏱ ~6000ms
│
├─ [~23000ms]  Step 6: Analyze claims pattern
│   full_hist_text = "2024-01-15 C001 ...\n2024-03-22 C002 ...\n..." (all 8 rows)
│   prompt = _VERBOSE_CLAIMS (full history text, open-ended)
│   LLM CALL #4: _raw_llm(prompt, max_tokens=400)
│   → Input tokens: ~800, Output tokens: ~300
│   ⏱ ~5000ms
│
├─ [~28000ms]  Step 7: Compile analyst brief
│   all_doc_content = FULL DOC 1 + FULL DOC 2 + FULL DOC 3 (re-sent!)
│   full_hist_text  = ALL 8 CLAIM ROWS (re-sent!)
│   prompt = _VERBOSE_BRIEF (ALL raw content, verbose, 5 sections)
│   LLM CALL #5: _raw_llm(prompt, max_tokens=800)
│   → Input tokens: ~1,450, Output tokens: ~700
│   ⏱ ~7000ms
│
└─ [~34908ms]  DONE
    Total LLM API calls: 5
    Total input tokens: ~4,525
    Total output tokens: ~2,400
    Total tokens: 6,925
```

### Comparison: Why 73.5% Token Savings?

| Factor | Optimized | Unoptimized | Reason |
|---|---|---|---|
| Doc 1 content size | ~1,100 chars (extracted) | ~3,000 chars (full) | Layer 1: Pre-extraction |
| Doc 1 prompt style | Concise JSON | Verbose narrative | Layer 2: Concise prompts |
| Doc 2 prompt size | ~800 chars | ~2,500 chars | Same as above |
| Doc 3 prompt size | ~900 chars | ~2,200 chars | Same as above |
| Claims pattern | ~200 tokens (stats only) | ~800 tokens (full history) | Layer 2: Aggregated stats |
| Brief doc content | Summaries (~500 chars) | Full raw (~7,500 chars) | Avoids re-sending |
| Brief history | Pattern summary (~70 chars) | Full history (~400 chars) | Avoids re-sending |
| Brief prompt style | 200-word limit | Open-ended 5 sections | Layer 2 |

---

## 13. The 5 Optimization Layers

### Layer 1: Content Pre-extraction
- **What:** Regex-based extraction of clinically relevant sections (FINDINGS, IMPRESSION, ASSESSMENT, PLAN)
- **Where:** `document_processor.py → extract_relevant_sections()`
- **Tokens used:** 0 (pure Python string processing)
- **Reduction:** 50–70% of raw document size
- **Example:** 3,000-char radiology report → 1,100-char extracted section

### Layer 2: Concise Structured Prompts
- **What:** JSON-output prompts asking for specific fields instead of open-ended prose
- **Where:** `document_processor.py → _CONCISE_PROMPT, _CLAIMS_PATTERN_CONCISE, _BRIEF_CONCISE`
- **Tokens used:** ~60% fewer than verbose equivalents
- **Additional benefit:** JSON output is also shorter than narrative prose

### Layer 3: LRU Content-Hash Cache
- **What:** SHA-256 hash of (content + mode) → skip LLM for identical inputs
- **Where:** `document_processor.py → _SUMMARY_CACHE, _CLAIMS_HISTORY_CACHE`
- **Tokens used:** 0 on cache hit
- **Best case:** C006 (same patient P001) → 3 LRU hits, only 1 new LLM call
- **Additional:** `llm_engine.py` also has a prompt-level response cache

### Layer 4: Semantic Similarity Cache
- **What:** TF-IDF cosine similarity (threshold 0.82) → reuse brief for similar claims
- **Where:** `core/semantic_cache.py`
- **Tokens used:** 0 on cache hit (saves the entire `compile_analyst_brief` call)
- **Works by:** Matching bag-of-words similarity, not exact text

### Layer 5: Context Prefix Cache
- **What:** Simulates Anthropic/OpenAI prompt prefix caching (90% discount on repeated prefix)
- **Where:** `core/context_cache.py`
- **Savings:** ~90% on repeated patient context prefixes
- **Works by:** Hashing patient context prefix; first use = full cost, repeat use = 10% cost

---

## 14. Part B: Failure Scenarios

Three failure scenarios test the resilience differences between the optimized and unoptimized routes.

### F1: Transient DB Failure (Patient Fetch Fails 2 Times)

**What's injected:** `scenario_config = {"fail_patient_fetch_times": 2}`

**Optimized route behavior:**
```
Attempt 1: fetch_patient("P001") → FAIL (fail_times=2, attempt 1 fails)
Attempt 2: fetch_patient("P001") → FAIL (fail_times=2, attempt 2 fails)
Attempt 3: fetch_patient("P001") → SUCCESS (fail_times=2, attempt 3 succeeds)
→ Logs: "[patient] ✅ recovered after 2 retry(ies)"
→ Brief contains: "Patient: John Doe, age 52, plan: PPO Gold"
→ 488 tokens (complete brief with patient data)
```

**Unoptimized route behavior:**
```
Attempt 1: fetch_patient("P001") → FAIL (single attempt, no retry)
→ Logs: "[patient] ❌ fetch failed – no retry (unoptimized)"
→ Brief contains: "Patient: Unknown" (no patient data!)
→ 6,909 tokens (even though data is missing, verbose prompts use more tokens!)
```

**Key difference:** The optimized route retries and recovers. The unoptimized route fails silently and produces an incomplete brief.

### F2: Guidelines Loop (DB Always Returns Empty)

**What's injected:** `scenario_config = {"simulate_guidelines_loop": True}`

**Optimized route behavior:**
```
Visit 1/3: fetch_guidelines() → empty (simulated loop)
Visit 2/3: fetch_guidelines() → empty (simulated loop)
Visit 3/3: fetch_guidelines() → empty (simulated loop)
→ ⛔ LOOP DETECTED: breaks out after 3 empty responses
→ Logs: "[guidelines] ⛔ loop detected after 3 visits – compiling brief without guidelines"
→ Brief compiled WITHOUT guidelines section (graceful degradation)
→ 485 tokens, loop_detected=True
```

**Unoptimized route behavior:**
```
Attempt 1: fetch_guidelines() → empty
→ Logs: "[guidelines] ❌ empty/failed – no loop detection (unoptimized)"
→ Brief compiled silently without guidelines
→ 6,769 tokens, loop_detected=False (NO detection at all!)
```

**Key difference:** The optimized route detects the loop after 3 visits and degrades gracefully with logging. The unoptimized route has NO loop detection — in a real system, this would mean an **infinite loop** or silently incomplete data.

### F3: Multi-Resource Failure (Records + History Both Fail Once)

**What's injected:** `scenario_config = {"fail_records_fetch_times": 1, "fail_history_fetch_times": 1}`

**Optimized route behavior:**
```
records fetch:
  Attempt 1: FAIL
  Attempt 2: SUCCESS → "[records] ✅ recovered after 1 retry(ies)"
history fetch:
  Attempt 1: FAIL
  Attempt 2: SUCCESS → "[history] ✅ recovered after 1 retry(ies)"
→ retry_count = 2 (one per resource)
→ Brief contains: pdf_summaries (3 docs processed) + claims_pattern analysis
→ 566 tokens (slightly higher because it actually processed documents)
```

**Unoptimized route behavior:**
```
records fetch: Attempt 1: FAIL → records = []
history fetch: Attempt 1: FAIL → history = []
→ retry_count = 0
→ Brief has: "No medical documents available" + "No prior claims"
→ 1,916 tokens (FEWER tokens, but that's BAD — it means data was SKIPPED)
```

**Key difference:** The optimized route has MORE tokens here, which is actually CORRECT — it means the documents were processed. The unoptimized route has FEWER tokens because it skipped all document processing when the fetch failed.

---

## Summary: All LLM Call Locations

| # | Route | Phase | Function | File | Tokens (typical) |
|---|---|---|---|---|---|
| 1 | Optimized | Phase 2 | `summarize_medical_pdf()` doc 1 | `document_processor.py` | ~350 in, ~120 out |
| 2 | Optimized | Phase 2 | `summarize_medical_pdf()` doc 2 | `document_processor.py` | ~280 in, ~130 out |
| 3 | Optimized | Phase 2 | `summarize_medical_pdf()` doc 3 | `document_processor.py` | ~260 in, ~100 out |
| 4 | Optimized | Phase 2 | `analyze_claims_pattern()` | `document_processor.py` | ~200 in, ~100 out |
| 5 | Optimized | Phase 3 | `compile_analyst_brief()` | `document_processor.py` | ~300 in, ~150 out |
| --- | --- | --- | --- | --- | --- |
| 6 | Unoptimized | Step 5 | `_raw_llm()` doc 1 (verbose) | `unoptimized_agent.py` | ~1,200 in, ~500 out |
| 7 | Unoptimized | Step 5 | `_raw_llm()` doc 2 (verbose) | `unoptimized_agent.py` | ~1,000 in, ~500 out |
| 8 | Unoptimized | Step 5 | `_raw_llm()` doc 3 (verbose) | `unoptimized_agent.py` | ~900 in, ~400 out |
| 9 | Unoptimized | Step 6 | `_raw_llm()` claims (verbose) | `unoptimized_agent.py` | ~800 in, ~300 out |
| 10 | Unoptimized | Step 7 | `_raw_llm()` brief (ALL content re-sent) | `unoptimized_agent.py` | ~1,450 in, ~700 out |

**Both routes make the same number of LLM calls (5 calls for 3 documents). The savings come from WHAT is sent to the LLM, not from reducing the number of calls.**

---

## One-Line Summary of Each File's Role

| File | Role |
|---|---|
| `demo.py` | Entry point — prints header, calls `run_stress_test()` |
| `scenarios/stress_test.py` | Orchestrator — runs all claims through both routes, prints comparisons, writes `RESULTS.md` |
| `ml_models/risk_model.py` | XGBoost simulator — returns random risk scores in stress-test mode, determines AUTO_APPROVE/NEEDS_REVIEW/AUTO_DENY |
| `agents/optimized_agent.py` | Optimized pipeline — 3-phase parallel execution, 5 optimization layers, retry + loop detection |
| `agents/unoptimized_agent.py` | Naive baseline — sequential execution, verbose prompts, full raw content, no caching, no retry |
| `tools/document_processor.py` | LLM summarization — Layers 1 (regex), 2 (concise prompts), 3 (content-hash cache) for docs, claims, and briefs |
| `tools/data_fetcher.py` | Mock database — returns synthetic patient records, history, and guidelines (no LLM) |
| `core/llm_engine.py` | OpenAI API wrapper — global token tracker, response cache, `llm_call()` for optimized route |
| `core/semantic_cache.py` | Layer 4 — TF-IDF cosine similarity cache for brief compilation |
| `core/context_cache.py` | Layer 5 — Simulated prompt prefix caching (90% discount on repeat prefixes) |
