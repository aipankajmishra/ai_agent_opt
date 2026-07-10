# Healthcare Claims — HITL Review Agent with Token Optimisation

A side-by-side stress test of **optimized vs unoptimized** LLM pipelines for healthcare claims review. Processes 30 claims across 12 patients, measuring token savings, cost reductions, and resilience to failures.

---

## The Problem

Healthcare claims adjudication requires a human analyst to review clinical documents, claims history, and treatment guidelines before approving or denying expensive procedures. Without automation, each claim takes 30–45 minutes of manual reading.

**With this agent:** each NEEDS_REVIEW claim is processed in ~10 seconds for **$0.0004** (optimized route), vs ~35 seconds for **$0.0019** (unoptimized route). That's **79% token savings and 83% cost savings** — just from prompt engineering and caching, with the same LLM model.

---

## Architecture

```
30 CLAIMS (C001–C030)
    │
    ▼
ML Risk Model (XGBoost Simulator)
    │
    ├── score < 0.3  →  AUTO_APPROVE   (0 tokens, 0 ms)
    ├── score > 0.7  →  AUTO_DENY      (0 tokens, 0 ms)
    └── 0.3 ≤ score ≤ 0.7  →  NEEDS_REVIEW  →  Full HITL Pipeline
                                                      │
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
                     OPTIMIZED ROUTE                                  UNOPTIMIZED ROUTE
                     (5 layers, parallel)                             (0 layers, sequential)
                              │                                               │
              Phase 1: 3 parallel fetches (retry)            Step 1-4: sequential fetches (no retry)
              Phase 2: 3 parallel analyses (LRU cache)       Step 5: LLM per doc (verbose, full content)
              Phase 3: semantic cache → brief → context      Step 6: LLM claims pattern (full history)
                              │                               Step 7: LLM brief (re-sends ALL content)
                              ▼                                               ▼
                              └─────────────────── COMPARE ───────────────────┘
                                                     │
                                          Token savings, cost, time, cache hits
```

---

## The 5 Optimisation Layers

| Layer | Technique | Where Applied | Token Saving |
|-------|-----------|--------------|-------------|
| **1** | Regex pre-extraction | Strip documents to FINDINGS/IMPRESSION/ASSESSMENT/PLAN | 50–70% |
| **2** | Concise JSON prompts | Structured prompts, specific fields only | ~60% |
| **3** | LRU content-hash cache | SHA-256 of (extracted content + mode) → skip LLM | 100% on hit |
| **4** | Semantic similarity cache | TF-IDF cosine ≥ 0.82 → reuse cached analyst brief | 100% on hit |
| **5** | Context prefix cache | Patient prefix (name, age, plan) charged at 10% on repeat use | ~90% on repeat |

**Additional resilience (optimized route only):**
- Retry on transient DB failures (up to 3 attempts per resource)
- Guidelines loop detection (break after 3 empty responses — graceful degradation)

---

## Stress Test Results (Actual Run)

```
30 claims processed sequentially (single pass, random.random() scores)

Decision breakdown:
  NEEDS_REVIEW:  15 claims (full LLM pipeline)
  AUTO_APPROVE:  12 claims (ML gate — 0 tokens)
  AUTO_DENY:      3 claims (ML gate — 0 tokens)

Token saving:  78.9%  (13,840 optimized vs 65,625 unoptimized)
Cost saving:   82.5%  ($0.00422 vs $0.02412)

Cache effectiveness:
  Semantic cache:     1 hit / 14 entries stored  (6.7%)
  Context cache:      5 reads / 58 tokens saved
  Unique prefixes:    9
```

### Cache in Action — Same-patient claims

| Claim | Patient | LRU Hits | Semantic Hit | Tokens (opt vs unopt) | Savings |
|-------|---------|----------|-------------|----------------------|---------|
| C001 | John Doe | 0 | 0 | 1,573 vs 5,379 | 70.8% |
| C007 | John Doe | **3** | 0 | 503 vs 5,323 | **90.6%** |
| C003 | Michael Chen | 0 | 0 | 1,433 vs 5,022 | 71.5% |
| C008 | Michael Chen | **3** | 0 | 483 vs 4,968 | **90.3%** |
| C009 | Michael Chen | **3** | **1** | **0 vs 5,022** | **💯 100%** |

**C009 (Michael Chen, Chemo Cycle 4) achieved 100% token savings** — all documents were LRU-cached from C003/C008, and the analyst brief was a semantic cache hit from C008's nearly-identical brief. The unoptimized route burned 5,022 tokens for the same result.

---

## Failure Scenarios (Part B)

Three injected-failure tests demonstrate resilience differences:

| Scenario | Optimized | Unoptimized |
|----------|-----------|-------------|
| **F1: DB Failure** | Retries 3×, recovers, brief complete | Fails silently, brief missing patient data |
| **F2: Guidelines Loop** | Detects after 3 visits, degrades gracefully | No detection — would loop forever in production |
| **F3: Multi-Resource** | Retries both resources, recovers | Proceeds with empty records and empty history |

---

## Setup

```bash
git clone https://github.com/aipankajmishra/ai_agent_opt.git
cd ai_agent_opt
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
cp .env.example .env         # add your OPENAI_API_KEY
python demo.py
```

**Without a valid OpenAI API key:**
- Input token counts are still **accurate** (tiktoken measures prompt size locally)
- Output tokens will reflect only error strings, not real LLM completions
- Token-saving **ratios** between routes remain meaningful (driven by prompt size differences)

---

## Project Structure

```
├── demo.py                         Entry point — prints header, runs stress test
├── requirements.txt                pip dependencies (openai, tiktoken, dotenv)
├── FLOW.md                         Mermaid flow diagrams (architecture + pipelines)
├── details.md                      Detailed walkthrough of every step
├── RESULTS.md                      Latest stress-test numbers (auto-generated)
│
├── data/
│   ├── claims_data.py              30 claims across 12 patients (C001–C030)
│   └── medical_records.py          Patient profiles, medical documents, claims history,
│                                   treatment guidelines (27 procedures), procedure map
│
├── ml_models/
│   └── risk_model.py               XGBoost simulator — random scores in stress-test mode,
│                                   determines AUTO_APPROVE / NEEDS_REVIEW / AUTO_DENY
│
├── agents/
│   ├── optimized_agent.py          Optimized pipeline — 3-phase parallel execution,
│   │                               5 optimization layers, retry + loop detection
│   └── unoptimized_agent.py        Naive baseline — sequential, verbose prompts,
│                                   full raw content, zero caching, no retry
│
├── tools/
│   ├── data_fetcher.py             Mock database — returns synthetic patient records,
│   │                               history, and guidelines (supports fail_times injection)
│   └── document_processor.py       Layers 1–3: regex pre-extraction, concise prompts,
│                                   content-hash LRU cache, brief compilation
│
├── core/
│   ├── llm_engine.py               OpenAI API wrapper — global token tracker,
│   │                               snapshot/delta measurement, response cache
│   ├── semantic_cache.py           Layer 4 — TF-IDF cosine similarity cache (threshold 0.82)
│   └── context_cache.py            Layer 5 — prompt prefix caching (90% discount on repeat)
│
└── scenarios/
    └── stress_test.py              Orchestrator — 30-claim sequential pass,
                                    Part B failure scenarios, writes RESULTS.md
```

---

## How It Works

### Optimized Route

1. **ML Triage Gate:** If the XGBoost score is outside the 0.3–0.7 range, the claim is auto-approved or auto-denied (0 LLM tokens).
2. **Phase 1 — Parallel Fetches:** Patient profile, medical records, and claims history are fetched concurrently via `ThreadPoolExecutor`. Each fetch retries up to 3 times on transient failure.
3. **Phase 2 — Parallel Analysis:** Three subagents run concurrently:
   - Documents are regex-pre-extracted (Layer 1), formatted as concise JSON prompts (Layer 2), and checked against a content-hash LRU cache (Layer 3) before any LLM call
   - Claims history is pre-aggregated into statistics (no LLM), then checked against cache
   - Treatment guidelines are fetched with loop detection (max 3 empty-response visits)
4. **Phase 3 — Brief Compilation:** A semantic cache key is built from the claim signature. If a semantically similar brief exists (Layer 4), it's returned with 0 tokens. Otherwise, the patient context prefix is checked for a 90% discount (Layer 5), and a concise brief is compiled via LLM. The result is stored in the semantic cache for future claims.

### Unoptimized Route

1. **Step 1–4:** Data is fetched sequentially (one attempt, no retry). Failures proceed silently with `None`/empty data.
2. **Step 5:** Each document is sent to the LLM with **full raw content** and **verbose open-ended prompts** (no extraction, no concise formatting).
3. **Step 6:** Claims history is sent to the LLM as **full text of every row** (no pre-aggregation).
4. **Step 7:** The analyst brief is compiled with **all raw document content re-sent** and **all claims history re-sent** in a single monolithic verbose prompt.
5. All LLM calls bypass all caching layers via `_raw_llm()`.

### The savings come from **what** is sent to the LLM, not from making fewer calls. Both routes make roughly the same number of API calls — the optimized route just sends far fewer tokens per call.

---

## Key Files

| File | Description |
|------|-------------|
| [FLOW.md](FLOW.md) | Mermaid diagrams — architecture, pipelines, cache flow, data layers |
| [details.md](details.md) | Complete step-by-step walkthrough of every function and LLM call |
| [RESULTS.md](RESULTS.md) | Auto-generated after each run — full 30-claim comparison table |