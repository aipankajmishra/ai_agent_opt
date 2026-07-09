# Results — Healthcare Claims HITL Agent Token Optimisation
### Real numbers from `python demo.py` — July 9, 2026

---

## How costs are counted

Token costs are measured by [`tiktoken`](https://github.com/openai/tiktoken) against every actual OpenAI API call.
Prices: **$0.15 / 1M input tokens, $0.60 / 1M output tokens** (gpt-4o-mini).

"Verbose" = open-ended prose prompts sent with the full raw document.  
"Concise" = structured JSON prompts sent with pre-extracted relevant sections only.

---

## Optimisation 1 — PDF pre-extraction (zero LLM, pure text)

Extract only clinically relevant sections (FINDINGS, IMPRESSION, ASSESSMENT, PLAN)
before the prompt is built. This happens in Python — no API call, no tokens.

| Scenario | Claim | Documents | Raw tokens est. | Actual tokens sent | Saved |
|----------|-------|-----------|-----------------|-------------------|-------|
| Knee surgery | C001 | 2 (MRI + Ortho consult) | ~2,695 | 1,081 | **60%** |
| Chemo cycle  | C003 | 2 (Oncology note + Labs) | ~2,252 | 994   | **56%** |
| Spine fusion | C004 | 3 (MRI + Neuro + PT)     | ~3,463 | 1,602 | **54%** |

---

## Optimisation 2 — Concise vs verbose prompts

Same pre-extracted content; different prompt style.

| Scenario | Claim | Concise tokens | Verbose tokens | Token saving | Cost saving |
|----------|-------|---------------|----------------|-------------|-------------|
| Knee surgery  | C001 | 1,817 | 4,727 | **62%** | $0.000274 → $0.000710 |
| Spine fusion  | C004 | 2,444 | 5,581 | **56%** | $0.000367 → $0.000837 |

---

## Optimisation 3 — Content-hash cache (return patient)

John Doe submits claim C006 (post-op complication) 3 days after C001.
PDF summaries and claims-pattern are already cached by content hash — 0 tokens reused.

| Step | First visit (C001) | Return visit (C006) | Saving |
|------|--------------------|---------------------|--------|
| PDF summarisation | 1,081 tokens | **0 (cache hit)** | 100% |
| Claims pattern    | 250 tokens   | **0 (cache hit)** | 100% |
| Analyst brief     | 486 tokens   | 494 tokens        | —     |
| **Total**         | **1,817**    | **494**           | **73%** |

Cost: $0.000274 → $0.000074 for the return visit.

---

## Optimisation 4 — Retry + loop detection (no wasted tokens)

**Scenario 2 (DB retry):**  
- Patient profile fetch: failed on attempt 1, 2 → succeeded on attempt 3  
- Claims history fetch: failed on attempt 1 → succeeded on attempt 2  
- **0 extra LLM tokens** — retries hit only the DB layer, not the LLM  
- Final brief still 100% complete (4/4 data sources gathered)

**Scenario 5 (loop detection):**  
- Guidelines DB returned empty on all 4 attempts (`simulate_loop=True`)  
- `node_visit_counts["fetch_guidelines"]` hit `MAX_NODE_VISITS=4` → skipped  
- LangGraph compiled the brief with 3/4 data sources — no infinite loop  
- **490 tokens** total (PDFs + history cached; brief still generated)

---

## Full session summary (all 5 scenarios)

| Metric | Value |
|--------|-------|
| Real LLM API calls made | 7 |
| Total input tokens | 4,226 |
| Total output tokens | 2,339 |
| Total tokens | 6,565 |
| Estimated cost | **$0.00204** |

---

## What is and isn't real

| Component | Real or simulated? |
|-----------|--------------------|
| DB fetch failures (retries) | **Real** — `time.sleep()` latency + actual failure flag |
| Token counting | **Real** — `tiktoken` counts every prompt character-for-character |
| LLM API calls | **Real** — OpenAI gpt-4o-mini, responses shown above |
| Content-hash cache | **Real** — SHA-256 keyed dict, 0 tokens on hit |
| Loop detection | **Real** — `node_visit_counts` in LangGraph state, fires at MAX_NODE_VISITS |
| Patient/PDF data | Simulated mock data (realistic text extracted from synthetic reports) |
| ML risk model | Simulated XGBoost-style scoring (deterministic, no LLM) |
