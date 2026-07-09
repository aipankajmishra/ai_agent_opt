# Stress-Test Results – 30 Claims, Optimized vs Unoptimized Routes
### Generated 2026-07-10 03:05:47

## Test Protocol
- **LLM**: Real OpenAI API calls (gpt-4o-mini)
- **30 claims across 12 patients** processed sequentially in a single pass
- One `random.random()` score per claim — natural mix of decision bands
- Caches (LRU, semantic, context) accumulate naturally across claims
- Optimized route: 5-layer optimization + parallel subagents + retry + loop detection
- Unoptimized route: sequential, verbose prompts, full raw content, zero caches, no retry

## Part A – All 30 Claims Summary Table

| Claim | Patient | Procedure | Score | Decision | Opt Tokens | Unopt Tokens | Token Saving | Cost Saving | Opt Time | Unopt Time | LRU Hits | Sem Hits | Ctx Saved |
|-------|---------|-----------|-------|----------|-----------|-------------|-------------|------------|---------|-----------|---------|---------|----------|
| C001 | John Doe | Knee Arthroscopy with Meniscectomy | 0.6131 | NEEDS_REVIEW | 1,573 | 5,379 | +70.8% | +74.7% | 12571ms | 42760ms | 0 | 0 | 0 |
| C002 | Mary Johnson | Annual Wellness Examination | 0.4102 | NEEDS_REVIEW | 608 | 1,839 | +66.9% | +75.3% | 6485ms | 16678ms | 0 | 0 | 0 |
| C003 | Michael Chen | R-CHOP Chemotherapy Cycle 3 | 0.6737 | NEEDS_REVIEW | 1,433 | 5,022 | +71.5% | +77.0% | 10875ms | 47260ms | 0 | 0 | 0 |
| C004 | Sarah Williams | Lumbar Spinal Fusion L4-S1 | 0.1404 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C005 | Robert Davis | Emergency Appendectomy | 0.0060 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C006 | John Doe | Post-Operative Complication Visit | 0.2845 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C007 | John Doe | Post-Operative Knee Physical Therapy (12 | 0.5252 | NEEDS_REVIEW | 503 | 5,323 | +90.6% | +91.8% | 5456ms | 38466ms | 3 | 0 | 11 |
| C008 | Michael Chen | Interim PET/CT Scan – Lymphoma Response  | 0.5495 | NEEDS_REVIEW | 483 | 4,968 | +90.3% | +91.8% | 5481ms | 37932ms | 3 | 0 | 22 |
| C009 | Michael Chen | R-CHOP Chemotherapy Cycle 4 | 0.6218 | NEEDS_REVIEW | 0 | 5,022 | +100.0% | +100.0% | 129ms | 34594ms | 3 | 1 | 0 |
| C010 | Sarah Williams | Post-Operative Spine Follow-Up & X-ray | 0.4019 | NEEDS_REVIEW | 2,054 | 6,512 | +68.5% | +73.8% | 14041ms | 41144ms | 0 | 0 | 0 |
| C011 | Lisa Wang | Diabetes Management Comprehensive Visit | 0.1706 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C012 | Lisa Wang | Diabetic Foot Ulcer Debridement & Wound  | 0.2664 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C013 | James Wilson | Post-CABG Cardiology Follow-Up | 0.2837 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C014 | James Wilson | Stress Echocardiogram | 0.9778 | AUTO_DENY | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C015 | Emily Rodriguez | Total Hip Replacement – Right | 0.2516 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C016 | Emily Rodriguez | Post-Operative Hip Physical Therapy (10  | 0.9184 | AUTO_DENY | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C017 | David Kim | Annual Wellness Examination | 0.4157 | NEEDS_REVIEW | 609 | 1,845 | +67.0% | +75.9% | 6646ms | 15593ms | 0 | 0 | 0 |
| C018 | David Kim | Routine Vaccination – Tdap + Influenza | 0.2558 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C019 | Anna Martinez | COPD Exacerbation – Inpatient Admission  | 0.5281 | NEEDS_REVIEW | 1,640 | 5,279 | +68.9% | +73.4% | 9472ms | 33095ms | 0 | 0 | 0 |
| C020 | Anna Martinez | Transthoracic Echocardiogram | 0.4050 | NEEDS_REVIEW | 480 | 5,254 | +90.9% | +92.3% | 3132ms | 34515ms | 3 | 0 | 34 |
| C021 | Anna Martinez | Complex Medication Management Visit | 0.3367 | NEEDS_REVIEW | 485 | 5,231 | +90.7% | +91.9% | 3435ms | 30340ms | 3 | 0 | 46 |
| C022 | Tom Baker | MRI Shoulder – Right (without contrast) | 0.2433 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C023 | Tom Baker | Arthroscopic Rotator Cuff Repair – Right | 0.1307 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C024 | Rachel Green | Prenatal Visit – 28-week Gestation | 0.4366 | NEEDS_REVIEW | 607 | 1,990 | +69.5% | +77.4% | 5577ms | 16247ms | 0 | 0 | 0 |
| C025 | Rachel Green | Vaginal Delivery with Epidural – Term | 0.7136 | AUTO_DENY | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C026 | Mary Johnson | Excision of Suspicious Skin Lesion – Bac | 0.6639 | NEEDS_REVIEW | 390 | 1,915 | +79.6% | +84.2% | 4225ms | 16188ms | 1 | 0 | 58 |
| C027 | Robert Davis | ER Visit – Abdominal Pain | 0.1932 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C028 | Lisa Wang | Phacoemulsification Cataract Surgery – L | 0.3468 | NEEDS_REVIEW | 1,426 | 4,980 | +71.4% | +76.2% | 10662ms | 33841ms | 0 | 0 | 0 |
| C029 | David Kim | Screening Colonoscopy | 0.0062 | AUTO_APPROVE | 0 | 0 | +0.0% | +0.0% | 0ms | 0ms | 0 | 0 | 0 |
| C030 | James Wilson | Polysomnography – Sleep Study | 0.3832 | NEEDS_REVIEW | 1,549 | 5,066 | +69.4% | +74.3% | 9616ms | 31742ms | 0 | 0 | 0 |

**Decision breakdown:** 15 NEEDS_REVIEW | 12 AUTO_APPROVE | 3 AUTO_DENY

## Part A – Cache Effectiveness

- **LRU content-hash cache:** (per-document, accumulated across claims)
- **Semantic cache:** 0 hits / 1 entries (0.0%) — threshold: 0.82
- **Context prefix cache:** 0 reads / 0 tokens saved — unique prefixes: 1

## Part B – Failure Scenario Results

| Scenario | Opt Tokens | Unopt Tokens | Opt Retries | Loop Detected (opt) | Opt Time | Unopt Time |
|----------|-----------|-------------|------------|-------------------|---------|-----------|
| F1 – Transient DB Failure (patient fetch fails 2x) | 490 | 5,363 | 2 | — | 4969ms | 36163ms |
| F2 – Guidelines Loop (DB always returns empty) | 494 | 5,241 | 0 | ✅ YES | 4088ms | 34011ms |
| F3 – Multi-Resource Failure (records + history both fail once) | 547 | 1,908 | 2 | — | 5044ms | 14004ms |

### Failure Log Comparison

#### F1 – Transient DB Failure (patient fetch fails 2x)
> Patient profile DB fails on attempts 1 and 2, succeeds on attempt 3.

**Optimized agent log:**
- `[patient] ❌ attempt 1/4 failed`
- `[patient] ❌ attempt 2/4 failed`
- `[patient] ✅ recovered after 2 retry(ies)`

**Unoptimized agent log:**
- `[patient] ❌ fetch failed – no retry (unoptimized)`

#### F2 – Guidelines Loop (DB always returns empty)
> Guidelines DB always returns empty data — simulates a stuck node.

**Optimized agent log:**
- `[guidelines] visit 1/3 – empty response (loop simulation)`
- `[guidelines] visit 2/3 – empty response (loop simulation)`
- `[guidelines] visit 3/3 – empty response (loop simulation)`
- `[guidelines] ⛔ loop detected after 3 visits – compiling brief without guidelines (graceful degradation)`

**Unoptimized agent log:**
- `[guidelines] ❌ empty/failed – no loop detection (unoptimized) – brief will lack guidelines`

#### F3 – Multi-Resource Failure (records + history both fail once)
> Medical records fetch fails once, claims history fetch fails once.

**Optimized agent log:**
- `[records] ❌ attempt 1/4 failed`
- `[records] ✅ recovered after 1 retry(ies)`
- `[history] ❌ attempt 1/4 failed`
- `[history] ✅ recovered after 1 retry(ies)`

**Unoptimized agent log:**
- `[records] ❌ fetch failed – no retry (unoptimized)`
- `[history] ❌ fetch failed – no retry (unoptimized)`

## Optimization Architecture

```
OPTIMIZED ROUTE (5 layers)
  Phase 1 [parallel ThreadPoolExecutor, retry up to MAX_RETRIES=3]
    fetch_patient | fetch_records | fetch_history
  Phase 2 [parallel, cascades from Phase 1]
    process_all_pdfs (Layer 1: pre-extract, Layer 2: concise, Layer 3: LRU cache)
    analyze_claims_pattern (Layer 2: concise, Layer 3: LRU cache)
    fetch_guidelines_with_loop_detection
  Phase 3
    Layer 4: Semantic similarity cache (0 tokens on hit)
    Layer 5: Context prefix cache (90% discount on prefix)
    compile_analyst_brief

UNOPTIMIZED ROUTE (zero optimizations)
  Sequential: fetch_patient → fetch_records → fetch_history →
              fetch_guidelines → LLM(doc1) → LLM(doc2) → ... →
              LLM(claims_pattern) → LLM(brief with ALL raw content)
  ONE attempt per resource. No retry. No loop detection.
  No caching. Full verbose prompts.
```