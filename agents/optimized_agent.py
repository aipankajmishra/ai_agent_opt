"""
Optimized Claims Review Agent – ALL optimizations enabled.

Optimization layers applied (5 total):

  Layer 1 – Pre-extraction (0 tokens, pure Python)
    Strip documents to clinically relevant sections before LLM call.
    Reduces document content by 50-70%.

  Layer 2 – Concise structured prompts
    JSON-output prompts instead of open-ended prose.
    Reduces prompt + completion tokens by ~60%.

  Layer 3 – LRU / content-hash cache
    SHA-256 of (relevant_content + mode) → skip LLM for identical docs.
    Used by existing document_processor.py.

  Layer 4 – Semantic similarity cache
    TF-IDF cosine similarity (threshold 0.82) → skip LLM for near-identical
    prompts. Catches same-patient / minor-variation cases that hash cache misses.

  Layer 5 – Context prefix cache
    Simulates Anthropic/OpenAI prompt caching: repeated patient-context prefix
    charged at 10% of normal input token price.

  Cascading Subagents
    Phase 1 (3 parallel threads): fetch patient profile + medical records
                                  + claims history concurrently.
    Phase 2 (3 parallel threads): process documents + analyze claims pattern
                                  + fetch guidelines concurrently (after Phase 1).
    Phase 3 (main thread):        semantic-cache lookup → compile analyst brief
                                  → context-cache charge → semantic-cache store.

  Failure resilience (optimized route does all of this; unoptimized does none):
    - Each DB fetch retries up to MAX_RETRIES times on transient failure.
    - Guidelines loop detection: if guidelines DB returns empty MAX_GUIDELINES_VISITS
      times in a row, the agent breaks out and compiles the brief without guidelines
      (graceful degradation rather than an infinite retry loop).
    - All failures and retries are recorded in RouteResult.failure_log.
"""

from __future__ import annotations
import time
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tools import data_fetcher as db
from tools.document_processor import (
    process_all_pdfs,
    analyze_claims_pattern,
    compile_analyst_brief,
)
from core.llm_engine import snapshot_tracker, delta_from_snapshot
from core.semantic_cache import semantic_cache
from core.context_cache import context_cache

_PRICE_IN  = 0.00015   # per 1K input tokens
_PRICE_OUT = 0.0006    # per 1K output tokens
_enc = tiktoken.encoding_for_model("gpt-4o-mini")

MAX_RETRIES           = 3    # max per-fetch retry attempts before giving up
MAX_GUIDELINES_VISITS = 3    # loop-detection threshold for guidelines node


# ══════════════════════════════════════════════════════════════════════════════
# Shared result dataclass  (imported by unoptimized_agent too)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteResult:
    """Uniform result structure for both optimized and unoptimized routes."""
    route: str                          # "optimized" | "unoptimized"
    claim_id: str        = ""
    risk_score: float    = 0.0
    risk_decision: str   = ""
    wall_time_ms: float  = 0.0

    # Token / cost metrics
    llm_api_calls: int          = 0
    total_input_tokens: int     = 0
    total_output_tokens: int    = 0
    total_tokens: int           = 0
    estimated_cost_usd: float   = 0.0

    # Cache breakdown (only meaningful for the optimized route)
    lru_cache_hits: int         = 0
    semantic_cache_hits: int    = 0
    context_tokens_saved: int   = 0

    # Failure / resilience tracking
    retry_count: int            = 0
    loop_detected: bool         = False
    failure_log: List[str]      = field(default_factory=list)

    # Output
    analyst_brief: str          = ""
    step_times_ms: Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Retry helper
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_with_retry(
    fetch_fn,
    label: str,
    fail_times: int,
) -> Tuple[Any, List[str], int]:
    """
    Call fetch_fn(attempt=n, fail_times=fail_times) up to MAX_RETRIES+1 times.

    Returns (data_or_None, log_lines, retries_performed).
    On exhaustion returns (None, log, MAX_RETRIES) – caller proceeds with None.
    """
    log: List[str] = []
    for attempt in range(1, MAX_RETRIES + 2):          # 1 … MAX_RETRIES+1
        success, data = fetch_fn(attempt=attempt, fail_times=fail_times)
        if success:
            if attempt > 1:
                log.append(f"[{label}] ✅ recovered after {attempt-1} retry(ies)")
            return data, log, attempt - 1
        log.append(f"[{label}] ❌ attempt {attempt}/{MAX_RETRIES+1} failed")
        if attempt > MAX_RETRIES:
            log.append(f"[{label}] ⚠️  max retries exhausted – proceeding without data")
            return None, log, MAX_RETRIES
    return None, log, MAX_RETRIES


# ══════════════════════════════════════════════════════════════════════════════
# Per-phase subagent callables
# ══════════════════════════════════════════════════════════════════════════════

def _sa_fetch_patient(patient_id: str, fail_times: int = 0):
    return _fetch_with_retry(
        lambda attempt, fail_times: db.fetch_patient_profile(patient_id, attempt, fail_times),
        "patient", fail_times,
    )

def _sa_fetch_records(patient_id: str, fail_times: int = 0):
    return _fetch_with_retry(
        lambda attempt, fail_times: db.fetch_medical_records(patient_id, attempt, fail_times),
        "records", fail_times,
    )

def _sa_fetch_history(patient_id: str, fail_times: int = 0):
    return _fetch_with_retry(
        lambda attempt, fail_times: db.fetch_claims_history(patient_id, attempt, fail_times),
        "history", fail_times,
    )

def _sa_process_docs(records, name, age):
    return process_all_pdfs(records, name, age, optimization_mode="concise")

def _sa_analyze_pattern(patient_id, history, claim):
    return analyze_claims_pattern(patient_id, history, claim, optimization_mode="concise")

def _sa_fetch_guidelines_with_loop_detection(
    proc_key: str,
    fail_times: int = 0,
    simulate_loop: bool = False,
) -> Tuple[Optional[Dict], bool, List[str]]:
    """
    Fetch guidelines with loop detection.

    simulate_loop=True means the guidelines DB always returns empty data
    (simulates an infinite-empty-response stuck node).

    Returns (guidelines_or_None, loop_was_detected, log_lines).
    """
    log: List[str] = []
    for visit in range(1, MAX_GUIDELINES_VISITS + 1):
        success, guidelines = db.fetch_treatment_guidelines(
            proc_key,
            attempt=visit,
            fail_times=fail_times,
            simulate_loop=simulate_loop,
        )
        if success and guidelines:
            return guidelines, False, log
        log.append(
            f"[guidelines] visit {visit}/{MAX_GUIDELINES_VISITS} – "
            f"{'empty response (loop simulation)' if simulate_loop else 'failed'}"
        )

    # Loop detected
    log.append(
        f"[guidelines] ⛔ loop detected after {MAX_GUIDELINES_VISITS} visits – "
        "compiling brief without guidelines (graceful degradation)"
    )
    return None, True, log


# ══════════════════════════════════════════════════════════════════════════════
# Public entry-point
# ══════════════════════════════════════════════════════════════════════════════

def run_optimized(
    claim: Dict[str, Any],
    risk_score: float,
    risk_decision: str,
    risk_features: Dict[str, float],
    scenario_config: Optional[Dict[str, Any]] = None,
) -> RouteResult:
    """
    Run the full claims review pipeline with ALL optimizations.

    scenario_config keys (all optional, default 0 / False):
      fail_patient_fetch_times   int  – how many consecutive patient fetches fail
      fail_records_fetch_times   int  – same for medical records
      fail_history_fetch_times   int  – same for claims history
      fail_guidelines_fetch_times int – same for guidelines
      simulate_guidelines_loop   bool – guidelines always returns empty (loop test)

    Pipeline:
      Phase 1 → 3 parallel fetch subagents       (with per-subagent retry)
      Phase 2 → 3 parallel analysis subagents    (guidelines has loop detection)
      Phase 3 → semantic cache → compile brief → context cache
    """
    cfg     = scenario_config or {}
    t_start = time.perf_counter()
    result  = RouteResult(
        route="optimized",
        claim_id=claim["claim_id"],
        risk_score=risk_score,
        risk_decision=risk_decision,
    )

    # ML triage: skip the whole LLM pipeline if auto-decided
    if risk_decision != "NEEDS_REVIEW":
        result.wall_time_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return result

    patient_id = claim["patient_id"]
    proc_key   = claim.get("procedure_key", "")
    snap0      = snapshot_tracker()

    # ── Phase 1: 3 parallel fetch subagents ──────────────────────────────────
    t0 = time.perf_counter()
    fail_pat  = cfg.get("fail_patient_fetch_times", 0)
    fail_rec  = cfg.get("fail_records_fetch_times",  0)
    fail_hist = cfg.get("fail_history_fetch_times",  0)

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="opt-fetch") as pool:
        f_patient = pool.submit(_sa_fetch_patient, patient_id, fail_pat)
        f_records = pool.submit(_sa_fetch_records,  patient_id, fail_rec)
        f_history = pool.submit(_sa_fetch_history,  patient_id, fail_hist)

        profile,     p_log, p_ret = f_patient.result()
        records_raw, r_log, r_ret = f_records.result()
        history_raw, h_log, h_ret = f_history.result()

    for log_lines in (p_log, r_log, h_log):
        result.failure_log.extend(log_lines)
    result.retry_count += p_ret + r_ret + h_ret

    records = records_raw or []
    history = history_raw or []
    result.step_times_ms["phase1_fetch_ms"] = round(
        (time.perf_counter() - t0) * 1000, 1
    )

    name = (profile or {}).get("name", claim.get("member_name", "Unknown"))
    age  = (profile or {}).get("age", 0)
    plan = (profile or {}).get("plan", "Unknown")

    # ── Phase 2: 3 parallel analysis subagents ────────────────────────────────
    t0 = time.perf_counter()
    fail_guide = cfg.get("fail_guidelines_fetch_times", 0)
    sim_loop   = cfg.get("simulate_guidelines_loop", False)

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="opt-analyze") as pool:
        f_docs       = pool.submit(_sa_process_docs, records, name, age)
        f_pattern    = pool.submit(_sa_analyze_pattern, patient_id, history, claim)
        f_guidelines = pool.submit(
            _sa_fetch_guidelines_with_loop_detection,
            proc_key, fail_guide, sim_loop,
        )

        pdf_summaries, _raw, _actual_doc_tok, lru_hits = f_docs.result()
        pattern_dict, _raw2, _pattern_tok, pat_cached  = f_pattern.result()
        guidelines, loop_detected, g_log               = f_guidelines.result()

    result.failure_log.extend(g_log)
    result.loop_detected = loop_detected
    result.lru_cache_hits = lru_hits + (1 if pat_cached else 0)
    result.step_times_ms["phase2_analysis_ms"] = round(
        (time.perf_counter() - t0) * 1000, 1
    )

    # ── Phase 3: Semantic cache → compile brief → context cache ──────────────
    sem_key = (
        f"brief|{name}|{age}|{claim.get('procedure','')}|"
        f"{len(records)}docs|{len(history)}hist|"
        f"{claim.get('billed_amount', 0):.0f}"
    )
    t0 = time.perf_counter()

    semantic_hit = semantic_cache.lookup(sem_key)
    if semantic_hit:
        result.analyst_brief       = semantic_hit
        result.semantic_cache_hits = 1
    else:
        # Layer 5: context prefix cache
        patient_ctx   = f"Patient: {name}, age {age}, plan: {plan}"
        ctx_tok_count = len(_enc.encode(patient_ctx))
        _, ctx_cached = context_cache.charge(patient_ctx, ctx_tok_count)
        if ctx_cached:
            result.context_tokens_saved = context_cache.stats()["context_tokens_saved"]

        brief, _, _ = compile_analyst_brief(
            claim=claim,
            patient_profile=profile,
            pdf_summaries=pdf_summaries,
            claims_pattern=pattern_dict,
            guidelines=guidelines,
            risk_score=risk_score,
            risk_decision=risk_decision,
            risk_features=risk_features,
            optimization_mode="concise",
        )
        result.analyst_brief = brief
        semantic_cache.store(sem_key, brief)

    result.step_times_ms["phase3_brief_ms"] = round(
        (time.perf_counter() - t0) * 1000, 1
    )

    # ── Finalise token/cost metrics ───────────────────────────────────────────
    delta = delta_from_snapshot(snap0)
    result.llm_api_calls       = delta["api_calls"] + delta["cache_hits"]
    result.total_input_tokens  = delta["total_input_tokens"]
    result.total_output_tokens = delta["total_output_tokens"]
    result.total_tokens        = result.total_input_tokens + result.total_output_tokens
    result.estimated_cost_usd  = float(delta["total_cost_usd"])
    result.wall_time_ms        = round((time.perf_counter() - t_start) * 1000, 1)

    return result
