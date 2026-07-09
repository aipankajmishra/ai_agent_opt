"""
claims_review_agent.py - LangGraph agent that prepares a complete analyst package
for the Human-in-the-Loop (HITL) reviewer.

Flow:
  fetch_patient_profile
    -> (retry / loop detection) ->
  fetch_medical_records
    -> (retry / loop detection) ->
  process_documents          ← LLM: PDF summarisation (token-optimised)
    ->
  fetch_claims_history
    -> (retry / loop detection) ->
  analyze_claims_pattern     ← LLM: claims pattern analysis (token-optimised)
    ->
  fetch_guidelines
    -> (retry / loop detection) ->
  compile_analyst_brief      ← LLM: final brief (token-optimised)
    ->
  END  (state.analyst_brief is ready for the human analyst)

Optimisation techniques used:
  * Concise vs verbose prompt comparison (each LLM step)
  * Pre-extraction of document sections before LLM call
  * Content-hash caching of PDF summaries and claims patterns
  * Loop detection (node_visit_counts >= MAX_NODE_VISITS -> skip, continue)
  * Retry with graceful degradation (max retries -> proceed without that data)
"""

from __future__ import annotations
import time
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END

from core.state import ClaimsReviewState, MAX_RETRIES, MAX_NODE_VISITS
from ml_models.risk_model import ClaimRiskModel, prepare_claim_features
from tools import data_fetcher as db
from tools.document_processor import (
    process_all_pdfs,
    analyze_claims_pattern  as _dp_analyze_claims_pattern,
    compile_analyst_brief   as _dp_compile_brief,
)

_risk_model = ClaimRiskModel()

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _inc_visits(state: ClaimsReviewState, node: str) -> Dict[str, int]:
    counts = dict(state.get("node_visit_counts") or {})
    counts[node] = counts.get(node, 0) + 1
    return counts


def _get_visits(state: ClaimsReviewState, node: str) -> int:
    return (state.get("node_visit_counts") or {}).get(node, 0)


def _log(state: ClaimsReviewState, msg: str) -> list:
    log = list(state.get("agent_log") or [])
    log.append(msg)
    return log


def _errors(state: ClaimsReviewState, msg: str) -> list:
    errs = list(state.get("errors") or [])
    errs.append(msg)
    return errs


def _completeness(state: ClaimsReviewState, key: str, val: bool) -> Dict[str, bool]:
    c = dict(state.get("data_completeness") or {})
    c[key] = val
    return c


def _scenario(state: ClaimsReviewState) -> Dict[str, Any]:
    return state.get("scenario_config") or {}


# ══════════════════════════════════════════════════════════════════════════════
# NODE: fetch_patient_profile
# ══════════════════════════════════════════════════════════════════════════════

def fetch_patient_profile(state: ClaimsReviewState) -> dict:
    node = "fetch_patient_profile"
    visits = _inc_visits(state, node)
    visit_n = visits[node]
    cfg    = _scenario(state)
    fail_n = cfg.get("fail_patient_fetch_times", 0)

    log = _log(state, f"[{node}] attempt {visit_n}/{MAX_NODE_VISITS}")

    success, profile = db.fetch_patient_profile(
        state["patient_id"], attempt=visit_n, fail_times=fail_n
    )

    if success and profile:
        log.append(f"[{node}] ✅ patient profile loaded ({profile.get('name')})")
        return {
            "patient_profile": profile,
            "node_visit_counts": visits,
            "data_completeness": _completeness(state, "patient_profile", True),
            "agent_log": log,
            "current_step": node,
        }

    errs = _errors(state, f"{node}: attempt {visit_n} failed - DB unavailable")
    log.append(f"[{node}] ❌ failed (attempt {visit_n})")
    return {
        "patient_profile": None,
        "node_visit_counts": visits,
        "data_completeness": _completeness(state, "patient_profile", False),
        "errors": errs,
        "agent_log": log,
        "current_step": node,
    }


def _route_patient_profile(state: ClaimsReviewState) -> str:
    if state.get("patient_profile") is not None:
        return "fetch_medical_records"
    visits = _get_visits(state, "fetch_patient_profile")
    if visits >= MAX_NODE_VISITS:
        # Graceful degradation: max retries hit, proceed without patient profile
        return "fetch_medical_records"
    return "fetch_patient_profile"     # retry


# ══════════════════════════════════════════════════════════════════════════════
# NODE: fetch_medical_records
# ══════════════════════════════════════════════════════════════════════════════

def fetch_medical_records(state: ClaimsReviewState) -> dict:
    node = "fetch_medical_records"
    visits = _inc_visits(state, node)
    visit_n = visits[node]
    cfg    = _scenario(state)
    fail_n = cfg.get("fail_records_fetch_times", 0)

    log = _log(state, f"[{node}] attempt {visit_n}/{MAX_NODE_VISITS}")

    success, records = db.fetch_medical_records(
        state["patient_id"], attempt=visit_n, fail_times=fail_n
    )

    if success and records is not None:
        log.append(f"[{node}] ✅ {len(records)} medical document(s) retrieved")
        return {
            "raw_medical_pdfs": records,
            "node_visit_counts": visits,
            "data_completeness": _completeness(state, "medical_records", True),
            "agent_log": log,
            "current_step": node,
        }

    errs = _errors(state, f"{node}: attempt {visit_n} failed - document store unavailable")
    log.append(f"[{node}] ❌ failed (attempt {visit_n})")
    return {
        "raw_medical_pdfs": [],
        "node_visit_counts": visits,
        "data_completeness": _completeness(state, "medical_records", False),
        "errors": errs,
        "agent_log": log,
        "current_step": node,
    }


def _route_medical_records(state: ClaimsReviewState) -> str:
    # Success condition: records list was set (even if empty)
    records = state.get("raw_medical_pdfs")
    if records is not None:
        return "process_documents"
    visits = _get_visits(state, "fetch_medical_records")
    if visits >= MAX_NODE_VISITS:
        return "process_documents"     # proceed without docs
    return "fetch_medical_records"     # retry


# ══════════════════════════════════════════════════════════════════════════════
# NODE: process_documents   ← LLM CALL (token-optimised)
# ══════════════════════════════════════════════════════════════════════════════

def process_documents(state: ClaimsReviewState) -> dict:
    node  = "process_documents"
    pdfs  = state.get("raw_medical_pdfs") or []
    mode  = state.get("optimization_mode", "concise")
    log   = _log(state, f"[{node}] processing {len(pdfs)} document(s) in '{mode}' mode")

    profile  = state.get("patient_profile") or {}
    pat_name = profile.get("name", state.get("claim_data", {}).get("member_name", "Patient"))
    pat_age  = profile.get("age", 0)

    if not pdfs:
        log.append(f"[{node}] ⚠️  no documents - skipping LLM summarisation")
        return {
            "pdf_summaries": [],
            "agent_log": log,
            "current_step": node,
        }

    summaries, total_raw, total_actual, cache_hits = process_all_pdfs(
        pdfs, pat_name, pat_age, mode
    )

    saving_pct = ((total_raw - total_actual) / total_raw * 100) if total_raw > 0 else 0
    log.append(
        f"[{node}] ✅ {len(summaries)} doc(s) summarised | "
        f"raw~{total_raw} tok -> actual {total_actual} tok | "
        f"saved {saving_pct:.0f}% | cache hits: {cache_hits}"
    )

    # Accumulate token usage
    token_steps = dict(state.get("token_usage_by_step") or {})
    token_steps[node] = {
        "step": node,
        "prompt_tokens":     total_actual,
        "completion_tokens": 0,
        "total_tokens":      total_actual,
        "cost_usd":          (total_actual / 1000) * 0.00015,
        "cached":            cache_hits == len(summaries),
    }

    return {
        "pdf_summaries": summaries,
        "agent_log": log,
        "current_step": node,
        "token_usage_by_step": token_steps,
        "total_input_tokens": (state.get("total_input_tokens") or 0) + total_actual,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE: fetch_claims_history
# ══════════════════════════════════════════════════════════════════════════════

def fetch_claims_history(state: ClaimsReviewState) -> dict:
    node = "fetch_claims_history"
    visits = _inc_visits(state, node)
    visit_n = visits[node]
    cfg    = _scenario(state)
    fail_n = cfg.get("fail_history_fetch_times", 0)

    log = _log(state, f"[{node}] attempt {visit_n}/{MAX_NODE_VISITS}")

    success, history = db.fetch_claims_history(
        state["patient_id"], attempt=visit_n, fail_times=fail_n
    )

    if success and history is not None:
        log.append(f"[{node}] ✅ {len(history)} historical claim(s) retrieved")
        return {
            "claims_history": history,
            "node_visit_counts": visits,
            "data_completeness": _completeness(state, "claims_history", True),
            "agent_log": log,
            "current_step": node,
        }

    errs = _errors(state, f"{node}: attempt {visit_n} failed - claims DB unavailable")
    log.append(f"[{node}] ❌ failed (attempt {visit_n})")
    return {
        "claims_history": None,
        "node_visit_counts": visits,
        "data_completeness": _completeness(state, "claims_history", False),
        "errors": errs,
        "agent_log": log,
        "current_step": node,
    }


def _route_claims_history(state: ClaimsReviewState) -> str:
    if state.get("claims_history") is not None:
        return "analyze_claims_pattern"
    visits = _get_visits(state, "fetch_claims_history")
    if visits >= MAX_NODE_VISITS:
        return "analyze_claims_pattern"    # degrade gracefully
    return "fetch_claims_history"          # retry


# ══════════════════════════════════════════════════════════════════════════════
# NODE: analyze_claims_pattern   ← LLM CALL (token-optimised)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_claims_pattern(state: ClaimsReviewState) -> dict:
    node    = "analyze_claims_pattern"
    mode    = state.get("optimization_mode", "concise")
    history = state.get("claims_history") or []
    claim   = state.get("claim_data") or {}
    log     = _log(state, f"[{node}] analysing {len(history)} claim(s) in '{mode}' mode")

    pattern, raw_est, actual, cached = _dp_analyze_claims_pattern(
        state["patient_id"], history, claim, mode
    )

    saving_pct = ((raw_est - actual) / raw_est * 100) if raw_est > 0 and not cached else 0
    status = "cached 💾" if cached else f"raw~{raw_est} tok -> actual {actual} tok | saved {saving_pct:.0f}%"
    log.append(f"[{node}] ✅ pattern='{pattern.get('pattern','?')}' | {status}")

    token_steps = dict(state.get("token_usage_by_step") or {})
    token_steps[node] = {
        "step": node, "prompt_tokens": actual, "completion_tokens": 0,
        "total_tokens": actual, "cost_usd": (actual / 1000) * 0.00015, "cached": cached,
    }

    return {
        "claims_pattern": pattern,
        "agent_log": log,
        "current_step": node,
        "token_usage_by_step": token_steps,
        "total_input_tokens": (state.get("total_input_tokens") or 0) + actual,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE: fetch_guidelines
# ══════════════════════════════════════════════════════════════════════════════

def fetch_guidelines(state: ClaimsReviewState) -> dict:
    """
    Fetch treatment guidelines.  Loop detection:
      If the function returns empty data AND the node has been visited >= MAX_NODE_VISITS,
      we conclude data is genuinely unavailable and proceed without.
    """
    node = "fetch_guidelines"
    visits = _inc_visits(state, node)
    visit_n = visits[node]
    cfg    = _scenario(state)

    proc_key     = (state.get("claim_data") or {}).get("procedure_key", "")
    fail_n       = cfg.get("fail_guidelines_fetch_times", 0)
    simulate_loop = cfg.get("simulate_guidelines_loop", False)

    log = _log(state, f"[{node}] attempt {visit_n}/{MAX_NODE_VISITS}")

    success, guidelines = db.fetch_treatment_guidelines(
        proc_key, attempt=visit_n, fail_times=fail_n, simulate_loop=simulate_loop
    )

    if success and guidelines:
        log.append(f"[{node}] ✅ guidelines loaded for '{proc_key}'")
        return {
            "treatment_guidelines": guidelines,
            "node_visit_counts": visits,
            "data_completeness": _completeness(state, "guidelines", True),
            "agent_log": log,
            "current_step": node,
        }

    # Data missing / empty
    errs = _errors(state, f"{node}: attempt {visit_n} - no guidelines for '{proc_key}'")
    log.append(f"[{node}] ⚠️  no/empty guidelines (attempt {visit_n})")
    return {
        "treatment_guidelines": None,
        "node_visit_counts": visits,
        "data_completeness": _completeness(state, "guidelines", False),
        "errors": errs,
        "agent_log": log,
        "current_step": node,
    }


def _route_guidelines(state: ClaimsReviewState) -> str:
    if state.get("treatment_guidelines"):
        return "compile_analyst_brief"
    visits = _get_visits(state, "fetch_guidelines")
    if visits >= MAX_NODE_VISITS:
        # Loop detected - proceed without guidelines (graceful degradation)
        return "compile_analyst_brief"
    return "fetch_guidelines"    # retry


# ══════════════════════════════════════════════════════════════════════════════
# NODE: compile_analyst_brief   ← LLM CALL (token-optimised)
# ══════════════════════════════════════════════════════════════════════════════

def compile_analyst_brief_node(state: ClaimsReviewState) -> dict:
    node = "compile_analyst_brief"
    mode = state.get("optimization_mode", "concise")
    log  = _log(state, f"[{node}] compiling analyst brief in '{mode}' mode")

    brief, raw_est, actual = _dp_compile_brief(
        claim=state.get("claim_data") or {},
        patient_profile=state.get("patient_profile"),
        pdf_summaries=state.get("pdf_summaries") or [],
        claims_pattern=state.get("claims_pattern"),
        guidelines=state.get("treatment_guidelines"),
        risk_score=state.get("risk_score", 0.0),
        risk_decision=state.get("risk_decision", "NEEDS_REVIEW"),
        risk_features=state.get("risk_features") or {},
        optimization_mode=mode,
    )

    saving_pct = ((raw_est - actual) / raw_est * 100) if raw_est > 0 else 0
    log.append(
        f"[{node}] ✅ brief compiled | "
        f"raw~{raw_est} tok -> actual {actual} tok | saved {saving_pct:.0f}%"
    )

    token_steps = dict(state.get("token_usage_by_step") or {})
    token_steps[node] = {
        "step": node, "prompt_tokens": actual, "completion_tokens": 0,
        "total_tokens": actual, "cost_usd": (actual / 1000) * 0.00015, "cached": False,
    }

    total_in  = (state.get("total_input_tokens") or 0) + actual
    total_out = state.get("total_output_tokens") or 0

    return {
        "analyst_brief": brief,
        "agent_log": log,
        "current_step": node,
        "token_usage_by_step": token_steps,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_cost_usd": (total_in / 1000) * 0.00015 + (total_out / 1000) * 0.0006,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Build the LangGraph graph
# ══════════════════════════════════════════════════════════════════════════════

def build_claims_review_graph() -> Any:
    """Compile and return the LangGraph StateGraph."""

    builder = StateGraph(ClaimsReviewState)

    # Register nodes
    builder.add_node("fetch_patient_profile",  fetch_patient_profile)
    builder.add_node("fetch_medical_records",   fetch_medical_records)
    builder.add_node("process_documents",       process_documents)
    builder.add_node("fetch_claims_history",    fetch_claims_history)
    builder.add_node("analyze_claims_pattern",  analyze_claims_pattern)
    builder.add_node("fetch_guidelines",        fetch_guidelines)
    builder.add_node("compile_analyst_brief",   compile_analyst_brief_node)

    # Entry point
    builder.set_entry_point("fetch_patient_profile")

    # Conditional retry edges
    builder.add_conditional_edges(
        "fetch_patient_profile",
        _route_patient_profile,
        {
            "fetch_patient_profile": "fetch_patient_profile",   # retry (cycle)
            "fetch_medical_records": "fetch_medical_records",   # continue
        },
    )

    builder.add_conditional_edges(
        "fetch_medical_records",
        _route_medical_records,
        {
            "fetch_medical_records": "fetch_medical_records",   # retry
            "process_documents":     "process_documents",       # continue
        },
    )

    # After processing docs -> always go to fetch_claims_history
    builder.add_edge("process_documents", "fetch_claims_history")

    builder.add_conditional_edges(
        "fetch_claims_history",
        _route_claims_history,
        {
            "fetch_claims_history":   "fetch_claims_history",   # retry
            "analyze_claims_pattern": "analyze_claims_pattern", # continue
        },
    )

    # After pattern analysis -> fetch guidelines
    builder.add_edge("analyze_claims_pattern", "fetch_guidelines")

    builder.add_conditional_edges(
        "fetch_guidelines",
        _route_guidelines,
        {
            "fetch_guidelines":     "fetch_guidelines",       # retry / loop
            "compile_analyst_brief": "compile_analyst_brief", # continue
        },
    )

    # Final step -> END
    builder.add_edge("compile_analyst_brief", END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Public entry-point
# ══════════════════════════════════════════════════════════════════════════════

_graph = None   # lazily compiled


def run_claims_review(
    claim: dict,
    risk_score: float,
    risk_decision: str,
    risk_features: dict,
    optimization_mode: str = "concise",
    scenario_config: dict | None = None,
) -> ClaimsReviewState:
    """
    Run the full HITL preparation pipeline for a single NEEDS_REVIEW claim.

    Returns the final LangGraph state, which contains:
      - analyst_brief: text ready for the human analyst
      - pdf_summaries: per-document LLM summaries with token savings
      - claims_pattern: LLM-analysed claims pattern
      - agent_log: full activity log (retries, loop events, timings)
      - token_usage_by_step: breakdown of tokens per LLM node
    """
    global _graph
    if _graph is None:
        _graph = build_claims_review_graph()

    initial: ClaimsReviewState = {
        # Input
        "claim_id":       claim["claim_id"],
        "patient_id":     claim["patient_id"],
        "claim_data":     claim,
        "optimization_mode": optimization_mode,
        "scenario_config":   scenario_config or {},
        # ML decision
        "risk_score":    risk_score,
        "risk_decision": risk_decision,
        "risk_features": risk_features,
        # Empty gathered data
        "patient_profile":     None,
        "raw_medical_pdfs":    [],
        "pdf_summaries":       [],
        "claims_history":      None,
        "claims_pattern":      None,
        "treatment_guidelines": None,
        # Tracking
        "current_step":       "start",
        "node_visit_counts":  {},
        "data_completeness":  {},
        "errors":             [],
        "agent_log":          [f"=== Starting review for {claim['claim_id']} ({optimization_mode} mode) ==="],
        # Output
        "analyst_brief": None,
        # Token tracking
        "token_usage_by_step": {},
        "total_input_tokens":  0,
        "total_output_tokens": 0,
        "estimated_cost_usd":  0.0,
    }

    t0     = time.perf_counter()
    result = _graph.invoke(initial)
    elapsed = (time.perf_counter() - t0) * 1000

    result["agent_log"] = list(result.get("agent_log", [])) + [
        f"=== Review complete in {elapsed:.0f} ms ==="
    ]
    return result
