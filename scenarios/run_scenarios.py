"""
run_scenarios.py - Five realistic HITL scenarios demonstrating the claims-review pipeline.
"""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from ml_models.risk_model import ClaimRiskModel
from data.claims_data import CLAIMS
from agents.claims_review_agent import run_claims_review
from core.llm_engine import tracker, clear_cache

_risk_model = ClaimRiskModel()

DIVIDER = "=" * 80
SUBDIV  = "-" * 80


# ══════════════════════════════════════════════════════════════════════════════
# Scenario definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    name: str
    description: str
    claim_id: str
    scenario_config: Dict[str, Any]
    demonstrates: List[str]


SCENARIOS: List[Scenario] = [

    # ── Scenario 1: Standard happy-path ──────────────────────────────────────
    Scenario(
        name="1. Standard Workflow",
        description="Knee surgery claim (C001) - all DB calls succeed first try. "
                    "Concise vs verbose token comparison.",
        claim_id="C001",
        scenario_config={},        # no failures
        demonstrates=[
            "Full 7-node LangGraph pipeline",
            "PDF pre-extraction -> 60% content reduction before LLM",
            "Structured JSON output (concise) vs prose (verbose)",
            "Per-step token breakdown",
        ],
    ),

    # ── Scenario 2: Transient DB failures + retry ─────────────────────────────
    Scenario(
        name="2. DB Retry Recovery",
        description="Chemo cycle claim (C003) - patient profile fetch fails twice, "
                    "claims history fails once. Agent retries and recovers.",
        claim_id="C003",
        scenario_config={
            "fail_patient_fetch_times": 2,    # fail on attempt 1,2 -> succeed on 3
            "fail_history_fetch_times": 1,    # fail on attempt 1 -> succeed on 2
        },
        demonstrates=[
            "Retry logic: conditional edges loop back to same node",
            "Partial failures do not abort the pipeline",
            "Error log captures each failed attempt",
            "Final brief still complete despite earlier failures",
        ],
    ),

    # ── Scenario 3: Large document set + chunked savings ─────────────────────
    Scenario(
        name="3. Multi-Document Case",
        description="Spine fusion claim (C004) - patient has 3 medical documents "
                    "(MRI, neurosurgery consult, PT discharge). Large token savings from "
                    "pre-extraction across multiple documents.",
        claim_id="C004",
        scenario_config={},        # no failures - focus on doc volume
        demonstrates=[
            "Pre-extraction applied per document (section-aware)",
            "Cumulative token savings across 3 documents",
            "Raw vs actual token cost comparison per document",
            "Combined savings: content reduction × concise prompts",
        ],
    ),

    # ── Scenario 4: Cache hit - return patient ───────────────────────────────
    Scenario(
        name="4. Cache Hit - Return Patient",
        description="John Doe (P001) submits a second claim (C006 - post-op complication) "
                    "3 days after C001 was processed. PDF summaries and claims history are "
                    "already cached.",
        claim_id="C006",
        scenario_config={},        # no failures - demonstrate cache
        demonstrates=[
            "Content-hash cache for PDF summaries (0 tokens on hit)",
            "Claims-pattern cache for same patient",
            "Near-instant processing vs first-visit latency",
            "Cache savings expressed as % and $ cost delta",
        ],
    ),

    # ── Scenario 5: Loop detection ────────────────────────────────────────────
    Scenario(
        name="5. Loop Detection - Guidelines Unavailable",
        description="Knee claim variant - guidelines DB keeps returning empty data "
                    "(simulated infinite-empty-response). Loop detector fires after "
                    f"MAX_NODE_VISITS attempts and compiles brief without guidelines.",
        claim_id="C001",
        scenario_config={
            "simulate_guidelines_loop": True,  # guidelines always return empty
        },
        demonstrates=[
            "node_visit_counts tracking per LangGraph node",
            "MAX_NODE_VISITS threshold triggers graceful skip",
            "Brief compiled with partial data (no guidelines section)",
            "errors[] list records the loop events",
        ],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_ml_result(claim: dict, risk_score: float, decision: str, features: dict) -> None:
    print(f"\n  📊 ML Risk Model:")
    print(f"     Score: {risk_score:.3f}  │  Decision: {decision}")
    top = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
    for k, v in top:
        print(f"     Factor '{k.replace('_',' ')}': {v:.3f}")


def _print_agent_log(log: List[str], show_full: bool = False) -> None:
    print(f"\n  🔄 Agent Activity Log:")
    display = log if show_full else log
    for line in display:
        indent = "     " if not line.startswith("===") else "  "
        print(f"{indent}{line}")


def _print_token_breakdown(token_steps: Dict[str, Any], total_in: int,
                            concise_run: bool = True) -> None:
    if not token_steps:
        return
    mode = "CONCISE" if concise_run else "VERBOSE"
    print(f"\n  🎯 Token Usage ({mode} mode):")
    print(f"     {'Step':<35} {'Tokens':>8}  {'Cached':>8}")
    print(f"     {'-'*35} {'-'*8}  {'-'*8}")
    for step_name, rec in token_steps.items():
        cached_str = "✅ HIT" if rec.get("cached") else "- API"
        print(f"     {step_name:<35} {rec['total_tokens']:>8}  {cached_str:>8}")
    print(f"     {'TOTAL':<35} {total_in:>8}")


def _print_pdf_savings(pdf_summaries: List[Dict]) -> None:
    if not pdf_summaries:
        print("\n  📄 No medical documents processed.")
        return
    print(f"\n  📄 Document Processing (token optimisation):")
    total_raw = total_actual = 0
    for s in pdf_summaries:
        cached_str = "💾 CACHED (0 tokens)" if s["cached"] else \
                     f"raw~{s['raw_tokens_estimate']} -> actual {s['actual_tokens']} (saved {s['token_savings_pct']:.0f}%)"
        print(f"     [{s['doc_type'].replace('_',' ').title():<30}]  {s['filename']}")
        print(f"       {cached_str}")
        total_raw    += s["raw_tokens_estimate"]
        total_actual += s["actual_tokens"]
    if total_raw > 0:
        total_saving = (total_raw - total_actual) / total_raw * 100
        print(f"\n     TOTAL: raw~{total_raw} tokens -> actual {total_actual} tokens  "
              f"(saved {total_saving:.0f}%)")


def _print_brief(brief: Optional[str]) -> None:
    if not brief:
        print("\n  📋 Analyst Brief: [not generated]")
        return
    print(f"\n  📋 Analyst Brief (ready for human reviewer):")
    print(f"  {SUBDIV}")
    for line in brief.split("\n"):
        print(f"  {line}")
    print(f"  {SUBDIV}")


def _print_retry_events(errors: List[str], log: List[str]) -> None:
    retry_lines = [l for l in log if "attempt" in l.lower() and ("✅" in l or "❌" in l)]
    if retry_lines:
        print(f"\n  🔁 Retry Events:")
        for line in retry_lines:
            print(f"     {line}")


def _print_loop_events(errors: List[str]) -> None:
    loop_errors = [e for e in errors if "guidelines" in e.lower()]
    if loop_errors:
        print(f"\n  ⛔ Loop Detection Events:")
        for e in loop_errors:
            print(f"     {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Concise vs Verbose comparison (run same claim twice)
# ══════════════════════════════════════════════════════════════════════════════

def run_with_comparison(scenario: Scenario) -> None:
    """
    Run a scenario in both concise and verbose modes and print the comparison.
    Used for Scenario 1 and Scenario 3 where the comparison is the main story.
    """
    claim = CLAIMS[scenario.claim_id]
    risk_score, risk_features = _risk_model.predict(claim)
    risk_decision = _risk_model.get_decision(risk_score)

    print(f"\n  Comparing CONCISE vs VERBOSE modes ...")

    # ── Concise run ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result_c = run_claims_review(
        claim=claim,
        risk_score=risk_score,
        risk_decision=risk_decision,
        risk_features=risk_features,
        optimization_mode="concise",
        scenario_config=scenario.scenario_config,
    )
    time_c = (time.perf_counter() - t0) * 1000

    # ── Verbose run ───────────────────────────────────────────────────────────
    # Clear LLM response cache so verbose makes fresh API calls
    clear_cache()
    t0 = time.perf_counter()
    result_v = run_claims_review(
        claim=claim,
        risk_score=risk_score,
        risk_decision=risk_decision,
        risk_features=risk_features,
        optimization_mode="verbose",
        scenario_config=scenario.scenario_config,
    )
    time_v = (time.perf_counter() - t0) * 1000

    tok_c = result_c.get("total_input_tokens", 0)
    tok_v = result_v.get("total_input_tokens", 0)
    saving_pct = ((tok_v - tok_c) / tok_v * 100) if tok_v > 0 else 0

    print(f"\n  {'Metric':<30} {'CONCISE':>12} {'VERBOSE':>12} {'Savings':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'LLM tokens (total)':<30} {tok_c:>12} {tok_v:>12} {saving_pct:>9.0f}%")
    print(f"  {'Processing time (ms)':<30} {time_c:>11.0f}ms {time_v:>11.0f}ms")
    print(f"  {'Estimated cost (USD)':<30} "
          f"  ${tok_c / 1e6 * 150:>9.5f}   ${tok_v / 1e6 * 150:>9.5f}")

    _print_pdf_savings(result_c.get("pdf_summaries") or [])
    _print_token_breakdown(
        result_c.get("token_usage_by_step") or {}, tok_c, concise_run=True
    )
    _print_brief(result_c.get("analyst_brief"))

    return result_c, result_v


# ══════════════════════════════════════════════════════════════════════════════
# Single scenario runner
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(scenario: Scenario, with_comparison: bool = False) -> None:
    print(f"\n{DIVIDER}")
    print(f"  SCENARIO: {scenario.name}")
    print(f"  Claim:  {scenario.claim_id}  │  {scenario.description}")
    print(f"  Demonstrates:")
    for d in scenario.demonstrates:
        print(f"    * {d}")
    print(DIVIDER)

    claim = CLAIMS[scenario.claim_id]
    risk_score, risk_features = _risk_model.predict(claim)
    risk_decision = _risk_model.get_decision(risk_score)

    _print_ml_result(claim, risk_score, risk_decision, risk_features)

    if risk_decision != "NEEDS_REVIEW":
        print(f"\n  ✅ ML Decision: {risk_decision} - no HITL review needed.")
        return

    print(f"\n  ↓  ML flagged for human review - LangGraph agent starting ...")

    if with_comparison:
        run_with_comparison(scenario)
        return

    # Standard single-mode run
    t0 = time.perf_counter()
    result = run_claims_review(
        claim=claim,
        risk_score=risk_score,
        risk_decision=risk_decision,
        risk_features=risk_features,
        optimization_mode="concise",
        scenario_config=scenario.scenario_config,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    _print_agent_log(result.get("agent_log") or [])
    _print_retry_events(result.get("errors") or [], result.get("agent_log") or [])
    _print_loop_events(result.get("errors") or [])
    _print_pdf_savings(result.get("pdf_summaries") or [])
    _print_token_breakdown(
        result.get("token_usage_by_step") or {},
        result.get("total_input_tokens", 0),
    )
    _print_brief(result.get("analyst_brief"))

    node_visits = result.get("node_visit_counts") or {}
    retried = {k: v for k, v in node_visits.items() if v > 1}
    if retried:
        print(f"\n  🔁 Retry Summary: {retried}")

    errors = result.get("errors") or []
    if errors:
        print(f"\n  ⚠️  Non-fatal errors encountered ({len(errors)}):")
        for e in errors:
            print(f"     {e}")

    completeness = result.get("data_completeness") or {}
    gathered = [k for k, v in completeness.items() if v]
    missing  = [k for k, v in completeness.items() if not v]
    print(f"\n  📦 Data Completeness: {len(gathered)}/{len(completeness)} gathered")
    if missing:
        print(f"     Missing (graceful degradation): {missing}")

    print(f"\n  ⏱  Total pipeline time: {elapsed:.0f} ms")
    print(f"  💰 Estimated LLM cost: ${result.get('estimated_cost_usd', 0):.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# Run all scenarios
# ══════════════════════════════════════════════════════════════════════════════

def run_all_scenarios() -> None:
    print(f"\n{DIVIDER}")
    print("  HEALTHCARE CLAIMS HITL REVIEW - TOKEN OPTIMISATION SCENARIOS")
    print("  Workflow: Claim -> ML Triage -> LangGraph Agent -> Analyst Package -> Human")
    print(f"{DIVIDER}")

    tracker.reset()
    clear_cache()

    # Scenario 1 - standard workflow with concise vs verbose comparison
    run_scenario(SCENARIOS[0], with_comparison=True)

    # Scenario 2 - retry recovery (no cache reset between scenarios)
    run_scenario(SCENARIOS[1])

    # Scenario 3 - multi-doc with comparison
    run_scenario(SCENARIOS[2], with_comparison=True)

    # Scenario 4 - cache hit (C006 is same patient as C001 which ran in Sc1/Sc3)
    # NOTE: do NOT clear cache before this one - that's the whole point
    run_scenario(SCENARIOS[3])

    # Scenario 5 - loop detection
    run_scenario(SCENARIOS[4])

    # ── Session summary ───────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  SESSION TOKEN SUMMARY (all scenarios)")
    print(DIVIDER)
    s = tracker.summary()
    print(f"  API calls made:        {s['api_calls']}")
    print(f"  Cache hits:            {s['cache_hits']} ({s['cache_hit_rate']})")
    print(f"  Total input tokens:    {s['total_input_tokens']}")
    print(f"  Total output tokens:   {s['total_output_tokens']}")
    print(f"  Total tokens:          {s['total_tokens']}")
    print(f"  Estimated cost:        {s['estimated_cost_usd']}")
    print(DIVIDER)
