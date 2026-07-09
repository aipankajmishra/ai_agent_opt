"""
stress_test.py – Single-pass 30-claim stress test: Optimized vs Unoptimized routes.

Processes ALL 30 claims sequentially in a single pass with one random score each.
Caches accumulate naturally — same-patient claims benefit from LRU document cache,
semantic brief cache, and context prefix cache. 

Part B – Failure scenarios (unchanged): 3 injected-failure cases.

Findings printed to stdout AND written to RESULTS.md.
"""

from __future__ import annotations
import os
import sys
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from data.claims_data import CLAIMS
from ml_models.risk_model import ClaimRiskModel
from agents.optimized_agent import RouteResult, run_optimized
from agents.unoptimized_agent import run_unoptimized
from core.llm_engine import clear_cache
from core.semantic_cache import semantic_cache
from core.context_cache import context_cache

_risk_model = ClaimRiskModel()

DIVIDER = "=" * 80
SUBDIV  = "-" * 80

# Process ALL claims in sorted order (C001, C002, ..., C030)
ALL_CLAIM_IDS = sorted(CLAIMS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Claim result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClaimResult:
    claim_id: str
    risk_score: float
    decision: str
    opt:   RouteResult = field(default_factory=lambda: RouteResult(route="optimized"))
    unopt: RouteResult = field(default_factory=lambda: RouteResult(route="unoptimized"))

    @property
    def token_saving_pct(self) -> float:
        if self.unopt.total_tokens <= 0:
            return 0.0
        return (self.unopt.total_tokens - self.opt.total_tokens) / self.unopt.total_tokens * 100

    @property
    def cost_saving_pct(self) -> float:
        if self.unopt.estimated_cost_usd <= 0:
            return 0.0
        return (
            (self.unopt.estimated_cost_usd - self.opt.estimated_cost_usd)
            / self.unopt.estimated_cost_usd * 100
        )

    @property
    def time_saving_pct(self) -> float:
        if self.unopt.wall_time_ms <= 0:
            return 0.0
        return (
            (self.unopt.wall_time_ms - self.opt.wall_time_ms)
            / self.unopt.wall_time_ms * 100
        )


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, int(pct / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def print_claim_comparison(cr: ClaimResult) -> None:
    """Print a formatted comparison table for one claim."""
    opt, unopt = cr.opt, cr.unopt
    dec_same = opt.risk_decision == unopt.risk_decision

    print(f"\n  Score: {cr.risk_score:.4f}  |  Decision: {cr.decision}"
          f"{'  ✅ (both agree)' if dec_same else '  ⚠️  (MISMATCH!)'}")

    if cr.decision != "NEEDS_REVIEW":
        print(f"  ⚡ ML triage → {cr.decision}. No LLM calls on either route (0 tokens, 0 ms).")
        return

    hdr = f"  {'Metric':<32} {'OPTIMIZED':>14} {'UNOPTIMIZED':>14} {'Δ Savings':>12}"
    sep = f"  {'─'*32} {'─'*14} {'─'*14} {'─'*12}"
    print(sep)
    print(hdr)
    print(sep)

    def row(label, opt_val, unopt_val, suffix="", saving_pct: Optional[float] = None):
        sp = f"{saving_pct:+.1f}%" if saving_pct is not None else ""
        print(f"  {label:<32} {str(opt_val)+suffix:>14} {str(unopt_val)+suffix:>14} {sp:>12}")

    row("Wall time",          f"{opt.wall_time_ms:.0f}", f"{unopt.wall_time_ms:.0f}", " ms",
        cr.time_saving_pct)
    row("LLM API calls",      opt.llm_api_calls,    unopt.llm_api_calls)
    row("Input tokens",       f"{opt.total_input_tokens:,}",
                               f"{unopt.total_input_tokens:,}",
         saving_pct=((unopt.total_input_tokens - opt.total_input_tokens)
                     / max(unopt.total_input_tokens, 1) * 100))
    row("Output tokens",      f"{opt.total_output_tokens:,}",
                               f"{unopt.total_output_tokens:,}")
    row("Total tokens",       f"{opt.total_tokens:,}",
                               f"{unopt.total_tokens:,}",
         saving_pct=cr.token_saving_pct)
    row("Est. cost (USD)",    f"${opt.estimated_cost_usd:.5f}",
                               f"${unopt.estimated_cost_usd:.5f}",
         saving_pct=cr.cost_saving_pct)
    print(sep)
    row("LRU cache hits",     opt.lru_cache_hits,    unopt.lru_cache_hits)
    row("Semantic cache hits",opt.semantic_cache_hits, unopt.semantic_cache_hits)
    row("Context tok. saved", f"{opt.context_tokens_saved:,}",
                               f"{unopt.context_tokens_saved:,}")
    print(sep)

    # Savings bar
    bar_tok = _bar(max(0, cr.token_saving_pct))
    bar_cst = _bar(max(0, cr.cost_saving_pct))
    print(f"\n  Token savings:  {bar_tok}  {cr.token_saving_pct:+.1f}%")
    print(f"  Cost  savings:  {bar_cst}  {cr.cost_saving_pct:+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Markdown output
# ══════════════════════════════════════════════════════════════════════════════

def _findings_to_markdown(
    results: List[ClaimResult],
    failure_findings: Optional[List] = None,
) -> str:
    lines: List[str] = []
    lines.append("# Stress-Test Results – 30 Claims, Optimized vs Unoptimized Routes")
    lines.append(f"### Generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Test Protocol")
    lines.append("- **LLM**: Real OpenAI API calls (gpt-4o-mini)")
    lines.append("- **30 claims across 12 patients** processed sequentially in a single pass")
    lines.append("- One `random.random()` score per claim — natural mix of decision bands")
    lines.append("- Caches (LRU, semantic, context) accumulate naturally across claims")
    lines.append("- Optimized route: 5-layer optimization + parallel subagents + retry + loop detection")
    lines.append("- Unoptimized route: sequential, verbose prompts, full raw content, zero caches, no retry")
    lines.append("")

    # Part A summary table
    lines.append("## Part A – All 30 Claims Summary Table")
    lines.append("")
    lines.append("| Claim | Patient | Procedure | Score | Decision | Opt Tokens | Unopt Tokens | Token Saving | Cost Saving | Opt Time | Unopt Time | LRU Hits | Sem Hits | Ctx Saved |")
    lines.append("|-------|---------|-----------|-------|----------|-----------|-------------|-------------|------------|---------|-----------|---------|---------|----------|")

    for cr in results:
        patient = CLAIMS[cr.claim_id]["member_name"]
        procedure = CLAIMS[cr.claim_id]["procedure"][:40]
        lines.append(
            f"| {cr.claim_id} | {patient} | {procedure} "
            f"| {cr.risk_score:.4f} | {cr.decision} "
            f"| {cr.opt.total_tokens:,} | {cr.unopt.total_tokens:,} "
            f"| {cr.token_saving_pct:+.1f}% | {cr.cost_saving_pct:+.1f}% "
            f"| {cr.opt.wall_time_ms:.0f}ms | {cr.unopt.wall_time_ms:.0f}ms "
            f"| {cr.opt.lru_cache_hits} | {cr.opt.semantic_cache_hits} | {cr.opt.context_tokens_saved:,} |"
        )

    # Decision breakdown
    review_count = sum(1 for cr in results if cr.decision == "NEEDS_REVIEW")
    approve_count = sum(1 for cr in results if cr.decision == "AUTO_APPROVE")
    deny_count = sum(1 for cr in results if cr.decision == "AUTO_DENY")
    lines.append("")
    lines.append(f"**Decision breakdown:** {review_count} NEEDS_REVIEW | {approve_count} AUTO_APPROVE | {deny_count} AUTO_DENY")

    # Cache stats
    sem_stats = semantic_cache.stats()
    ctx_stats = context_cache.stats()
    lines.append("")
    lines.append("## Part A – Cache Effectiveness")
    lines.append("")
    lines.append(f"- **LRU content-hash cache:** (per-document, accumulated across claims)")
    lines.append(f"- **Semantic cache:** {sem_stats['semantic_hits']} hits / {sem_stats['entries_stored']} entries ({sem_stats['semantic_hit_rate']}) — threshold: {sem_stats['threshold']}")
    lines.append(f"- **Context prefix cache:** {ctx_stats['context_cache_reads']} reads / {ctx_stats['context_tokens_saved']:,} tokens saved — unique prefixes: {ctx_stats['unique_prefixes_seen']}")

    # ── Part B: failure scenarios ─────────────────────────────────────────────
    if failure_findings:
        lines.append("")
        lines.append("## Part B – Failure Scenario Results")
        lines.append("")
        lines.append("| Scenario | Opt Tokens | Unopt Tokens | Opt Retries | Loop Detected (opt) | Opt Time | Unopt Time |")
        lines.append("|----------|-----------|-------------|------------|-------------------|---------|-----------|")
        for ff in failure_findings:
            lines.append(
                f"| {ff.scenario_name} "
                f"| {ff.opt.total_tokens:,} | {ff.unopt.total_tokens:,} "
                f"| {ff.opt.retry_count} "
                f"| {'✅ YES' if ff.opt.loop_detected else '—'} "
                f"| {ff.opt.wall_time_ms:.0f}ms | {ff.unopt.wall_time_ms:.0f}ms |"
            )

        lines.append("")
        lines.append("### Failure Log Comparison")
        lines.append("")
        for ff in failure_findings:
            lines.append(f"#### {ff.scenario_name}")
            lines.append(f"> {ff.description.split(chr(10))[0]}")
            lines.append("")
            lines.append("**Optimized agent log:**")
            for entry in (ff.opt.failure_log or ["(no failures)"]):
                lines.append(f"- `{entry}`")
            lines.append("")
            lines.append("**Unoptimized agent log:**")
            for entry in (ff.unopt.failure_log or ["(no failures)"]):
                lines.append(f"- `{entry}`")
            lines.append("")

    lines.append("## Optimization Architecture")
    lines.append("")
    lines.append("```")
    lines.append("OPTIMIZED ROUTE (5 layers)")
    lines.append("  Phase 1 [parallel ThreadPoolExecutor, retry up to MAX_RETRIES=3]")
    lines.append("    fetch_patient | fetch_records | fetch_history")
    lines.append("  Phase 2 [parallel, cascades from Phase 1]")
    lines.append("    process_all_pdfs (Layer 1: pre-extract, Layer 2: concise, Layer 3: LRU cache)")
    lines.append("    analyze_claims_pattern (Layer 2: concise, Layer 3: LRU cache)")
    lines.append("    fetch_guidelines_with_loop_detection")
    lines.append("  Phase 3")
    lines.append("    Layer 4: Semantic similarity cache (0 tokens on hit)")
    lines.append("    Layer 5: Context prefix cache (90% discount on prefix)")
    lines.append("    compile_analyst_brief")
    lines.append("")
    lines.append("UNOPTIMIZED ROUTE (zero optimizations)")
    lines.append("  Sequential: fetch_patient → fetch_records → fetch_history →")
    lines.append("              fetch_guidelines → LLM(doc1) → LLM(doc2) → ... →")
    lines.append("              LLM(claims_pattern) → LLM(brief with ALL raw content)")
    lines.append("  ONE attempt per resource. No retry. No loop detection.")
    lines.append("  No caching. Full verbose prompts.")
    lines.append("```")

    return "\n".join(lines)


def _write_results_md(content: str) -> None:
    results_path = os.path.join(os.path.dirname(__file__), "..", "RESULTS.md")
    results_path = os.path.normpath(results_path)
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\n  📄 Full findings written → {results_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Core stress-test logic – single pass, all 30 claims
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_claim(claim: Dict[str, Any]) -> ClaimResult:
    """Run both routes for a single claim with its random score."""
    risk_score = random.random()
    decision    = _risk_model.get_decision(risk_score)
    _, features = _risk_model.predict(claim, force_score=risk_score)

    if decision == "NEEDS_REVIEW":
        opt_result   = run_optimized(claim, risk_score, decision, features)
        unopt_result = run_unoptimized(claim, risk_score, decision, features)
    else:
        # ML triage — both routes skip LLM entirely
        opt_result   = RouteResult(route="optimized")
        unopt_result = RouteResult(route="unoptimized")

    return ClaimResult(
        claim_id=claim["claim_id"],
        risk_score=risk_score,
        decision=decision,
        opt=opt_result,
        unopt=unopt_result,
    )


def run_stress_test(seed: Optional[int] = None) -> None:
    """
    Full stress test:
      Part A – All 30 claims, sequential, one random score each
      Part B – 3 failure scenarios
    """
    if seed is not None:
        random.seed(seed)

    print(f"\n{DIVIDER}")
    print("  STRESS TEST: OPTIMIZED vs UNOPTIMIZED ROUTES")
    print("  30 claims across 12 patients, single sequential pass")
    print("  One random.random() score per claim")
    print("  Caches accumulate naturally — no artificial clearing between claims")
    print(DIVIDER)

    # Reset ALL caches at the start
    clear_cache()
    semantic_cache.clear()
    context_cache.clear()

    results: List[ClaimResult] = []
    grand_opt_tokens   = 0
    grand_unopt_tokens = 0
    grand_opt_cost     = 0.0
    grand_unopt_cost   = 0.0
    review_count       = 0
    auto_approve_count = 0
    auto_deny_count    = 0

    # ── Part A: sequential pass through all 30 claims ──────────────────────────
    for claim_id in ALL_CLAIM_IDS:
        claim = CLAIMS[claim_id]

        print(f"\n{DIVIDER}")
        print(f"  CLAIM {claim_id} – {claim.get('procedure','?')}")
        print(f"  Patient: {claim.get('member_name','?')} ({claim['patient_id']})")
        print(f"  Amount: ${claim.get('billed_amount', 0):,.2f}")
        print(DIVIDER)

        cr = _run_one_claim(claim)
        results.append(cr)

        # Count decisions
        if cr.decision == "NEEDS_REVIEW":
            review_count += 1
        elif cr.decision == "AUTO_APPROVE":
            auto_approve_count += 1
        else:
            auto_deny_count += 1

        # Quick summary line
        if cr.decision == "NEEDS_REVIEW":
            icon = "✅" if not cr.opt.failure_log else "⚠️ "
            print(f"    {icon} Optimized:   {cr.opt.total_tokens:>6,} tokens  "
                  f"{cr.opt.wall_time_ms:>6.0f} ms  "
                  f"LRU={cr.opt.lru_cache_hits} sem={cr.opt.semantic_cache_hits}  "
                  f"retries={cr.opt.retry_count}")
            print(f"    🐢 Unoptimized: {cr.unopt.total_tokens:>6,} tokens  "
                  f"{cr.unopt.wall_time_ms:>6.0f} ms")
        else:
            print(f"    ⚡ ML triage → {cr.decision} (0 tokens both routes)")

        print_claim_comparison(cr)

        grand_opt_tokens   += cr.opt.total_tokens
        grand_unopt_tokens += cr.unopt.total_tokens
        grand_opt_cost     += cr.opt.estimated_cost_usd
        grand_unopt_cost   += cr.unopt.estimated_cost_usd

    # ── Part A summary ────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  PART A SUMMARY – {len(results)} Claims Processed Sequentially")
    print(DIVIDER)
    print(f"  Decision breakdown:")
    print(f"    NEEDS_REVIEW:  {review_count} claims (full LLM pipeline)")
    print(f"    AUTO_APPROVE:  {auto_approve_count} claims (ML gate — 0 tokens)")
    print(f"    AUTO_DENY:     {auto_deny_count} claims (ML gate — 0 tokens)")

    if review_count > 0:
        tok_saving  = (grand_unopt_tokens - grand_opt_tokens) / max(grand_unopt_tokens, 1) * 100
        cost_saving = (grand_unopt_cost   - grand_opt_cost)   / max(grand_unopt_cost, 1e-9) * 100
        print(f"\n  Token / Cost Comparison (NEEDS_REVIEW claims only):")
        print(f"  {'Metric':<35} {'OPTIMIZED':>14} {'UNOPTIMIZED':>14}")
        print(f"  {'─'*35} {'─'*14} {'─'*14}")
        print(f"  {'Total tokens':<35} {grand_opt_tokens:>14,} {grand_unopt_tokens:>14,}")
        print(f"  {'Est. cost (USD)':<35} ${grand_opt_cost:>13.5f} ${grand_unopt_cost:>13.5f}")
        print(f"  {'Token saving':<35} {tok_saving:>13.1f}%")
        print(f"  {'Cost saving':<35} {cost_saving:>13.1f}%")

    sem_stats = semantic_cache.stats()
    ctx_stats = context_cache.stats()
    print(f"\n  Cache Statistics (natural accumulation across all 30 claims):")
    print(f"  {'─'*35} {'─'*14} {'─'*14}")
    print(f"  Semantic cache: {sem_stats['semantic_hits']} hits / "
          f"{sem_stats['entries_stored']} entries stored ({sem_stats['semantic_hit_rate']})")
    print(f"  Context cache:  {ctx_stats['context_cache_reads']} context prefix reads / "
          f"{ctx_stats['context_tokens_saved']:,} tokens saved")
    print(f"  Unique prefix keys seen: {ctx_stats['unique_prefixes_seen']}")

    # ── Part B: failure scenarios ─────────────────────────────────────────────
    failure_findings = run_failure_scenarios()

    # ── Write findings to RESULTS.md ─────────────────────────────────────────
    md = _findings_to_markdown(results, failure_findings)
    _write_results_md(md)
    print(DIVIDER)


# ══════════════════════════════════════════════════════════════════════════════
# Part B: Failure scenario runner (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureFinding:
    scenario_name: str
    claim_id: str
    description: str
    opt: RouteResult   = field(default_factory=lambda: RouteResult(route="optimized"))
    unopt: RouteResult = field(default_factory=lambda: RouteResult(route="unoptimized"))


FAILURE_SCENARIOS = [
    {
        "name": "F1 – Transient DB Failure (patient fetch fails 2x)",
        "claim_id": "C001",
        "force_score": 0.45,
        "scenario_config": {"fail_patient_fetch_times": 2},
        "description": (
            "Patient profile DB fails on attempts 1 and 2, succeeds on attempt 3.\n"
            "  Optimized: retries (up to MAX_RETRIES=3), recovers, brief is COMPLETE.\n"
            "  Unoptimized: one attempt, fails, brief compiled WITHOUT patient data."
        ),
        "what_to_watch": [
            "opt.retry_count should be 2",
            "opt.failure_log shows recovery message",
            "unopt.failure_log shows silent miss",
            "opt brief contains patient name; unopt may say 'Unknown'",
        ],
    },
    {
        "name": "F2 – Guidelines Loop (DB always returns empty)",
        "claim_id": "C001",
        "force_score": 0.45,
        "scenario_config": {"simulate_guidelines_loop": True},
        "description": (
            "Guidelines DB always returns empty data — simulates a stuck node.\n"
            "  Optimized: loop detection fires after MAX_GUIDELINES_VISITS=3 attempts,\n"
            "             compiles brief without guidelines (GRACEFUL DEGRADATION, LOGGED).\n"
            "  Unoptimized: one attempt, gets nothing, brief silently missing guidelines,\n"
            "               NO loop detection – a real stuck loop would run forever."
        ),
        "what_to_watch": [
            "opt.loop_detected == True",
            "opt.failure_log contains '⛔ loop detected'",
            "unopt.loop_detected == False (no detection at all)",
            "unopt.failure_log shows silent miss",
        ],
    },
    {
        "name": "F3 – Multi-Resource Failure (records + history both fail once)",
        "claim_id": "C004",
        "force_score": 0.45,
        "scenario_config": {
            "fail_records_fetch_times": 1,
            "fail_history_fetch_times": 1,
        },
        "description": (
            "Medical records fetch fails once, claims history fetch fails once.\n"
            "  Optimized: retries both, recovers both, brief is COMPLETE with all data.\n"
            "  Unoptimized: proceeds with empty records list AND empty history,\n"
            "               brief has no document summaries and no claims pattern."
        ),
        "what_to_watch": [
            "opt.retry_count should be 2 (one per resource)",
            "opt produces pdf_summaries; unopt produces none",
            "opt token count higher (it actually processed docs); unopt skips them",
        ],
    },
]


def _print_failure_comparison(ff: FailureFinding) -> None:
    opt, unopt = ff.opt, ff.unopt
    print(f"\n  {SUBDIV}")
    print(f"  {ff.scenario_name}")
    print(f"  {SUBDIV}")
    print(f"  {ff.description}")
    print(f"\n  {'Metric':<32} {'OPTIMIZED':>14} {'UNOPTIMIZED':>14}")
    print(f"  {'─'*32} {'─'*14} {'─'*14}")

    def row(label, ov, uv):
        print(f"  {label:<32} {str(ov):>14} {str(uv):>14}")

    row("Wall time",            f"{opt.wall_time_ms:.0f}ms",    f"{unopt.wall_time_ms:.0f}ms")
    row("Total tokens",         f"{opt.total_tokens:,}",         f"{unopt.total_tokens:,}")
    row("Retry count",          opt.retry_count,                 unopt.retry_count)
    row("Loop detected",        "✅ YES" if opt.loop_detected else "—",
                                 "⚠️  NO" if ff.opt.loop_detected else "—")
    print(f"  {'─'*32} {'─'*14} {'─'*14}")

    if opt.failure_log:
        print(f"\n  🔁 Optimized failure/retry log:")
        for line in opt.failure_log:
            print(f"       {line}")
    if unopt.failure_log:
        print(f"\n  ⚠️  Unoptimized failure log (no recovery):")
        for line in unopt.failure_log:
            print(f"       {line}")


def run_failure_scenarios() -> List[FailureFinding]:
    """Run Part B: inject specific failures into both routes and compare resilience."""
    print(f"\n{DIVIDER}")
    print("  PART B – FAILURE SCENARIO TESTS")
    print("  Same claim, forced NEEDS_REVIEW score (0.45)")
    print("  DB failures injected identically into both routes")
    print("  Optimized: retry + loop detection ON | Unoptimized: none of that")
    print(DIVIDER)

    results: List[FailureFinding] = []
    for sc in FAILURE_SCENARIOS:
        claim    = CLAIMS[sc["claim_id"]]
        score    = sc["force_score"]
        decision = _risk_model.get_decision(score)
        _, feats = _risk_model.predict(claim, force_score=score)
        cfg      = sc["scenario_config"]

        print(f"\n  ▶  {sc['name']}")
        print(f"     Claim: {sc['claim_id']} | score={score} → {decision}")

        # Clear caches between failure scenarios for clean measurement
        clear_cache()
        semantic_cache.clear()
        context_cache.clear()

        opt_res   = run_optimized(claim, score, decision, feats, scenario_config=cfg)
        unopt_res = run_unoptimized(claim, score, decision, feats, scenario_config=cfg)

        icon = "⛔" if opt_res.loop_detected else ("⚠️ " if opt_res.retry_count > 0 else "✅")
        print(f"     {icon} Optimized:   tok={opt_res.total_tokens:,}  "
              f"retries={opt_res.retry_count}  loop={opt_res.loop_detected}  "
              f"t={opt_res.wall_time_ms:.0f}ms")
        print(f"     🐢 Unoptimized: tok={unopt_res.total_tokens:,}  "
              f"retries={unopt_res.retry_count}  loop={unopt_res.loop_detected}  "
              f"t={unopt_res.wall_time_ms:.0f}ms")

        ff = FailureFinding(
            scenario_name=sc["name"],
            claim_id=sc["claim_id"],
            description=sc["description"],
            opt=opt_res,
            unopt=unopt_res,
        )
        results.append(ff)
        _print_failure_comparison(ff)

        print(f"\n  👁  What to watch:")
        for point in sc["what_to_watch"]:
            print(f"       • {point}")

    return results