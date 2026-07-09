"""
Unoptimized Claims Review Agent – ZERO optimizations (baseline).

This is the "naive" implementation that represents what you'd write without
thinking about cost. Every anti-pattern is intentional for comparison:

  ✗ Sequential data fetching – one DB call blocks the next
  ✗ Full raw document content sent to LLM – no pre-extraction
  ✗ Open-ended verbose prompts – large input + large output
  ✗ No caching at any layer – every run makes fresh API calls
  ✗ Monolithic final brief – re-sends ALL document content again
  ✗ No retry on DB failure – one attempt, then proceeds with None/empty
  ✗ No loop detection – if guidelines always returns empty it just gives up
    after the first miss (no graceful degradation, just missing data silently)

Token overhead vs optimized route: typically 3× – 6× more tokens per run.
Used as the denominator in the stress-test savings calculation.
"""

from __future__ import annotations
import time
import os
import tiktoken
import openai
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Import the shared RouteResult dataclass from the optimized agent
from agents.optimized_agent import RouteResult

MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_PRICE_IN       = 0.00015
_PRICE_OUT      = 0.0006
_enc            = tiktoken.encoding_for_model("gpt-4o-mini")


# ══════════════════════════════════════════════════════════════════════════════
# Direct (uncached) LLM call
# ══════════════════════════════════════════════════════════════════════════════

def _raw_llm(prompt: str, max_tokens: int = 800) -> Tuple[str, int, int]:
    """
    Call the LLM directly, bypassing ALL caches.
    Returns (response_text, input_tokens, output_tokens).
    """
    try:
        _proxy_backup: Dict[str, str] = {}
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            if k in os.environ:
                _proxy_backup[k] = os.environ.pop(k)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        response = resp.choices[0].message.content or ""
    except Exception as exc:
        response = f"[LLM_ERROR: {exc}]"
    finally:
        for k, v in _proxy_backup.items():
            os.environ[k] = v

    inp = len(_enc.encode(prompt))
    out = len(_enc.encode(response))
    return response, inp, out


# ══════════════════════════════════════════════════════════════════════════════
# Verbose prompt templates (open-ended, large context, no structure)
# ══════════════════════════════════════════════════════════════════════════════

_VERBOSE_DOC = """\
You are a senior healthcare insurance specialist reviewing medical documentation.

PATIENT INFORMATION:
  Name: {name}
  Age:  {age} years old

DOCUMENT TYPE:     {doc_type}
DOCUMENT FILENAME: {filename}

Please provide a thorough and comprehensive analysis of the following medical document.
Your analysis must include:
  1. All clinical findings in detail
  2. Complete diagnosis description with supporting evidence
  3. Medical necessity evaluation
  4. Prior treatment history and patient response
  5. Physician recommendations and rationale
  6. Your professional severity assessment

COMPLETE DOCUMENT CONTENT:
{content}

Please provide your comprehensive analysis below:"""

_VERBOSE_CLAIMS = """\
You are a senior healthcare insurance fraud analyst with 15 years of experience.

MEMBER INFORMATION: {patient_id}

NEW CLAIM:
  Procedure: {procedure}
  Billed Amount: ${amount:.2f}

COMPLETE CLAIMS HISTORY ({total_claims} prior claims):
{full_history}

Please perform a comprehensive analysis covering:
  1. Overall claims frequency and cost patterns over time
  2. Comparison with member's historical utilization profile
  3. Consistency of new claim with prior claims in same specialty
  4. Any patterns suggestive of overutilization, upcoding, or fraud
  5. Overall risk assessment and recommendation

Provide a detailed narrative with your professional conclusion:"""

_VERBOSE_BRIEF = """\
You are a senior healthcare insurance claims analyst preparing a review brief
for the medical director who will make the final adjudication decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLAIM DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claim ID:       {claim_id}
Procedure:      {procedure}
Billed Amount:  ${billed:.2f}
Network Rate:   ${network:.2f}
Provider:       {provider}
In-Network:     {in_network}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {name} | Age: {age} | Plan: {plan}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Risk Score:  {risk_score:.4f}
Decision:    {risk_decision}
Factors:     {risk_features}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL MEDICAL DOCUMENTATION ({doc_count} documents)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{all_doc_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL CLAIMS HISTORY ({total_claims} prior claims – ${total_paid:.2f} total paid)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{full_history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREATMENT GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{guidelines}

Please write a comprehensive analyst brief that:
  1. Summarises the clinical picture and medical necessity evidence in detail
  2. Evaluates the financial appropriateness of the billed amounts
  3. Highlights every area of concern for the medical director
  4. Provides a clear recommendation with full supporting rationale
  5. Notes all additional information required for final determination

ANALYST BRIEF:"""


# ══════════════════════════════════════════════════════════════════════════════
# Public entry-point
# ══════════════════════════════════════════════════════════════════════════════

def run_unoptimized(
    claim: Dict[str, Any],
    risk_score: float,
    risk_decision: str,
    risk_features: Dict[str, float],
    scenario_config: Optional[Dict[str, Any]] = None,
) -> RouteResult:
    """
    Run the full claims review pipeline WITHOUT any optimizations.

    scenario_config is accepted so the stress test can inject the same DB-failure
    settings as the optimized route, but this agent deliberately does NOT retry
    or detect loops – it is the naive baseline that shows what happens without
    those protections.

    Failure behaviour (intentionally bad):
      - One fetch attempt per resource. If it fails, data is None/empty.
      - No retry loop. The pipeline continues with incomplete data silently.
      - Guidelines: one attempt only; if empty, brief is compiled without them.
        No loop detection – a stuck node would never be caught.

    Steps (all sequential):
      1. Fetch patient profile       (1 attempt – no retry)
      2. Fetch medical records       (1 attempt – no retry)
      3. Fetch claims history        (1 attempt – no retry)
      4. Fetch treatment guidelines  (1 attempt – no loop detection)
      5. Summarise each document with LLM (verbose, full content, uncached)
      6. Analyse claims pattern with LLM  (verbose, full history, uncached)
      7. Compile analyst brief with LLM   (verbose, ALL raw content re-sent, uncached)
    """
    from tools import data_fetcher as db

    cfg        = scenario_config or {}
    t_start    = time.perf_counter()
    total_inp  = 0
    total_out  = 0
    api_calls  = 0

    result = RouteResult(
        route="unoptimized",
        claim_id=claim["claim_id"],
        risk_score=risk_score,
        risk_decision=risk_decision,
    )

    # ML triage: same as optimized – no LLM if auto-decided
    if risk_decision != "NEEDS_REVIEW":
        result.wall_time_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return result

    patient_id = claim["patient_id"]
    proc_key   = claim.get("procedure_key", "")

    # Extract failure counts from scenario_config – we use attempt=1 always
    # (no retry), so if fail_times >= 1 the fetch will fail and we get None.
    fail_pat  = cfg.get("fail_patient_fetch_times",  0)
    fail_rec  = cfg.get("fail_records_fetch_times",  0)
    fail_hist = cfg.get("fail_history_fetch_times",  0)
    fail_guide = cfg.get("fail_guidelines_fetch_times", 0)
    sim_loop   = cfg.get("simulate_guidelines_loop", False)

    # ── Step 1: Patient profile – ONE attempt, no retry ───────────────────────
    t0 = time.perf_counter()
    ok, profile = db.fetch_patient_profile(patient_id, attempt=1, fail_times=fail_pat)
    if not ok:
        result.failure_log.append("[patient] ❌ fetch failed – no retry (unoptimized)")
    result.step_times_ms["fetch_patient_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Step 2: Medical records – ONE attempt, no retry ───────────────────────
    t0 = time.perf_counter()
    ok, records_raw = db.fetch_medical_records(patient_id, attempt=1, fail_times=fail_rec)
    records = records_raw or []
    if not ok:
        result.failure_log.append("[records] ❌ fetch failed – no retry (unoptimized)")
    result.step_times_ms["fetch_records_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Step 3: Claims history – ONE attempt, no retry ────────────────────────
    t0 = time.perf_counter()
    ok, history_raw = db.fetch_claims_history(patient_id, attempt=1, fail_times=fail_hist)
    history = history_raw or []
    if not ok:
        result.failure_log.append("[history] ❌ fetch failed – no retry (unoptimized)")
    result.step_times_ms["fetch_history_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Step 4: Guidelines – ONE attempt, no loop detection ───────────────────
    # simulate_guidelines_loop means the DB always returns empty; the unoptimized
    # agent has no loop detection so it simply proceeds without guidelines.
    t0 = time.perf_counter()
    ok, guidelines = db.fetch_treatment_guidelines(
        proc_key, attempt=1, fail_times=fail_guide, simulate_loop=sim_loop
    )
    if not ok or (sim_loop and not guidelines):
        result.failure_log.append(
            "[guidelines] ❌ empty/failed – no loop detection (unoptimized) – brief will lack guidelines"
        )
    result.step_times_ms["fetch_guidelines_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    name = (profile or {}).get("name", claim.get("member_name", "Unknown"))
    age  = (profile or {}).get("age", "?")
    plan = (profile or {}).get("plan", "Unknown")

    # ── Step 5: LLM summarise each doc (verbose, full content, NO cache) ─────
    doc_summaries: List[str] = []
    for doc in records:
        t0 = time.perf_counter()
        prompt = _VERBOSE_DOC.format(
            name=name, age=age,
            doc_type=doc.get("type", "document").replace("_", " ").title(),
            filename=doc.get("filename", "unknown"),
            content=doc.get("content", ""),       # ← FULL raw content, no extraction
        )
        response, inp, out = _raw_llm(prompt, max_tokens=600)
        total_inp  += inp
        total_out  += out
        api_calls  += 1
        doc_summaries.append(response)
        result.step_times_ms[f"doc_{doc.get('doc_id','?')}_ms"] = round(
            (time.perf_counter() - t0) * 1000, 1
        )

    # ── Step 6: LLM claims pattern (verbose, full history, NO cache) ──────────
    t0 = time.perf_counter()
    total_paid_h = sum(c.get("amount", 0) for c in history)
    full_hist_text = "\n".join(
        f"  {c['date']}  {c['claim_id']}  {c['procedure']:<50}  ${c['amount']:.0f}"
        for c in history
    ) or "  No prior claims"

    claims_prompt = _VERBOSE_CLAIMS.format(
        patient_id=patient_id,
        procedure=claim.get("procedure", "Unknown"),
        amount=claim.get("billed_amount", 0),
        total_claims=len(history),
        full_history=full_hist_text,
    )
    claims_resp, inp, out = _raw_llm(claims_prompt, max_tokens=400)
    total_inp  += inp
    total_out  += out
    api_calls  += 1
    result.step_times_ms["analyze_claims_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Step 7: LLM compile brief (verbose, RE-SENDS ALL raw content, NO cache)
    t0 = time.perf_counter()
    all_doc_content = ""
    for i, doc in enumerate(records, 1):
        all_doc_content += (
            f"\n\n{'='*60}\n"
            f"Document {i}: {doc.get('filename', '?')} "
            f"[{doc.get('type','?').replace('_',' ').title()}]\n"
            f"{'='*60}\n"
            + doc.get("content", "")   # ← full raw content re-sent
        )
    if not all_doc_content:
        all_doc_content = "No medical documents available."

    brief_prompt = _VERBOSE_BRIEF.format(
        claim_id=claim["claim_id"],
        procedure=claim.get("procedure", ""),
        billed=claim.get("billed_amount", 0),
        network=claim.get("approved_amount", 0),
        provider=claim.get("provider", "Unknown"),
        in_network="Yes" if claim.get("provider_in_network") else "No (out-of-network)",
        name=name, age=age, plan=plan,
        risk_score=risk_score,
        risk_decision=risk_decision,
        risk_features=str(risk_features),
        doc_count=len(records),
        all_doc_content=all_doc_content,
        total_claims=len(history),
        total_paid=total_paid_h,
        full_history=full_hist_text,
        guidelines=str(guidelines)[:600] if guidelines else "Not available",
    )
    brief_resp, inp, out = _raw_llm(brief_prompt, max_tokens=800)
    total_inp  += inp
    total_out  += out
    api_calls  += 1
    result.step_times_ms["compile_brief_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Finalise metrics ──────────────────────────────────────────────────────
    result.analyst_brief        = brief_resp
    result.llm_api_calls        = api_calls
    result.total_input_tokens   = total_inp
    result.total_output_tokens  = total_out
    result.total_tokens         = total_inp + total_out
    result.estimated_cost_usd   = (
        total_inp / 1000 * _PRICE_IN + total_out / 1000 * _PRICE_OUT
    )
    result.lru_cache_hits       = 0    # intentionally zero
    result.semantic_cache_hits  = 0    # intentionally zero
    result.context_tokens_saved = 0    # intentionally zero
    result.wall_time_ms         = round((time.perf_counter() - t_start) * 1000, 1)

    return result
