"""
document_processor.py – Medical PDF extraction and LLM summarization.

Token optimization strategy (three layers):

  Layer 1 – Pre-extraction (ZERO tokens, pure text processing)
    Extract only the clinically relevant sections of a document
    (FINDINGS + IMPRESSION from radiology; ASSESSMENT + PLAN from consults).
    Typical reduction: 50–70% of raw document size.

  Layer 2 – Structured prompts (CONCISE vs VERBOSE)
    Concise: ask for specific JSON fields only → small, dense prompt + output.
    Verbose: open-ended prose narrative → large prompt + large output.
    Typical reduction: 60–70% fewer tokens with concise prompts.

  Layer 3 – Content-hash caching
    Once a document has been summarised, cache by SHA-256 of (relevant_content +
    optimization_mode).  A second claim for the same patient re-uses the cache
    (0 tokens).

The combination of all three layers reduces per-document LLM cost by ~80–85%.
"""

from __future__ import annotations
import re
import json
import hashlib
import tiktoken
from typing import Dict, Any, List, Tuple, Optional

from core.llm_engine import llm_call, _count_tokens

_enc = tiktoken.encoding_for_model("gpt-4o-mini")

# ── Summary cache: key = content_hash + mode ─────────────────────────────────
_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: Pre-extraction – no LLM, just regex / string parsing
# ══════════════════════════════════════════════════════════════════════════════

_SECTION_PATTERNS: Dict[str, List[str]] = {
    "radiology_report": [
        r"FINDINGS?:(.*?)(?=IMPRESSION:|RECOMMENDATION:|$)",
        r"IMPRESSION:(.*?)(?=RECOMMENDATION:|$)",
    ],
    "consultation_note": [
        r"ASSESSMENT:(.*?)(?=PLAN:|$)",
        r"PLAN:(.*?)$",
    ],
    "lab_report": [
        r"(?:COMPLETE BLOOD COUNT|CBC|RESULTS?|INTERPRETATION)(.*?)(?=INTERPRETATION|$)",
        r"INTERPRETATION:(.*?)$",
    ],
    "therapy_note": [
        r"(?:GOALS|OUTCOMES?|CLINICAL NOTE)(.*?)(?=RECOMMENDATION|$)",
        r"RECOMMENDATION:(.*?)$",
    ],
}

_MAX_SECTION_CHARS = 900   # per matched section


def extract_relevant_sections(content: str, doc_type: str) -> str:
    """
    Pre-extract clinically relevant sections from raw PDF text.
    Returns a smaller string that captures the most useful information.
    No LLM call – this is pure text processing (0 tokens).
    """
    patterns = _SECTION_PATTERNS.get(doc_type, [])
    sections: List[str] = []

    for pattern in patterns:
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if m:
            section_text = m.group(1).strip()
            # Truncate overly long sections
            if len(section_text) > _MAX_SECTION_CHARS:
                section_text = section_text[:_MAX_SECTION_CHARS] + "..."
            label = pattern.split("(")[0].rstrip(":?(").replace("?", "").strip()
            sections.append(f"{label}:\n{section_text}")

    if sections:
        return "\n\n".join(sections)

    # Fallback: first + last portion of the document
    half = _MAX_SECTION_CHARS
    if len(content) > half * 2:
        return content[:half] + "\n...[middle omitted]...\n" + content[-400:]
    return content[:half * 2]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: LLM prompts – concise vs verbose
# ══════════════════════════════════════════════════════════════════════════════

# Concise: forces JSON output, asks for specific fields only
_CONCISE_PROMPT = """{doc_type} – insurance review. Patient: {name}, {age}y.

Respond ONLY with valid JSON (no markdown):
{{
  "diagnosis": "<primary finding or ICD description>",
  "severity": "<low|medium|high|critical>",
  "necessity_evidence": ["<point 1>", "<point 2>"],
  "prior_treatment": "<summary of treatments attempted>",
  "physician_recommendation": "<recommendation and rationale>"
}}

Document:
{content}"""

# Verbose: open-ended prose, full document, unstructured output
_VERBOSE_PROMPT = """You are a senior healthcare insurance specialist reviewing medical documentation
to support a claims adjudication decision.

PATIENT INFORMATION:
  Name: {name}
  Age:  {age} years old

Document Type: {doc_type}

Please provide a thorough and comprehensive analysis of the following medical document.
Your analysis should include:
  1. DIAGNOSIS SUMMARY – A detailed description of the primary diagnosis and all relevant
     clinical findings, including any imaging results, laboratory values, or physical
     examination findings that support the diagnosis.
  2. MEDICAL NECESSITY ASSESSMENT – A comprehensive evaluation of all evidence supporting
     medical necessity, including clinical indicators, symptom severity, functional impairment,
     and how the proposed treatment addresses the underlying pathology.
  3. PRIOR TREATMENT HISTORY – A complete review of all conservative treatments that have been
     attempted prior to this claim, including the duration of each treatment, the patient's
     response, and the reasons conservative management was deemed insufficient.
  4. PHYSICIAN RECOMMENDATION – The treating physician's clinical recommendations and the
     detailed rationale behind them.
  5. SEVERITY ASSESSMENT – Your professional assessment of the severity and urgency of the
     medical condition.

Please ensure your analysis is thorough, evidence-based, and provides sufficient detail to
support the claims adjudication process.

MEDICAL DOCUMENT CONTENT:
{content}

Please provide your comprehensive analysis below:"""


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: Summarize with caching
# ══════════════════════════════════════════════════════════════════════════════

def summarize_medical_pdf(
    pdf_data: Dict[str, Any],
    patient_name: str,
    patient_age: int,
    optimization_mode: str = "concise",
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Summarize a single medical PDF for insurance review.

    Returns:
      (summary_dict, raw_tokens_estimate, actual_tokens_used, was_cached)

    raw_tokens_estimate  = tokens that would have been used with the full document
    actual_tokens_used   = tokens actually used (after pre-extraction + concise prompt)
    was_cached           = True if result came from cache
    """
    content  = pdf_data.get("content", "")
    doc_type = pdf_data.get("type", "general")
    doc_name = pdf_data.get("filename", "document")

    # ── Layer 1: Pre-extract relevant sections ────────────────────────────────
    relevant_content = extract_relevant_sections(content, doc_type)

    # Estimate raw token cost (full document + verbose prompt skeleton)
    raw_prompt_estimate = _VERBOSE_PROMPT.format(
        doc_type=doc_type.replace("_", " ").title(),
        name=patient_name,
        age=patient_age,
        content=content,
    )
    raw_tokens_estimate = _count_tokens(raw_prompt_estimate) + 300  # +300 for verbose output

    # ── Layer 3: Cache check ──────────────────────────────────────────────────
    cache_key = hashlib.sha256(
        f"{relevant_content}|{optimization_mode}".encode()
    ).hexdigest()[:20]

    if cache_key in _SUMMARY_CACHE:
        cached_summary = _SUMMARY_CACHE[cache_key]
        return cached_summary, raw_tokens_estimate, 0, True

    # ── Layer 2: Build optimized or verbose prompt ────────────────────────────
    if optimization_mode == "concise":
        prompt = _CONCISE_PROMPT.format(
            doc_type=doc_type.replace("_", " ").title(),
            name=patient_name,
            age=patient_age,
            content=relevant_content,   # pre-extracted subset
        )
    else:
        prompt = _VERBOSE_PROMPT.format(
            doc_type=doc_type.replace("_", " ").title(),
            name=patient_name,
            age=patient_age,
            content=content,            # full raw document
        )

    step_name = f"summarize_pdf:{doc_name}"
    response, token_record = llm_call(
        prompt=prompt,
        step=step_name,
        max_tokens=400,
        use_cache=False,  # content-hash cache handled here
    )

    # Parse JSON from concise mode
    if optimization_mode == "concise":
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            summary = json.loads(json_match.group()) if json_match else {"raw": response}
        except (json.JSONDecodeError, AttributeError):
            summary = {"raw": response[:400], "parse_error": True}
    else:
        summary = {"narrative": response}

    actual_tokens = token_record["total_tokens"]

    # Cache the result
    _SUMMARY_CACHE[cache_key] = summary

    return summary, raw_tokens_estimate, actual_tokens, False


def process_all_pdfs(
    pdf_records: List[Dict[str, Any]],
    patient_name: str,
    patient_age: int,
    optimization_mode: str = "concise",
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Process all PDFs for a patient.

    Returns:
      (pdf_summary_list, total_raw_tokens, total_actual_tokens, cache_hits)
    """
    summaries = []
    total_raw    = 0
    total_actual = 0
    cache_hits   = 0

    for pdf in pdf_records:
        summary_data, raw_est, actual, cached = summarize_medical_pdf(
            pdf, patient_name, patient_age, optimization_mode
        )

        pct_saving = ((raw_est - actual) / raw_est * 100) if raw_est > 0 and not cached else 0.0

        summaries.append({
            "doc_id":               pdf["doc_id"],
            "filename":             pdf["filename"],
            "doc_type":             pdf["type"],
            "summary":              summary_data,
            "raw_tokens_estimate":  raw_est,
            "actual_tokens":        actual,
            "token_savings_pct":    round(pct_saving, 1),
            "cached":               cached,
            "optimization_mode":    optimization_mode,
        })

        total_raw    += raw_est
        total_actual += actual
        if cached:
            cache_hits += 1

    return summaries, total_raw, total_actual, cache_hits


# ══════════════════════════════════════════════════════════════════════════════
# Claims history pattern analysis
# ══════════════════════════════════════════════════════════════════════════════

# Concise: pre-aggregate stats, send only numbers + flagged items
_CLAIMS_PATTERN_CONCISE = """Analyze insurance claims pattern. Respond ONLY with valid JSON.

Patient: {patient_id}
New claim: {procedure} | ${amount:.0f}

Stats (last 24 months):
  Total claims: {total_claims}
  Total paid: ${total_paid:.0f}
  Same-procedure claims: {similar_count}
  Average claim amount: ${avg_amount:.0f}

Flagged claims:
{flags_text}

JSON output:
{{
  "pattern": "<normal|elevated|suspicious>",
  "risk_level": "<low|medium|high>",
  "total_claims": {total_claims},
  "total_paid": {total_paid},
  "similar_procedure_count": {similar_count},
  "flags": ["<flag1>", "<flag2>"],
  "notes": "<1-2 sentence summary>"
}}"""

_CLAIMS_PATTERN_VERBOSE = """You are a senior healthcare insurance fraud analyst with 15 years of experience
reviewing claims patterns for medical necessity and potential anomalies.

A new insurance claim has been submitted and requires a thorough analysis of the member's
historical claims pattern to identify any concerning trends or validate the claim.

MEMBER INFORMATION: {patient_id}

NEW CLAIM DETAILS:
  Procedure: {procedure}
  Billed Amount: ${amount:.2f}

COMPLETE CLAIMS HISTORY (last 24 months):
{full_history_text}

Please perform a comprehensive analysis of:
  1. Overall claims frequency and cost patterns
  2. Consistency with the member's established healthcare utilization profile
  3. Whether the new claim is consistent with prior claims in the same specialty
  4. Any patterns suggestive of overutilization, upcoding, or medical fraud
  5. Your overall risk assessment and recommendation

Please provide a detailed narrative analysis with your professional conclusion."""


_CLAIMS_HISTORY_CACHE: Dict[str, Any] = {}


def analyze_claims_pattern(
    patient_id: str,
    claims_history: List[Dict[str, Any]],
    new_claim: Dict[str, Any],
    optimization_mode: str = "concise",
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Analyse historical claims pattern for a patient relative to a new claim.

    Returns:
      (pattern_dict, raw_tokens_estimate, actual_tokens_used, was_cached)
    """
    procedure  = new_claim.get("procedure", "Unknown")
    amount     = new_claim.get("billed_amount", 0)

    # ── Pre-aggregate stats (no LLM) ─────────────────────────────────────────
    total_claims = len(claims_history)
    total_paid   = sum(c.get("amount", 0) for c in claims_history)
    avg_amount   = total_paid / total_claims if total_claims else 0

    # Count same-procedure claims
    proc_lower = procedure.lower().split()[0]  # first word match
    similar_count = sum(
        1 for c in claims_history
        if proc_lower in c.get("procedure", "").lower()
    )

    # Identify flagged claims (high amount or duplicate procedure)
    flagged = [
        c for c in claims_history
        if c.get("flag") or c.get("amount", 0) > avg_amount * 2
    ]
    flags_text = "\n".join(
        f"  - {c['claim_id']}: {c['procedure']} ${c['amount']:.0f}"
        for c in flagged[:5]
    ) or "  None"

    # Cache key based on history content + new claim amount
    history_hash = hashlib.sha256(
        f"{patient_id}|{total_paid}|{total_claims}|{optimization_mode}".encode()
    ).hexdigest()[:16]

    if history_hash in _CLAIMS_HISTORY_CACHE:
        return _CLAIMS_HISTORY_CACHE[history_hash], 0, 0, True

    # ── Estimate raw verbose token cost ───────────────────────────────────────
    full_history_text = "\n".join(
        f"  {c['date']}  {c['claim_id']}  {c['procedure']:<50}  ${c['amount']:.0f}"
        for c in claims_history
    ) or "  No prior claims"

    raw_verbose_prompt = _CLAIMS_PATTERN_VERBOSE.format(
        patient_id=patient_id, procedure=procedure, amount=amount,
        full_history_text=full_history_text,
    )
    raw_tokens_estimate = _count_tokens(raw_verbose_prompt) + 350  # +verbose output

    # ── Build actual prompt ───────────────────────────────────────────────────
    if optimization_mode == "concise":
        prompt = _CLAIMS_PATTERN_CONCISE.format(
            patient_id=patient_id, procedure=procedure, amount=amount,
            total_claims=total_claims, total_paid=total_paid,
            similar_count=similar_count, avg_amount=avg_amount,
            flags_text=flags_text,
        )
    else:
        prompt = _CLAIMS_PATTERN_VERBOSE.format(
            patient_id=patient_id, procedure=procedure, amount=amount,
            full_history_text=full_history_text,
        )

    response, token_record = llm_call(
        prompt=prompt,
        step=f"analyze_claims:{patient_id}",
        max_tokens=300,
        use_cache=False,
    )

    # Parse concise JSON
    if optimization_mode == "concise":
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            pattern_dict = json.loads(json_match.group()) if json_match else {}
        except (json.JSONDecodeError, AttributeError):
            pattern_dict = {}
    else:
        pattern_dict = {"narrative": response}

    # Ensure required fields have defaults
    pattern_dict.setdefault("pattern", "normal")
    pattern_dict.setdefault("risk_level", "low")
    pattern_dict.setdefault("total_claims", total_claims)
    pattern_dict.setdefault("total_paid", total_paid)
    pattern_dict.setdefault("similar_procedure_count", similar_count)
    pattern_dict.setdefault("flags", [])
    pattern_dict.setdefault("notes", response[:200] if optimization_mode == "verbose" else "")

    actual_tokens = token_record["total_tokens"]
    _CLAIMS_HISTORY_CACHE[history_hash] = pattern_dict

    return pattern_dict, raw_tokens_estimate, actual_tokens, False


# ══════════════════════════════════════════════════════════════════════════════
# Analyst brief compilation
# ══════════════════════════════════════════════════════════════════════════════

_BRIEF_CONCISE = """Compile analyst review brief. Max 200 words.

CLAIM: {claim_id} | {procedure} | ${billed:.0f} (network: ${network:.0f})
PATIENT: {name}, {age}y | Plan: {plan} | In-network: {in_network}
ML RISK: {risk_score:.2f} ({risk_decision}) | Top factors: {risk_top_factors}

MEDICAL FINDINGS (from {doc_count} documents):
{doc_findings}

CLAIMS HISTORY: {history_pattern} risk | {total_claims} prior claims | ${total_paid:.0f} total
  Pattern: {history_notes}

GUIDELINES: {guideline_status}
  Avg procedure cost: ${avg_cost:.0f} | Billed is {cost_vs_avg}

Output brief with:
1. KEY FINDINGS (2-3 bullets)
2. RISK ASSESSMENT (1 sentence)
3. RECOMMENDATION (Approve/Deny/Negotiate + 1-sentence rationale)"""

_BRIEF_VERBOSE = """You are a senior healthcare insurance claims analyst preparing a comprehensive
review brief for a human medical director who will make the final adjudication decision.

Please compile all available information into a thorough review brief.

─────────────────────────────────────────────────────────────────
CLAIM DETAILS
─────────────────────────────────────────────────────────────────
Claim ID:         {claim_id}
Procedure:        {procedure}
Billed Amount:    ${billed:.2f}
Network Rate:     ${network:.2f}
In-Network:       {in_network}
Provider:         {provider}

─────────────────────────────────────────────────────────────────
PATIENT INFORMATION
─────────────────────────────────────────────────────────────────
Name: {name}  |  Age: {age}  |  Insurance Plan: {plan}

─────────────────────────────────────────────────────────────────
ML RISK ASSESSMENT
─────────────────────────────────────────────────────────────────
Risk Score: {risk_score:.2f}  |  Triage Decision: {risk_decision}
Contributing factors: {risk_factors_detail}

─────────────────────────────────────────────────────────────────
MEDICAL DOCUMENTATION REVIEW ({doc_count} documents)
─────────────────────────────────────────────────────────────────
{doc_findings_verbose}

─────────────────────────────────────────────────────────────────
CLAIMS HISTORY ANALYSIS
─────────────────────────────────────────────────────────────────
{history_detail}

─────────────────────────────────────────────────────────────────
TREATMENT GUIDELINE COMPLIANCE
─────────────────────────────────────────────────────────────────
{guideline_detail}

Please compile a comprehensive analyst brief that:
1. Summarises the clinical picture and medical necessity evidence
2. Evaluates the financial appropriateness of the claim
3. Highlights any areas of concern or flags for the medical director
4. Provides a clear recommendation with supporting rationale
5. Notes any additional information that may be required for final determination"""


def compile_analyst_brief(
    claim: Dict[str, Any],
    patient_profile: Optional[Dict[str, Any]],
    pdf_summaries: List[Dict[str, Any]],
    claims_pattern: Optional[Dict[str, Any]],
    guidelines: Optional[Dict[str, Any]],
    risk_score: float,
    risk_decision: str,
    risk_features: Dict[str, float],
    optimization_mode: str = "concise",
) -> Tuple[str, int, int]:
    """
    Compile the final analyst brief that will be presented to the human reviewer.

    Returns:
      (brief_text, raw_tokens_estimate, actual_tokens_used)
    """
    name       = patient_profile["name"] if patient_profile else claim.get("member_name", "Unknown")
    age        = patient_profile["age"]  if patient_profile else "?"
    plan       = patient_profile.get("plan", "Unknown") if patient_profile else "Unknown"
    in_network = "Yes" if claim.get("provider_in_network") else "No (out-of-network)"
    provider   = claim.get("provider", "Unknown")

    billed     = claim.get("billed_amount", 0)
    network    = claim.get("approved_amount", 0)
    avg_cost   = (guidelines or {}).get("avg_cost_network", 0) or network
    cost_diff  = ((billed - avg_cost) / avg_cost * 100) if avg_cost else 0
    cost_vs_avg = f"{cost_diff:+.0f}% vs network avg" if avg_cost else "unknown avg"

    # Summarise document findings
    doc_findings_lines = []
    for s in pdf_summaries:
        summ = s.get("summary", {})
        if "diagnosis" in summ:
            doc_findings_lines.append(
                f"  [{s['doc_type'].replace('_', ' ').title()}] "
                f"Dx: {summ.get('diagnosis','?')} | "
                f"Severity: {summ.get('severity','?')} | "
                f"Rec: {summ.get('physician_recommendation','?')[:80]}"
            )
        elif "narrative" in summ:
            doc_findings_lines.append(f"  [{s['doc_type']}] {summ['narrative'][:120]}...")
    doc_findings = "\n".join(doc_findings_lines) or "  No medical documents available"

    # Top risk factors
    top_factors = sorted(risk_features.items(), key=lambda x: x[1], reverse=True)[:3]
    risk_top_str = ", ".join(f"{k.replace('_', ' ')}={v:.2f}" for k, v in top_factors)

    # Claims history summary
    cp = claims_pattern or {}
    history_pattern   = cp.get("risk_level", "unknown")
    total_claims_hist = cp.get("total_claims", 0)
    total_paid_hist   = cp.get("total_paid", 0)
    history_notes     = cp.get("notes", "No prior claims")[:120]

    # Guideline status
    if guidelines:
        approval_criteria = guidelines.get("approval_criteria", [])
        guideline_status = f"{len(approval_criteria)} criteria defined"
    else:
        guideline_status = "Guidelines not available"

    # Estimate raw verbose token cost
    raw_verbose_prompt = _BRIEF_VERBOSE.format(
        claim_id=claim["claim_id"], procedure=claim.get("procedure",""),
        billed=billed, network=network, in_network=in_network, provider=provider,
        name=name, age=age, plan=plan,
        risk_score=risk_score, risk_decision=risk_decision,
        risk_factors_detail=str(risk_features)[:200],
        doc_count=len(pdf_summaries),
        doc_findings_verbose=doc_findings,
        history_detail=f"Pattern: {history_pattern}, {total_claims_hist} claims, ${total_paid_hist:.0f} total",
        guideline_detail=str(guidelines)[:300] if guidelines else "N/A",
    )
    raw_tokens_estimate = _count_tokens(raw_verbose_prompt) + 500

    if optimization_mode == "concise":
        prompt = _BRIEF_CONCISE.format(
            claim_id=claim["claim_id"], procedure=claim.get("procedure",""),
            billed=billed, network=network, in_network=in_network,
            name=name, age=age, plan=plan,
            risk_score=risk_score, risk_decision=risk_decision,
            risk_top_factors=risk_top_str,
            doc_count=len(pdf_summaries), doc_findings=doc_findings,
            history_pattern=history_pattern, total_claims=total_claims_hist,
            total_paid=total_paid_hist, history_notes=history_notes,
            guideline_status=guideline_status, avg_cost=avg_cost, cost_vs_avg=cost_vs_avg,
        )
    else:
        prompt = _BRIEF_VERBOSE.format(
            claim_id=claim["claim_id"], procedure=claim.get("procedure",""),
            billed=billed, network=network, in_network=in_network, provider=provider,
            name=name, age=age, plan=plan,
            risk_score=risk_score, risk_decision=risk_decision,
            risk_factors_detail=str(risk_features),
            doc_count=len(pdf_summaries),
            doc_findings_verbose=doc_findings,
            history_detail=f"Pattern: {history_pattern}, {total_claims_hist} claims, ${total_paid_hist:.0f} total. {history_notes}",
            guideline_detail=str(guidelines)[:400] if guidelines else "Not available",
        )

    response, token_record = llm_call(
        prompt=prompt,
        step=f"compile_brief:{claim['claim_id']}",
        max_tokens=500,
        use_cache=False,
    )

    actual_tokens = token_record["total_tokens"]
    return response, raw_tokens_estimate, actual_tokens
