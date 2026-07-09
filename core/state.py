"""
LangGraph state definition for the Healthcare Claims HITL Review Pipeline.

Workflow:
  Claim → ML Risk Score → NEEDS_REVIEW → LangGraph Agent → Analyst Brief → Human Decision

The agent prepares everything the human analyst needs:
  - Patient profile (from patient DB)
  - Medical document summaries (from PDFs via document store)
  - Claims history pattern (from claims DB)
  - Treatment guideline check (from guidelines DB)
  - Compiled analyst brief (from LLM)

Token optimization happens in the summarization and brief compilation steps.
"""

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any

# ── Retry / loop-detection constants ─────────────────────────────────────────
MAX_RETRIES = 3          # max retry attempts per fetch node
MAX_NODE_VISITS = MAX_RETRIES + 1  # total visits allowed before forced continue


# ── Sub-type definitions ──────────────────────────────────────────────────────

class PDFSummary(TypedDict):
    doc_id: str
    filename: str
    doc_type: str
    summary: Dict[str, Any]   # structured extraction result
    raw_tokens_estimate: int  # tokens if we had sent full doc
    actual_tokens: int        # tokens actually used (optimized)
    token_savings_pct: float  # (raw - actual) / raw * 100
    cached: bool
    optimization_mode: str    # "concise" | "verbose"


class ClaimsPattern(TypedDict):
    pattern: str              # "normal" | "elevated" | "suspicious"
    risk_level: str           # "low" | "medium" | "high"
    total_claims: int
    total_paid: float
    similar_procedure_count: int
    flags: List[str]
    notes: str


class StepTokenUsage(TypedDict):
    step: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cached: bool


# ── Main agent state ──────────────────────────────────────────────────────────

class ClaimsReviewState(TypedDict):

    # ── Input ─────────────────────────────────────────────────────────────────
    claim_id: str
    patient_id: str
    claim_data: Dict[str, Any]
    optimization_mode: str          # "concise" | "verbose"
    scenario_config: Dict[str, Any] # controls failure simulation per scenario

    # ── ML Decision (from ClaimRiskModel, not from LLM) ───────────────────────
    risk_score: float
    risk_decision: str              # AUTO_APPROVE | AUTO_DENY | NEEDS_REVIEW
    risk_features: Dict[str, float]

    # ── Gathered data (filled progressively by agent nodes) ───────────────────
    patient_profile: Optional[Dict[str, Any]]        # fetched from patient DB
    raw_medical_pdfs: List[Dict[str, Any]]            # raw PDF records from doc store
    pdf_summaries: List[PDFSummary]                   # LLM-summarized (optimized)
    claims_history: List[Dict[str, Any]]              # raw claims records from DB
    claims_pattern: Optional[ClaimsPattern]           # LLM-analyzed claims pattern
    treatment_guidelines: Optional[Dict[str, Any]]    # from guidelines DB

    # ── Agent flow tracking ───────────────────────────────────────────────────
    current_step: str
    node_visit_counts: Dict[str, int]   # key: node_name, value: visit count → loop detection
    data_completeness: Dict[str, bool]  # key: data_name, value: was it gathered?
    errors: List[str]                   # collected errors (non-fatal)
    agent_log: List[str]               # human-readable pipeline activity log

    # ── Output for human analyst ──────────────────────────────────────────────
    analyst_brief: Optional[str]

    # ── Token tracking (per step) ─────────────────────────────────────────────
    token_usage_by_step: Dict[str, StepTokenUsage]
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
