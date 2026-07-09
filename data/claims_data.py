"""
Claims submitted for processing.
ML risk model triage:
  - AUTO_APPROVE  → processed immediately, no LLM involved
  - AUTO_DENY     → immediate denial, no LLM involved
  - NEEDS_REVIEW  → LangGraph agent prepares full analyst package
"""

from __future__ import annotations
from typing import Dict, Any, Optional

CLAIMS: Dict[str, Dict[str, Any]] = {

    # ── C001: NEEDS_REVIEW – knee surgery, amount above network average ──────
    "C001": {
        "claim_id": "C001",
        "patient_id": "P001",
        "member_name": "John Doe",
        "date": "2026-07-01",
        "provider": "Metro Orthopedic Associates",
        "provider_in_network": False,   # out-of-network flag
        "procedure": "Knee Arthroscopy with Meniscectomy",
        "procedure_key": "knee_arthroscopy",
        "diagnosis": "Medial Meniscus Tear + Knee Osteoarthritis",
        "icd10": ["M23.201", "M17.11"],
        "billed_amount": 12500.00,
        "approved_amount": 8900.00,    # network rate
        "member_avg_cost": 5400.00,
        "diagnosis_risk_score": 0.45,
        "provider_reliability_score": 0.80,
        "procedure_code_valid": True,
        "description": "Arthroscopic meniscectomy and chondroplasty, right knee",
        "status": "pending",
    },

    # ── C002: AUTO_APPROVE – routine annual wellness ─────────────────────────
    "C002": {
        "claim_id": "C002",
        "patient_id": "P002",
        "member_name": "Mary Johnson",
        "date": "2026-07-05",
        "provider": "Family Health Clinic",
        "provider_in_network": True,
        "procedure": "Annual Wellness Examination",
        "procedure_key": "annual_wellness",
        "diagnosis": "Preventive Care / Health Maintenance",
        "icd10": ["Z00.00"],
        "billed_amount": 380.00,
        "approved_amount": 350.00,
        "member_avg_cost": 350.00,
        "diagnosis_risk_score": 0.05,
        "provider_reliability_score": 0.98,
        "procedure_code_valid": True,
        "description": "Annual wellness visit with preventive care counselling",
        "status": "pending",
    },

    # ── C003: NEEDS_REVIEW – chemotherapy cycle, high amount ─────────────────
    "C003": {
        "claim_id": "C003",
        "patient_id": "P003",
        "member_name": "Michael Chen",
        "date": "2026-07-06",
        "provider": "City Cancer Center",
        "provider_in_network": True,
        "procedure": "R-CHOP Chemotherapy Cycle 3",
        "procedure_key": "chemotherapy_rchop",
        "diagnosis": "Non-Hodgkin Lymphoma – DLBCL Stage IIIA",
        "icd10": ["C83.30"],
        "billed_amount": 28400.00,
        "approved_amount": 24000.00,
        "member_avg_cost": 24000.00,   # consistent with prior cycles
        "diagnosis_risk_score": 0.90,  # high-cost oncology always needs review
        "provider_reliability_score": 0.95,
        "procedure_code_valid": True,
        "description": "Cycle 3 of 6 R-CHOP: Rituximab, Cyclophosphamide, Doxorubicin, Vincristine, Prednisone",
        "status": "pending",
    },

    # ── C004: NEEDS_REVIEW – complex spine surgery, highest cost ─────────────
    "C004": {
        "claim_id": "C004",
        "patient_id": "P004",
        "member_name": "Sarah Williams",
        "date": "2026-07-07",
        "provider": "Regional Spine Center",
        "provider_in_network": True,
        "procedure": "Lumbar Spinal Fusion L4-S1",
        "procedure_key": "lumbar_fusion",
        "diagnosis": "Lumbar DDD with Spondylolisthesis + Radiculopathy",
        "icd10": ["M47.816", "M43.16", "M54.4"],
        "billed_amount": 67200.00,
        "approved_amount": 52000.00,
        "member_avg_cost": 8500.00,    # 7.9× member average → high flag
        "diagnosis_risk_score": 0.65,
        "provider_reliability_score": 0.90,
        "procedure_code_valid": True,
        "description": "L4-L5 and L5-S1 posterior lumbar interbody fusion with pedicle screw instrumentation",
        "status": "pending",
    },

    # ── C005: AUTO_APPROVE – emergency appendectomy ──────────────────────────
    "C005": {
        "claim_id": "C005",
        "patient_id": "P005",
        "member_name": "Robert Davis",
        "date": "2026-07-08",
        "provider": "City General Hospital – Emergency",
        "provider_in_network": True,
        "procedure": "Emergency Appendectomy",
        "procedure_key": "appendectomy_emergency",
        "diagnosis": "Acute Appendicitis",
        "icd10": ["K37"],
        "billed_amount": 8900.00,
        "approved_amount": 9200.00,   # within expected range
        "member_avg_cost": 9200.00,
        "diagnosis_risk_score": 0.15,  # emergency – retrospective auth
        "provider_reliability_score": 0.97,
        "procedure_code_valid": True,
        "description": "Emergency laparoscopic appendectomy for acute appendicitis",
        "status": "pending",
    },

    # ── C006: NEEDS_REVIEW – same patient as C001, follow-up (cache-hit scenario)
    "C006": {
        "claim_id": "C006",
        "patient_id": "P001",          # same patient as C001
        "member_name": "John Doe",
        "date": "2026-07-09",
        "provider": "Metro Orthopedic Associates",
        "provider_in_network": False,
        "procedure": "Post-Operative Complication Visit",
        "procedure_key": "knee_arthroscopy",
        "diagnosis": "Post-surgical joint effusion – right knee",
        "icd10": ["M79.821", "Z96.651"],
        "billed_amount": 9800.00,
        "approved_amount": 5500.00,
        "member_avg_cost": 5400.00,
        "diagnosis_risk_score": 0.65,
        "provider_reliability_score": 0.80,
        "procedure_code_valid": True,
        "description": "Follow-up for post-operative right knee effusion; aspiration and steroid injection",
        "status": "pending",
    },
}


def get_claim(claim_id: str) -> Optional[Dict[str, Any]]:
    return CLAIMS.get(claim_id)


def get_needs_review_claims() -> list:
    """Returns claims that are candidates for NEEDS_REVIEW after ML triage."""
    return [c for c in CLAIMS.values() if c["status"] == "pending"]
