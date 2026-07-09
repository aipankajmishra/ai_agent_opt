"""
data_fetcher.py – Simulated DB / document-store fetch functions with retry support.

Each function accepts an `attempt` integer and a `fail_times` integer (from scenario_config).
If attempt <= fail_times  → simulate a transient failure (DB timeout / network error).
Otherwise               → return real data from the mock data layer.

This lets scenarios control exactly when failures happen without patching global state.
"""

from __future__ import annotations
import time
from typing import Optional, List, Dict, Any, Tuple

from data.medical_records import (
    PATIENT_PROFILES,
    MEDICAL_PDF_RECORDS,
    CLAIMS_HISTORY,
    TREATMENT_GUIDELINES,
    PROCEDURE_GUIDELINE_MAP,
)


# ── Simulated DB latency ──────────────────────────────────────────────────────
_DB_LATENCY_OK_SEC   = 0.06   # normal round-trip
_DB_LATENCY_FAIL_SEC = 0.15   # timeout before failure is detected


def _simulate_db_call(success: bool) -> None:
    """Mimic network / DB latency."""
    time.sleep(_DB_LATENCY_FAIL_SEC if not success else _DB_LATENCY_OK_SEC)


# ── Patient profile ───────────────────────────────────────────────────────────

def fetch_patient_profile(
    patient_id: str,
    attempt: int,
    fail_times: int = 0,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Fetch patient profile from patient DB.

    Returns (success, data).
    Simulates transient failure for the first `fail_times` attempts.
    """
    should_fail = attempt <= fail_times
    _simulate_db_call(not should_fail)

    if should_fail:
        return False, None

    profile = PATIENT_PROFILES.get(patient_id)
    if profile:
        return True, dict(profile)

    return False, None


# ── Medical PDF records ───────────────────────────────────────────────────────

def fetch_medical_records(
    patient_id: str,
    attempt: int,
    fail_times: int = 0,
) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    """
    Fetch list of medical document records from document store.

    Returns (success, list_of_pdf_records).
    """
    should_fail = attempt <= fail_times
    _simulate_db_call(not should_fail)

    if should_fail:
        return False, None

    docs = MEDICAL_PDF_RECORDS.get(patient_id, [])
    return True, list(docs)


# ── Claims history ────────────────────────────────────────────────────────────

def fetch_claims_history(
    patient_id: str,
    attempt: int,
    fail_times: int = 0,
) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    """
    Fetch historical claims from claims DB.

    Returns (success, list_of_claims).
    """
    should_fail = attempt <= fail_times
    _simulate_db_call(not should_fail)

    if should_fail:
        return False, None

    history = CLAIMS_HISTORY.get(patient_id, [])
    return True, list(history)


# ── Treatment guidelines ──────────────────────────────────────────────────────

def fetch_treatment_guidelines(
    procedure_key: str,
    attempt: int,
    fail_times: int = 0,
    simulate_loop: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Fetch treatment guidelines for a procedure.

    `simulate_loop=True` makes the function return empty data on every call,
    which triggers the loop-detection path in the LangGraph agent.
    """
    should_fail = attempt <= fail_times
    _simulate_db_call(not should_fail)

    if should_fail:
        return False, None

    if simulate_loop:
        # Return empty dict – agent expects non-empty data and will retry
        return True, {}

    guidelines = TREATMENT_GUIDELINES.get(procedure_key)
    if guidelines:
        return True, dict(guidelines)

    # Fall back to procedure map lookup
    mapped_key = PROCEDURE_GUIDELINE_MAP.get(procedure_key, "")
    if mapped_key:
        guidelines = TREATMENT_GUIDELINES.get(mapped_key)
        if guidelines:
            return True, dict(guidelines)

    return True, {}   # empty but not a hard failure
