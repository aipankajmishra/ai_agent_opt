"""
demo.py – Main entry point for the Healthcare Claims HITL Token Optimisation demo.

Run:
    python demo.py
"""

from __future__ import annotations
import sys
import os

# Force UTF-8 output on Windows so Unicode prints work correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from scenarios.run_scenarios import run_all_scenarios

HEADER = """
================================================================================
  HEALTHCARE CLAIMS - HITL REVIEW AGENT + TOKEN OPTIMISATION
================================================================================

  Workflow:
    1. Claim arrives
    2. ML Risk Model scores it in <1 ms ($0 LLM cost)
       AUTO_APPROVE / AUTO_DENY  --> done immediately
       NEEDS_REVIEW              --> LangGraph agent kicks in
    3. LangGraph Agent (7 nodes) prepares analyst package:
       * Fetches patient profile          (with retry on DB failure)
       * Fetches medical documents/PDFs   (with retry on DB failure)
       * Summarises PDFs with LLM         (token-optimised)
       * Fetches claims history           (with retry on DB failure)
       * Analyses claims pattern with LLM (token-optimised)
       * Fetches treatment guidelines     (loop detection)
       * Compiles analyst brief with LLM  (token-optimised)
    4. Human analyst reviews the package and makes the final decision

  Token Optimisation Techniques:
    * Pre-extract relevant PDF sections    (50-70% content reduction before LLM)
    * Structured JSON output prompts       (60% less output tokens vs prose)
    * Content-hash LRU caching             (0 tokens on repeat documents)
    * Retry with graceful degradation      (no infinite retry loops)
    * Loop detection via node_visit_counts (abort stuck nodes)

  5 Scenarios:
    1. Standard workflow      (C001 - knee surgery)
    2. DB retry recovery      (C003 - chemo cycle)
    3. Multi-document case    (C004 - spine fusion, 3 PDFs)
    4. Cache hit              (C006 - same patient as Scenario 1)
    5. Loop detection         (guidelines DB returns empty repeatedly)

================================================================================
"""


def main() -> None:
    print(HEADER)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("⚠️  WARNING: OPENAI_API_KEY not set or is a placeholder.")
        print("   LLM calls will return error strings and token counts will be minimal.")
        print("   Set a real API key in .env for meaningful results.\n")

    run_all_scenarios()


if __name__ == "__main__":
    main()
