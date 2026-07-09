"""
demo.py – Stress-test entry point: Optimized vs Unoptimized routes.

Run:
    python demo.py

What this does:
  1. Picks 4 claims (C001, C003, C004, C006)
  2. For each claim runs 2 rounds:
       Round 1 → XGBoost simulator returns random score R1
                  both routes run with SAME R1
       Round 2 → XGBoost simulator returns NEW random score R2
                  both routes run with SAME R2
  3. Prints side-by-side comparison per round
  4. Shows cross-round delta (how each route responds to score change)
  5. Writes full findings to RESULTS.md

Routes compared:

  OPTIMIZED ROUTE (5 optimization layers):
    · Phase 1: 3 parallel fetch subagents (ThreadPoolExecutor)
    · Phase 2: 3 parallel analysis subagents (cascade from Phase 1)
    · Layer 1: pre-extract relevant doc sections  (50-70% content ↓)
    · Layer 2: concise JSON prompts               (~60% token ↓)
    · Layer 3: LRU content-hash cache             (0 tokens on hit)
    · Layer 4: semantic similarity cache          (0 tokens on ~sim hit)
    · Layer 5: context prefix cache               (90% discount on prefix)

  UNOPTIMIZED ROUTE (zero optimizations):
    · Sequential: patient → records → history → guidelines → LLM calls
    · Full raw document content in every prompt (no pre-extraction)
    · Verbose open-ended prompts (3-6× more tokens)
    · Zero caching – fresh API call every time
    · Monolithic brief re-sends ALL raw document content
"""

from __future__ import annotations
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.insert(0, os.path.dirname(__file__))

from scenarios.stress_test import run_stress_test

HEADER = """
================================================================================
  HEALTHCARE CLAIMS – OPTIMIZED vs UNOPTIMIZED ROUTE STRESS TEST
================================================================================

  Two routes, same inputs:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  OPTIMIZED ROUTE                                                        │
  │  Phase 1: 3 parallel fetch subagents (ThreadPoolExecutor)              │
  │  Phase 2: 3 parallel analysis subagents (cascading from Phase 1)       │
  │  Layer 1: pre-extract relevant sections  → 50-70% content reduction    │
  │  Layer 2: concise structured JSON prompts → ~60% token reduction       │
  │  Layer 3: LRU content-hash cache          → 0 tokens on exact hit      │
  │  Layer 4: semantic similarity cache       → 0 tokens on similar hit    │
  │  Layer 5: context prefix cache            → 90% discount on prefix     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  UNOPTIMIZED ROUTE                                                      │
  │  Sequential data fetching (blocks on each DB call)                     │
  │  Full raw document content sent to LLM (no pre-extraction)             │
  │  Verbose open-ended prompts (3-6× more tokens)                         │
  │  Zero caching at any layer (fresh API call every time)                 │
  │  Monolithic brief re-sends ALL raw document content                    │
  └─────────────────────────────────────────────────────────────────────────┘

  XGBoost simulator: returns random.random() (0-1) to stress-test all
  decision bands (AUTO_APPROVE / NEEDS_REVIEW / AUTO_DENY).

  Test: 4 claims × 2 rounds = 8 comparisons total.
  Findings written to RESULTS.md after each run.

================================================================================
"""


def main() -> None:
    print(HEADER)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("⚠️  WARNING: OPENAI_API_KEY not set or is a placeholder.")
        print("   All LLM calls are REAL (no simulation). Without a valid key:")
        print("   • Input token counts are accurate — tiktoken measures prompt size locally.")
        print("   • Output token counts are wrong — the LLM returns [LLM_ERROR:...] strings,")
        print("     so output tokens reflect only that tiny error string, not real completions.")
        print("   • Token-saving RATIOS between routes are still meaningful")
        print("     (driven by prompt size, not output).")
        print("   • Analyst briefs will contain error text, not real clinical summaries.")
        print("   Set OPENAI_API_KEY in a .env file for real LLM output and accurate costs.\n")

    run_stress_test()


if __name__ == "__main__":
    main()

