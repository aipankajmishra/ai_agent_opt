"""
Context Cache – simulates provider-level prompt prefix caching.

In production APIs:
  - Anthropic context caching: repeated prompt prefixes cost ~10% of normal input price
  - OpenAI prompt caching:     automatic for prompts > 1024 tokens, ~50% discount
  - Google Gemini:             explicit context caching with TTL

We simulate the Anthropic model (90% savings on cached prefix tokens) locally by:
  1. Hashing the "static context" portion of each prompt (patient profile,
     claim metadata, document content that doesn't change within a session)
  2. On first call → full token cost; subsequent calls → 10% cost
  3. Reporting total tokens saved (useful for cost projection)

Where it helps most:
  - Same patient across multiple LLM calls (doc summary + pattern + brief all
    share the same patient context prefix → amortized across 3 calls)
  - Same claim reviewed in both optimized routes across stress-test rounds
"""

from __future__ import annotations
import hashlib
from typing import Dict, Tuple

# Anthropic context caching pricing model: cached tokens cost 10% of normal
CACHE_READ_MULTIPLIER = 0.10


class ContextCache:
    def __init__(self):
        self._seen: Dict[str, int] = {}   # prefix_hash → original token count
        self.reads        = 0
        self.tokens_saved = 0

    def _h(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:20]

    def charge(self, prefix: str, prefix_tokens: int) -> Tuple[int, bool]:
        """
        Calculate effective token cost for a context prefix.

        First encounter  → full cost,  returns (prefix_tokens, False)
        Repeat encounter → 10% cost,   returns (effective_tokens, True)
        """
        h = self._h(prefix)
        if h in self._seen:
            self.reads += 1
            saved = int(prefix_tokens * (1 - CACHE_READ_MULTIPLIER))
            self.tokens_saved += saved
            return max(1, int(prefix_tokens * CACHE_READ_MULTIPLIER)), True
        self._seen[h] = prefix_tokens
        return prefix_tokens, False

    def clear(self) -> None:
        self._seen.clear()
        self.reads        = 0
        self.tokens_saved = 0

    def stats(self) -> Dict:
        return {
            "context_cache_reads":  self.reads,
            "context_tokens_saved": self.tokens_saved,
            "unique_prefixes_seen": len(self._seen),
        }


# ── Module-level singleton used by the optimized agent ───────────────────────
context_cache = ContextCache()
