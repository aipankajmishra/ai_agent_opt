"""
LLM engine with step-aware token tracking, LRU caching, and cost estimation.
Wraps OpenAI API for all LLM calls in the pipeline.
"""

import os
import hashlib
import tiktoken
import openai
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Pricing per 1K tokens (gpt-4o-mini)
PRICE_INPUT_PER_1K  = 0.00015
PRICE_OUTPUT_PER_1K = 0.0006

_enc = tiktoken.encoding_for_model("gpt-4o-mini")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _cost(input_tok: int, output_tok: int) -> float:
    return (input_tok / 1000) * PRICE_INPUT_PER_1K + (output_tok / 1000) * PRICE_OUTPUT_PER_1K

# ── Global token tracker ──────────────────────────────────────────────────────

class TokenTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost_usd      = 0.0
        self.api_calls           = 0
        self.cache_hits          = 0
        self.step_records: list  = []

    def record(self, step: str, prompt: str, response: str, cached: bool = False) -> Dict[str, Any]:
        inp  = _count_tokens(prompt)
        out  = _count_tokens(response)
        cost = 0.0 if cached else _cost(inp, out)

        if not cached:
            self.total_input_tokens  += inp
            self.total_output_tokens += out
            self.total_cost_usd      += cost
            self.api_calls           += 1
        else:
            self.cache_hits += 1

        rec = {
            "step":               step,
            "prompt_tokens":      inp,
            "completion_tokens":  out,
            "total_tokens":       inp + out,
            "cost_usd":           cost,
            "cached":             cached,
        }
        self.step_records.append(rec)
        return rec

    def summary(self) -> Dict[str, Any]:
        total = self.api_calls + self.cache_hits
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "api_calls":           self.api_calls,
            "cache_hits":          self.cache_hits,
            "cache_hit_rate":      f"{hit_rate:.1f}%",
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens":        self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd":  f"${self.total_cost_usd:.5f}",
        }


tracker = TokenTracker()

# ── Response cache (keyed by prompt hash) ────────────────────────────────────

_response_cache: Dict[str, str] = {}


def llm_call(
    prompt: str,
    step: str = "unknown",
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 600,
    use_cache: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call the LLM with optional caching.
    Returns (response_text, token_record).
    token_record matches StepTokenUsage schema in state.py.
    """
    prompt_hash = hashlib.sha256(f"{prompt}|{model}|{temperature}".encode()).hexdigest()[:16]

    # Check response cache
    if use_cache and prompt_hash in _response_cache:
        response = _response_cache[prompt_hash]
        record = tracker.record(step, prompt, response, cached=True)
        return response, record

    # Call OpenAI
    # Some environments inject HTTP_PROXY / HTTPS_PROXY env vars that cause
    # the newer openai client to fail.  We create the client in a clean scope.
    try:
        import os as _os
        _proxy_backup = {}
        for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            if _k in _os.environ:
                _proxy_backup[_k] = _os.environ.pop(_k)
        client = openai.OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = resp.choices[0].message.content or ""
    except Exception as e:
        response = f"[LLM_ERROR: {e}]"
    finally:
        for _k, _v in _proxy_backup.items():
            _os.environ[_k] = _v

    if use_cache:
        _response_cache[prompt_hash] = response

    record = tracker.record(step, prompt, response, cached=False)
    return response, record


def clear_cache():
    _response_cache.clear()
    tracker.reset()
