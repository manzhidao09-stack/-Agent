"""单次评估周期内累计 LLM 请求/响应字符数（粗估成本用）。"""

from __future__ import annotations

from contextvars import ContextVar

_ctx_chars: ContextVar[int] = ContextVar("llm_usage_chars", default=0)


def reset_llm_usage() -> None:
    _ctx_chars.set(0)


def add_llm_chars(n: int) -> None:
    if n <= 0:
        return
    _ctx_chars.set(_ctx_chars.get(0) + n)


def get_llm_usage_chars() -> int:
    return max(0, _ctx_chars.get(0))
