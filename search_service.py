"""Tavily 搜索 + 大模型提炼：输出 150 字以内的地段商业情报摘要。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_LL_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_BASE_URL",
    "OPENAI_BASE_URL",
    "DEEPSEEK_MODEL",
    "OPENAI_MODEL",
    "TAVILY_API_KEY",
    "AMAP_API_KEY",
)


def _show_missing_api_key_error_once() -> None:
    """仅在 Streamlit 页面环境提示一次缺少 API 密钥。"""
    try:
        import streamlit as st

        flag = "_missing_api_key_error_shown"
        if st.session_state.get(flag):
            return
        st.session_state[flag] = True
        st.error("未配置 API 密钥，请检查后台设置")
    except Exception:
        # CLI / 非 Streamlit 运行时忽略
        return


def _bootstrap_env() -> None:
    """
    Streamlit Cloud 优先：先从 st.secrets 读取；本地兜底：再从 .env / .env.local 读取。

    说明：仅 `os.environ` 时，在资源管理器双击运行、或未继承你「终端里 export」
    的进程里会读不到密钥；用 .env 或与 Streamlit 一致的 secrets 可避免。
    """
    root = Path(__file__).resolve().parent
    # 1) 优先 st.secrets（Streamlit Cloud 标准方式）
    try:
        import streamlit as st

        sec = getattr(st, "secrets", None)
        if sec is not None:
            for key in _LL_ENV_KEYS:
                try:
                    if key not in sec:
                        continue
                except Exception:
                    continue
                val = sec[key]
                if val is None or not str(val).strip():
                    continue
                os.environ[key] = str(val).strip()
    except Exception:
        pass

    # 2) 兜底 .env / .env.local（仅填充仍缺失的键）
    try:
        from dotenv import load_dotenv

        for name in (".env", ".env.local"):
            p = root / name
            if p.is_file():
                load_dotenv(p, override=False)
    except ImportError:
        pass


_bootstrap_env()

from search_cache_db import get_cached_intelligence, set_cached_intelligence
from usage_context import add_llm_chars

# Tavily Key 仅从环境变量 / .env / st.secrets 读取，不再内置默认值。
_INSTALL_HINT = (
    "未安装 tavily-python，请执行：python -m pip install tavily-python"
)
_NO_TAVILY_HINT = (
    "未配置 Tavily 密钥：请设置 TAVILY_API_KEY，或在 .env/.streamlit/secrets.toml 中提供。"
)
_FALLBACK = "暂未搜索到周边实时情报"
_NO_LLM_HINT = (
    "未配置大模型密钥：请在运行 Streamlit 的同一环境中设置 DEEPSEEK_API_KEY 或 "
    "OPENAI_API_KEY；或在项目根目录放置 .env（见 .env.example），亦可使用 "
    ".streamlit/secrets.toml 中的同名键。"
)
_DEEPSEEK_BASE_DEFAULT = "https://api.deepseek.com/v1"
_DEEPSEEK_MODEL_DEFAULT = "deepseek-chat"
_TOP_K = 5
_MAX_RAW_PER_HIT = 600
_MAX_RAW_FOR_LLM = 12000
_MAX_OUTPUT_CHARS = 150
_CACHE_HIT_SUFFIX = "（来自本地缓存）"

# 所有经 llm_chat 的调用（情报摘要、拓店/风控 Agent 等）统一注入
_GLOBAL_BRAND_LLM_CONTEXT = (
    "你现在是在为「柠季手打柠檬茶」做决策。品牌定位：年轻、快节奏、高坪效。"
    "核心选址逻辑：[租金中低、商圈活跃]。"
)


def _messages_with_brand_context(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """在对话首条 system 前合并全局品牌背景；若无 system 则插入一条。"""
    out: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "")
        out.append({"role": role, "content": content})
    if not out:
        return [{"role": "system", "content": _GLOBAL_BRAND_LLM_CONTEXT}]
    if out[0].get("role") == "system":
        merged = (out[0].get("content") or "").strip()
        out[0] = {
            "role": "system",
            "content": _GLOBAL_BRAND_LLM_CONTEXT
            + ("\n\n" + merged if merged else ""),
        }
    else:
        out.insert(0, {"role": "system", "content": _GLOBAL_BRAND_LLM_CONTEXT})
    return out


_ANALYST_PROMPT = """你是一位资深商业分析师。下面是一段针对「{address}」进行网络搜索后得到的原始摘录（可能杂乱、有噪音、重复），请仅依据其中可查证或可合理推断的信息作答，不得编造具体数据、日期或新闻标题。

【原始素材】
{raw}

任务：将素材提炼为一段连续中文结论，必须体现下列维度（若某维度素材完全无法支撑，该维度用「待核实」简述即可）：
1）该地段消费水平或客群大致层次；
2）周边茶饮或同类门店的竞争密集程度；
3）潜在负面风险（如修路、撤铺、人流下滑、倒闭潮、竞争过激等）。

硬性约束：输出一段纯文本，不要使用 Markdown、不要使用分点序号；总长度严格不超过 {max_chars} 个汉字（含标点）。"""


def _api_key() -> str:
    return (os.environ.get("TAVILY_API_KEY") or "").strip()


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()  # type: ignore[no-any-return]
    return dict(response)


def _gather_tavily_raw(address: str) -> str:
    try:
        from tavily import TavilyClient
    except ImportError:
        raise RuntimeError("tavily_missing") from None

    key = _api_key()
    if not key:
        _show_missing_api_key_error_once()
        raise RuntimeError("tavily_key_missing")

    query = f"{address} 周边茶饮店分布及商圈人流实时评价"
    client = TavilyClient(key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=_TOP_K,
    )
    data = _response_to_dict(response)
    results = data.get("results") or []
    if not isinstance(results, list):
        return ""

    chunks: list[str] = []
    for i, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        text = (item.get("content") or item.get("snippet") or "").strip()
        blob = f"{title}\n{text}" if title else text
        blob = " ".join(blob.split())
        if len(blob) > _MAX_RAW_PER_HIT:
            blob = blob[: _MAX_RAW_PER_HIT - 1] + "…"
        if blob:
            chunks.append(f"---来源{i}---\n{blob}")
    return "\n\n".join(chunks)


def _clip_text(s: str, max_chars: int) -> str:
    s = " ".join(s.strip().split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _resolve_llm_auth() -> tuple[str, str, str] | None:
    """返回 (api_key, base_url_without_trailing_slash, model)。"""
    ds_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    oa_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if ds_key:
        base = (
            os.environ.get("DEEPSEEK_BASE_URL") or _DEEPSEEK_BASE_DEFAULT
        ).rstrip("/")
        model = (
            os.environ.get("DEEPSEEK_MODEL") or _DEEPSEEK_MODEL_DEFAULT
        ).strip()
        return ds_key, base, model
    if oa_key:
        base = (
            os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        model = (os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()
        return oa_key, base, model
    return None


def llm_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str | None:
    """
    OpenAI 兼容 Chat Completions，支持 system/user 多轮。
    供 agents 等多智能体场景复用；统一在首条 system 中注入柠季品牌全局背景。
    """
    auth = _resolve_llm_auth()
    if not auth:
        _show_missing_api_key_error_once()
        return None
    key, base, model = auth
    url = f"{base}/chat/completions"
    try:
        import httpx
    except ImportError:
        return None

    msgs = _messages_with_brand_context(messages)
    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        prompt_chars = sum(
            len(m.get("content") or "") for m in msgs if isinstance(m, dict)
        )
        add_llm_chars(prompt_chars + len(text))
        return text
    except Exception:
        return None


def _llm_openai_compatible(user_prompt: str) -> str | None:
    """单轮 user 消息，供情报摘要使用。"""
    return llm_chat(
        [{"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=512,
    )


def _summarize_with_llm(address: str, raw: str) -> str:
    prompt = _ANALYST_PROMPT.format(
        address=address,
        raw=raw[:_MAX_RAW_FOR_LLM],
        max_chars=_MAX_OUTPUT_CHARS,
    )
    text = _llm_openai_compatible(prompt)
    if text is None:
        return _NO_LLM_HINT
    return _clip_text(text, _MAX_OUTPUT_CHARS)


def get_real_world_context(address: str) -> str:
    """
    先查 SQLite 表 search_cache：24 小时内同一地址命中则直接返回摘要并附带「来自本地缓存」。
    未命中则 Tavily + 大模型生成摘要，写入缓存（不缓存安装提示、无密钥提示等无效结果）。

    DeepSeek：DEEPSEEK_API_KEY；可选 DEEPSEEK_BASE_URL、DEEPSEEK_MODEL。
    OpenAI 兼容：OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL。
    """
    addr = (address or "").strip()
    if not addr:
        return _FALLBACK

    cached = get_cached_intelligence(addr)
    if cached is not None:
        return f"{cached}{_CACHE_HIT_SUFFIX}"

    try:
        raw = _gather_tavily_raw(addr)
    except RuntimeError as e:
        if str(e) == "tavily_missing":
            return _INSTALL_HINT
        if str(e) == "tavily_key_missing":
            return _NO_TAVILY_HINT
        raise
    except Exception:
        return _FALLBACK

    if not raw.strip():
        return _FALLBACK

    intel = _summarize_with_llm(addr, raw)
    if intel not in (_NO_LLM_HINT, _INSTALL_HINT, _FALLBACK):
        set_cached_intelligence(addr, intel)
    return intel
