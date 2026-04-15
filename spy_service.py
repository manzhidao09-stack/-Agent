"""Day 19 全网情报侦察：多平台搜索聚合 + LLM 结构化清洗。

说明：美团/点评/抖音/京东等无官方开放接口时，通过 Tavily 检索公开网页摘录，
再用已配置的 DeepSeek/OpenAI（search_service.llm_chat）抽取为结构化表格数据。
结果不等同于各平台实时排行或官方 API，仅供选址研判辅助。
"""

from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import quote_plus
from typing import Any

import pandas as pd

from search_service import llm_chat

_TAVILY_THROTTLE_SEC = 1.05
_MAX_RAW_FOR_LLM = 14_000
_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
    "Mobile/15E148 Safari/604.1"
)


def _tavily_api_key() -> str:
    return (os.environ.get("TAVILY_API_KEY") or "").strip()


def _llm_configured() -> bool:
    return bool(
        (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )


def _deepseek_configured() -> bool:
    return bool((os.environ.get("DEEPSEEK_API_KEY") or "").strip())


def _tavily_import_ok() -> bool:
    try:
        import tavily  # noqa: F401
    except ImportError:
        return False
    return True


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[no-any-return]
    if hasattr(response, "dict"):
        return response.dict()  # type: ignore[no-any-return]
    return dict(response)


def _tavily_search(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    key = _tavily_api_key()
    if not key:
        return []
    try:
        from tavily import TavilyClient
    except ImportError:
        return []
    try:
        client = TavilyClient(key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
        )
        data = _response_to_dict(response)
        results = data.get("results") or []
        return results if isinstance(results, list) else []
    except Exception:
        return []


def _format_hits(platform_label: str, results: list[dict[str, Any]]) -> str:
    lines: list[str] = [f"=== {platform_label} 搜索摘录 ==="]
    for i, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        text = (item.get("content") or item.get("snippet") or "").strip()
        url = (item.get("url") or "").strip()
        blob = f"[{i}] title={title}\nurl={url}\nbody={text[:900]}"
        lines.append(blob)
    return "\n".join(lines)


def gather_cross_platform_intel(address: str) -> str:
    """
    围绕同一地址，从「美团/点评/抖音/京东」视角构造检索词，抓取公开网页摘录并拼接。
    """
    addr = (address or "").strip()
    if not addr:
        return ""
    plan: list[tuple[str, str]] = [
        (
            "美团茶饮",
            f"{addr} 美团 茶饮 奶茶 品牌 销量 月售 评分 人均",
        ),
        (
            "大众点评茶饮",
            f"{addr} 大众点评 奶茶 茶饮 排行 评分 人均 月销",
        ),
        (
            "抖音团购茶饮",
            f"{addr} 抖音 团购 茶饮 奶茶 热销 评分 人均",
        ),
        (
            "京东小时达闪购",
            f"{addr} 京东 小时达 闪购 茶饮 配送 价格带 评分",
        ),
    ]
    chunks: list[str] = []
    for label, q in plan:
        hits = _tavily_search(q, max_results=5)
        chunks.append(_format_hits(label, hits))
        time.sleep(_TAVILY_THROTTLE_SEC)
    return "\n\n".join(chunks)


def _firecrawl_search_pages(address: str) -> list[tuple[str, str]]:
    q = quote_plus(f"{address} 茶饮 柠檬茶 评分 月销")
    return [
        ("大众点评H5", f"https://m.dianping.com/searchshopall?query={q}"),
        ("美团H5", f"https://i.meituan.com/s/{q}"),
        ("抖音搜索", f"https://www.douyin.com/search/{quote_plus(address + ' 柠檬茶')}"),
    ]


def _firecrawl_mobile_scrape(address: str) -> str:
    """
    使用 Firecrawl 抓取移动端搜索页文本，附加手机 UA 降低反爬命中率。
    返回拼接后的粗文本，供 DeepSeek 降噪抽取。
    """
    key = (os.environ.get("FIRECRAWL_API_KEY") or "").strip()
    if not key:
        return ""
    try:
        from firecrawl import FirecrawlApp
    except Exception:
        return ""

    app = FirecrawlApp(api_key=key)
    chunks: list[str] = []
    for label, url in _firecrawl_search_pages(address):
        try:
            # 兼容 firecrawl-py 不同版本参数结构
            response = app.scrape_url(
                url,
                params={
                    "formats": ["markdown"],
                    "onlyMainContent": False,
                    "headers": {"User-Agent": _MOBILE_UA},
                },
            )
        except TypeError:
            try:
                response = app.scrape_url(
                    url,
                    formats=["markdown"],
                    only_main_content=False,
                    headers={"User-Agent": _MOBILE_UA},
                )
            except Exception:
                continue
        except Exception:
            continue
        payload = _response_to_dict(response)
        md = (
            str(payload.get("markdown") or "").strip()
            or str((payload.get("data") or {}).get("markdown") or "").strip()
            or str(payload.get("content") or "").strip()
        )
        if not md:
            continue
        chunks.append(f"=== {label} Firecrawl抓取 ===\nurl={url}\n{md[:7000]}")
        time.sleep(0.4)
    return "\n\n".join(chunks)


def _deepseek_extract_competitors(address: str, noisy_text: str) -> list[dict[str, Any]]:
    """
    让 DeepSeek 对混乱网页文本做降噪抽取：
    1) 前 3 名茶饮店
    2) 所有柠檬茶店
    """
    if not _deepseek_configured():
        return []
    raw = (noisy_text or "").strip()
    if not raw:
        return []
    sys_msg = (
        "你是网页情报降噪专家。"
        "这是一段从点评网页抓取的混乱文本，请帮我从中提取出前 3 名茶饮店和所有柠檬茶店的店名、月销和评分。"
        "输出必须是合法 JSON 对象，不要 Markdown，不要解释性文字。"
        '严格格式：{"items":[{"品牌":"","评分":"","月销":"","平台":"","类别":""}]}。'
        "类别只能是“前3茶饮”或“柠檬茶”，字段缺失填“不详”。"
    )
    user_msg = f"点位：{address}\n\n抓取文本：\n{raw[:12000]}"
    txt = llm_chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=1800,
        json_object=False,
    )
    if not txt:
        return []
    rows = _extract_json_array(txt)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "品牌": str(row.get("品牌", "不详") or "不详"),
                "评分": str(row.get("评分", "不详") or "不详"),
                "人均消费": "不详",
                "月销": str(row.get("月销", "不详") or "不详"),
                "核心差评点": "不详",
                "平台": str(row.get("平台", "大众点评") or "大众点评"),
            }
        )
    return out


def _normalize_smart_quotes(s: str) -> str:
    return (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _unwrap_markdown_code(s: str) -> str:
    """去掉最外层 Markdown 代码块（可嵌套多次）。"""
    t = (s or "").strip()
    while True:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
        if not m:
            break
        inner = m.group(1).strip()
        if not inner:
            t = re.sub(r"```(?:json)?\s*```", "", t, count=1, flags=re.IGNORECASE)
            continue
        t = inner
    return t.strip()


def _repair_trailing_commas(s: str) -> str:
    """修复 LLM 常见的尾随逗号。"""
    s2 = re.sub(r",\s*]", "]", s)
    return re.sub(r",\s*}", "}", s2)


def _slice_balanced(s: str, open_c: str, close_c: str) -> str | None:
    """从首个 open_c 起截取与之配平的 close_c 子串（忽略字符串内的括号）。"""
    start = s.find(open_c)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _rows_from_parsed(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in (
            "items",
            "data",
            "rows",
            "list",
            "results",
            "competitors",
            "情报",
        ):
            inner = data.get(k)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
        if any(data.get(f) is not None for f in ("品牌", "评分", "平台")):
            return [data]
    return []


def _try_json_load_rows(s: str) -> list[dict[str, Any]]:
    try:
        return _rows_from_parsed(json.loads(s))
    except json.JSONDecodeError:
        return []


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    从大模型回复中解析竞品行：支持 {"items":[...]}、顶层数组、代码块、前后废话、尾随逗号。
    """
    s = _normalize_smart_quotes((text or "").strip())
    if not s:
        return []
    s = s.lstrip("\ufeff")
    s = _unwrap_markdown_code(s)

    candidates: list[str] = []
    if s:
        candidates.append(s)
    obj = _slice_balanced(s, "{", "}")
    if obj and obj not in candidates:
        candidates.append(obj)
    arr = _slice_balanced(s, "[", "]")
    if arr and arr not in candidates:
        candidates.append(arr)

    for cand in candidates:
        for variant in (cand, _repair_trailing_commas(cand)):
            rows = _try_json_load_rows(variant)
            if rows:
                return rows
    return []


def structure_spy_intel(
    address: str, raw_snippets: str
) -> tuple[list[dict[str, Any]], str]:
    """
    使用 DeepSeek/OpenAI（经 search_service.llm_chat）将杂乱摘录清洗为结构化 JSON 数组。
    每项字段：品牌、评分、人均消费、月销、核心差评点、平台。
    第二个返回值："" 表示成功；否则为失败原因码（llm_empty / parse_fail）。
    """
    raw = (raw_snippets or "").strip()
    if not raw:
        return [], "no_raw"
    sys_msg = (
        "你是商业情报清洗专家。请仅依据输入摘录抽取信息，禁止编造摘录中不存在的品牌、"
        "评分、销量、价格。无法从摘录支持时填「不详」。"
        "你必须只输出一个合法 JSON 对象（不要 Markdown、不要解释性前后文），"
        '格式严格为：{"items":[{"品牌":"","评分":"","人均消费":"","月销":"","核心差评点":"","平台":""}, ...]} 。'
        "items 为数组，每项 6 个键名必须与上文一致（中文键）。"
        "平台只能取：美团、大众点评、抖音、京东 之一。"
        "优先整理与茶饮/咖啡/奶茶同类竞品相关的条目，最多 20 条；若无可用信息则输出 {\"items\":[]} 。"
        "说明：本任务要求输出 JSON，请确保可被 json.loads 解析。"
    )
    user_msg = f"点位：{address}\n\n【原始摘录】\n{raw[:_MAX_RAW_FOR_LLM]}"
    text = llm_chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=2400,
        json_object=True,
    )
    if not text:
        return [], "llm_empty"
    rows = _extract_json_array(text)
    if not rows:
        return [], "parse_fail"
    return rows, ""


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = ["品牌", "评分", "人均消费", "月销", "核心差评点", "平台"]
    if not rows:
        return pd.DataFrame(columns=cols)
    norm: list[dict[str, Any]] = []
    for r in rows:
        norm.append(
            {
                "品牌": str(r.get("品牌", "不详") or "不详"),
                "评分": str(r.get("评分", "不详") or "不详"),
                "人均消费": str(r.get("人均消费", "不详") or "不详"),
                "月销": str(r.get("月销", "不详") or "不详"),
                "核心差评点": str(r.get("核心差评点", "不详") or "不详"),
                "平台": str(r.get("平台", "不详") or "不详"),
            }
        )
    return pd.DataFrame(norm)


def build_competitor_intel_table(address: str) -> tuple[pd.DataFrame, str]:
    """
    返回 (DataFrame, 说明)。
    DataFrame 为空时，说明字符串给出可操作的失败原因。
    """
    cols = ["品牌", "评分", "人均消费", "月销", "核心差评点", "平台"]
    empty = pd.DataFrame(columns=cols)
    addr = (address or "").strip()
    if not addr:
        return empty, "地址为空，无法检索竞品情报。"

    if not _tavily_api_key():
        return (
            empty,
            "未检测到 **TAVILY_API_KEY**：请在 Streamlit Cloud Secrets 或本地 `.streamlit/secrets.toml` "
            "中配置，并重新部署/刷新页面。",
        )
    if not _tavily_import_ok():
        return (
            empty,
            "未安装 **tavily-python**：请在运行环境执行 `python -m pip install tavily-python`。",
        )
    if not _llm_configured():
        return (
            empty,
            "未检测到 **DEEPSEEK_API_KEY** 或 **OPENAI_API_KEY**：竞品表需要 LLM 将摘录清洗为结构化 JSON。",
        )

    raw = gather_cross_platform_intel(addr)
    firecrawl_raw = _firecrawl_mobile_scrape(addr)
    if firecrawl_raw:
        raw = f"{raw}\n\n{firecrawl_raw}".strip()
    excerpt_lines = sum(1 for line in raw.splitlines() if line.strip().startswith("["))
    if excerpt_lines == 0 and not firecrawl_raw:
        return (
            empty,
            "Tavily 未返回有效网页摘录（检索无命中、配额用尽或网络异常）。可尝试把地址写得更具体（含区/路），"
            "或稍后再试。",
        )

    rows, err = structure_spy_intel(addr, raw)
    if not rows:
        if err == "llm_empty":
            return (
                empty,
                "大模型返回为空：请检查 **DEEPSEEK_API_KEY / OPENAI_API_KEY** 是否有效、"
                "额度是否充足，或网络是否能访问对应 API。",
            )
        if err == "parse_fail":
            ds_rows = _deepseek_extract_competitors(addr, raw)
            if ds_rows:
                return _rows_to_dataframe(ds_rows), ""
            return (
                empty,
                "大模型有返回，但无法解析为结构化 JSON。"
                "已尝试 Firecrawl + DeepSeek 降噪兜底仍失败，请确认 **FIRECRAWL_API_KEY** 与 "
                "**DEEPSEEK_API_KEY** 已配置，并稍后重试。",
            )
        return empty, "结构化清洗失败（原因未知）。"
    return _rows_to_dataframe(rows), ""


def build_competitor_intel_dataframe(address: str) -> pd.DataFrame:
    """返回可直接用于 st.dataframe 的竞品情报表（无说明文本）。"""
    df, _ = build_competitor_intel_table(address)
    return df


def fetch_platform_intelligence(address: str) -> pd.DataFrame:
    """对外别名：抓取并结构化多平台情报。"""
    return build_competitor_intel_dataframe(address)
