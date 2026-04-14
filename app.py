"""
智能选址 Web 界面。

评估管线集成（均由 engine.calculate_site_roi 编排）：
- search_service：全网情报（Tavily + LLM 摘要与缓存）
- guardrails：policy.json 一票否决（verify_safety）
另含 tools、agents 等，见 engine 模块。
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import unicodedata
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from geopy.location import Location
from pydantic import ValidationError

import guardrails  # noqa: F401
from email_service import send_assessment_email
from notification_service import send_enterprise_notification
import search_service  # noqa: F401
from engine import calculate_site_roi
from schema import SiteInput
from spy_service import build_competitor_intel_table

# 与 engine.py 中 ROI 公式保持一致
_DAYS_PER_MONTH = 30
_LABOR_PER_FTE_CNY = 6000
_MATERIAL_RATE = 0.35

# Nominatim 使用政策：约 1 请求/秒；仅在缓存未命中时休眠（见 _geocode_address）
_NOMINATIM_MIN_INTERVAL_SEC = 1.05
_NOMINATIM_USER_AGENT = "retail-expansion-agent/1.0 (streamlit; educational)"

# OSM/Nominatim 对「上海XX路门牌」中文检索常失败，英文路名命中率高（按路名长度降序匹配）
_SHANGHAI_ROAD_EN: tuple[tuple[str, str], ...] = (
    ("南京西路", "West Nanjing Road"),
    ("南京东路", "East Nanjing Road"),
    ("淮海中路", "Middle Huaihai Road"),
    ("淮海东路", "East Huaihai Road"),
    ("淮海西路", "West Huaihai Road"),
    ("西藏中路", "Middle Xizang Road"),
    ("西藏南路", "South Xizang Road"),
    ("四川北路", "North Sichuan Road"),
    ("四川中路", "Middle Sichuan Road"),
    ("马当路", "Madang Road"),
    ("世纪大道", "Century Avenue"),
    ("陆家嘴环路", "Lujiazui Ring Road"),
    ("张扬路", "Zhangyang Road"),
    ("金科路", "Jinke Road"),
    ("淞沪路", "Songhu Road"),
    ("徐家汇路", "Xujiahui Road"),
    ("共和新路", "Gonghexin Road"),
    ("愚园路", "Yuyuan Road"),
    ("长乐路", "Changle Road"),
    ("延安中路", "Middle Yan'an Road"),
    ("河南中路", "Middle Henan Road"),
    ("福州路", "Fuzhou Road"),
    ("人民广场", "People's Square"),
)


def _notification_summary(address: str, margin_pct: float, conclusion: str) -> str:
    """统一通知文案：地址、利润率、核心结论。"""
    return (
        "【选址评估通知】\n"
        f"地址：{address}\n"
        f"利润率：{margin_pct:.2f}%\n"
        f"核心结论：{conclusion}"
    )


def _email_payload(site: SiteInput, result: Any) -> dict[str, Any]:
    """构造邮件服务所需字段。"""
    return {
        "address": site.address,
        "monthly_rent_cny": site.monthly_rent_cny,
        "estimated_monthly_revenue_cny": result.estimated_monthly_revenue_cny,
        "estimated_net_margin_pct": result.estimated_net_margin_pct,
        "roi_pct": result.estimated_net_margin_pct,
        "expansion_approved": result.expansion_approved,
        "risk_block_reason": result.risk_block_reason or "",
        "promoter_opinion": result.promoter_opinion or "",
        "critic_opinion": result.critic_opinion or "",
    }


def analyze_site_image(image_file: Any) -> dict[str, Any]:
    """
    调用 Gemini 1.5 Flash 对现场图片做选址解析。
    返回结构化字段，失败时返回 {"ok": False, "error": "..."}。
    """
    api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return {"ok": False, "error": "缺失 GEMINI_API_KEY 配置"}
    model = str(st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")).strip()
    if not model:
        model = "gemini-1.5-flash"

    mime_type = getattr(image_file, "type", "") or "image/jpeg"
    if not str(mime_type).startswith("image/"):
        return {"ok": False, "error": "仅支持图片文件（image/*）"}

    image_bytes = image_file.getvalue()
    if not image_bytes:
        return {"ok": False, "error": "上传图片为空"}
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "你是“资深选址考察官”。请仅基于图片内容提取信息，禁止编造。"
        "请严格返回 JSON（不要 Markdown、不要额外说明），字段如下："
        '{'
        '"contact_phone":"",'
        '"store_area_sqm":null,'
        '"rent_requirement_text":"",'
        '"transfer_fee_text":"",'
        '"competitor_brands":[""],'
        '"crowd_quality":"",'
        '"summary":"",'
        '"confidence":"高/中/低"'
        "}"
        "；其中："
        "1) contact_phone 提取招租电话；"
        "2) store_area_sqm 仅填数字（平米），不确定填 null；"
        "3) competitor_brands 仅列画面可见同类品牌（如古茗/茶百道/瑞幸等）；"
        "4) crowd_quality 根据建筑形态与人群特征判断消费能力，并简述依据。"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text:
            return {"ok": False, "error": "Gemini 未返回可解析文本"}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return {"ok": False, "error": "Gemini 返回结构异常"}
        parsed["ok"] = True
        return parsed
    except Exception as e:
        return {"ok": False, "error": f"图像解析失败：{e}"}

# 批量表列名兼容（优先匹配左侧别名；见 _norm_header 规范化）
_COLUMN_ALIASES: dict[str, list[str]] = {
    "address": [
        "地址",
        "address",
        "点位地址",
        "门店地址",
        "店铺地址",
        "位置",
    ],
    "store_area_sqm": [
        "门店面积",
        "面积",
        "store_area_sqm",
        "面积平米",
        "面积㎡",
        "铺位面积",
        "建筑面积",
    ],
    "monthly_rent_cny": [
        "月租金",
        "租金",
        "monthly_rent_cny",
        "月租",
        "店铺租金",
        "房租",
    ],
    "estimated_daily_cups": [
        "预估日出杯量",
        "日出杯量",
        "杯量",
        "estimated_daily_cups",
        "日杯量",
        "日均杯量",
        "每日杯量",
    ],
    "avg_ticket_cny": [
        "平均客单价",
        "客单价",
        "avg_ticket_cny",
        "单价",
        "均价",
        "笔单价",
    ],
    "full_time_staff_count": [
        "单店全职员工数",
        "全职员工数",
        "全职员工",
        "员工数",
        "员工人数",
        "店员数",
        "店员人数",
        "人数",
        "人员数",
        "用工人数",
        "编制",
        "full_time_staff_count",
        "staff",
        "staff_count",
        "employees",
        "fte",
        "headcount",
        "员工",
        "店员",
        "人员",
    ],
}


def _norm_header(s: str) -> str:
    """表头规范化：去 BOM、全角空格、括号备注、空白，NFKC 后小写。"""
    t = str(s).strip().lstrip("\ufeff")
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[（(].*?[）)]", "", t)
    t = re.sub(r"[\s\u3000:_\-]+", "", t)
    return t.lower()


def _match_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # 原始列名 -> 规范化键（多对一取首次出现）
    key_to_orig: dict[str, str] = {}
    for c in df.columns:
        orig = str(c).strip().lstrip("\ufeff")
        key = _norm_header(orig)
        if key and key not in key_to_orig:
            key_to_orig[key] = orig

    # 1) 别名完全匹配（长别名优先）
    for cand in sorted(candidates, key=len, reverse=True):
        ck = _norm_header(cand)
        if ck and ck in key_to_orig:
            return key_to_orig[ck]

    # 2) 宽松：规范化后的表头「包含」核心别名（仍按长别名优先）
    for cand in sorted(candidates, key=len, reverse=True):
        ck = _norm_header(cand)
        if len(ck) < 2:
            continue
        for key, orig in key_to_orig.items():
            if ck in key or key in ck:
                return orig

    return None


def _guess_staff_column(df: pd.DataFrame) -> str | None:
    """员工数列最后兜底：表头语义接近「员工/人数」。"""
    for c in df.columns:
        orig = str(c).strip().lstrip("\ufeff")
        k = _norm_header(orig)
        if not k:
            continue
        if k in ("员工", "店员") or k.endswith(("员工数", "店员数", "员工人")):
            return orig
        if any(
            x in k
            for x in (
                "员工数",
                "员工人",
                "店员数",
                "全职",
                "staff",
                "fte",
                "headcount",
                "employees",
            )
        ):
            return orig
        if k in ("人数", "人员数", "用工", "编制"):
            return orig
    return None


def _resolve_input_columns(df: pd.DataFrame) -> dict[str, str]:
    """canonical SiteInput 字段 -> 实际列名"""
    out: dict[str, str] = {}
    missing: list[str] = []
    for canon, aliases in _COLUMN_ALIASES.items():
        col = _match_column(df, aliases)
        if col is None and canon == "full_time_staff_count":
            col = _guess_staff_column(df)
        if col is None:
            missing.append(canon)
        else:
            out[canon] = col
    if missing:
        raise ValueError(
            "无法匹配必填列："
            + "、".join(missing)
            + "。请确保表头包含：地址、门店面积、月租金、日出杯量、客单价、员工数（或英文同义列）。"
        )
    return out


def _row_to_site(row: pd.Series, colmap: dict[str, str]) -> SiteInput:
    def g(key: str) -> Any:
        v = row[colmap[key]]
        if pd.isna(v):
            raise ValueError(f"{key} 为空")
        return v

    return SiteInput(
        address=str(g("address")).strip(),
        store_area_sqm=float(g("store_area_sqm")),
        monthly_rent_cny=int(float(g("monthly_rent_cny"))),
        estimated_daily_cups=int(float(g("estimated_daily_cups"))),
        avg_ticket_cny=float(g("avg_ticket_cny")),
        full_time_staff_count=int(float(g("full_time_staff_count"))),
    )


def _read_uploaded_table(uploaded: Any) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    # 必须先判断 .xlsx：文件名若以 .xlsx 结尾，也会匹配 “.xls” 后缀
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    if name.endswith(".xls"):
        try:
            return pd.read_excel(io.BytesIO(raw), engine="xlrd")
        except ImportError as e:
            raise ValueError(
                "读取 .xls 需安装 xlrd：python -m pip install xlrd"
            ) from e
    raise ValueError("仅支持 .csv、.xlsx 或 .xls")


def _monthly_cost_breakdown(
    site: SiteInput, result_revenue: float
) -> tuple[float, float, float, float]:
    revenue = float(result_revenue)
    rent = float(site.monthly_rent_cny)
    labor = float(site.full_time_staff_count * _LABOR_PER_FTE_CNY)
    material = revenue * _MATERIAL_RATE
    net = revenue - rent - labor - material
    return rent, labor, material, net


def _build_cost_pie_fig(
    rent: float,
    labor: float,
    material: float,
    net: float,
):
    if net >= 0:
        df = pd.DataFrame(
            {
                "项目": [
                    "月租金",
                    "预估人工成本",
                    "预估原材料成本",
                    "预估净利润",
                ],
                "金额": [rent, labor, material, net],
            }
        )
        title = "月度成本结构（租金 + 人工 + 原材料 + 净利润 = 月营业额）"
    else:
        df = pd.DataFrame(
            {
                "项目": ["月租金", "预估人工成本", "预估原材料成本"],
                "金额": [rent, labor, material],
            }
        )
        title = "月度成本结构（三项成本占比；净利润为负，见下方说明）"

    fig = px.pie(
        df,
        names="项目",
        values="金额",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value:,.0f} 元<br>(%{percent})",
        textinfo="text",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "金额: %{value:,.2f} 元<br>"
            "占本图合计: %{percent}<extra></extra>"
        ),
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=60, b=80),
    )
    return fig


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _english_shanghai_queries(address: str) -> list[str]:
    """将常见「上海 + 路名 + 门牌」转为 Nominatim 易命中的英文检索串。"""
    a = unicodedata.normalize("NFKC", (address or "")).strip()
    if not a:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def push(q: str) -> None:
        q = " ".join(q.split())
        if len(q) < 4 or q in seen:
            return
        seen.add(q)
        out.append(q)

    for cn, en in sorted(_SHANGHAI_ROAD_EN, key=lambda x: -len(x[0])):
        if cn not in a:
            continue
        m = re.search(rf"{re.escape(cn)}\s*([0-9]+)\s*号?", a)
        num = m.group(1) if m else None
        if num:
            push(f"{en} {num}, Shanghai, China")
        push(f"{en}, Shanghai, China")
    return out


def _geocode_query_variants(address: str) -> list[str]:
    """
    生成 Nominatim 检索串：优先英文上海路名（OSM 命中率高），再试中文全称/路级回退。
    """
    a = unicodedata.normalize("NFKC", (address or "")).strip()
    if not a:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def add(q: str) -> None:
        q = " ".join(q.split())
        if len(q) < 2 or q in seen:
            return
        seen.add(q)
        out.append(q)

    # 1) 上海路名英译（解决「上海南京西路200号」类中文门址在 Nominatim 上普遍无结果）
    for q in _english_shanghai_queries(a):
        add(q)

    # 2) 中文变体
    add(a)
    add(f"{a}, 中国")

    municipals = (
        ("上海", "上海市"),
        ("北京", "北京市"),
        ("天津", "天津市"),
        ("重庆", "重庆市"),
    )
    for short, full in municipals:
        if a.startswith(full):
            rest = a[len(full) :].lstrip()
            if rest:
                add(f"{rest}, {full}, 中国")
            break
        if a.startswith(short):
            rest = a[len(short) :].lstrip("市").lstrip()
            expanded = full + rest
            add(expanded)
            add(f"{expanded}, 中国")
            if rest:
                add(f"{rest}, {full}, 中国")
            break

    street_only = re.sub(r"[0-9]+号\s*$", "", a).strip()
    if street_only != a and len(street_only) >= 4:
        add(f"{street_only}, 中国")
        for short, full in municipals:
            if street_only.startswith(full):
                r2 = street_only[len(full) :].lstrip()
                if r2:
                    add(f"{r2}, {full}, 中国")
                break
            if street_only.startswith(short):
                r2 = street_only[len(short) :].lstrip("市").lstrip()
                exp = full + r2
                add(f"{exp}, 中国")
                if r2:
                    add(f"{r2}, {full}, 中国")
                break

    return out[:22]


def _geocode_one(
    geolocator: Nominatim, query: str, *, restrict_china: bool
) -> Location | None:
    try:
        if _has_cjk(query):
            if restrict_china:
                return geolocator.geocode(
                    query,
                    language="zh",
                    country_codes="cn",
                    exactly_one=True,
                )
            return geolocator.geocode(query, language="zh", exactly_one=True)
        if restrict_china:
            return geolocator.geocode(
                query,
                language="en",
                country_codes="cn",
                exactly_one=True,
            )
        return geolocator.geocode(query, language="en", exactly_one=True)
    except (GeocoderTimedOut, GeocoderServiceError, OSError):
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def _geocode_address(address: str) -> tuple[float | None, float | None]:
    """
    将地址解析为 (纬度, 经度)。使用 Streamlit 磁盘缓存减少重复请求；
    缓存未命中时先休眠一次；多检索串之间再节流，以符合 Nominatim 使用政策。
    """
    addr = unicodedata.normalize("NFKC", (address or "")).strip()
    if not addr:
        return (None, None)
    time.sleep(_NOMINATIM_MIN_INTERVAL_SEC)
    geolocator = Nominatim(user_agent=_NOMINATIM_USER_AGENT, timeout=25)
    variants = _geocode_query_variants(addr)

    for i, q in enumerate(variants):
        if i > 0:
            time.sleep(_NOMINATIM_MIN_INTERVAL_SEC)
        loc = _geocode_one(geolocator, q, restrict_china=True)
        if loc is not None:
            return (float(loc.latitude), float(loc.longitude))

    if variants:
        time.sleep(_NOMINATIM_MIN_INTERVAL_SEC)
    for i, q in enumerate(variants[:10]):
        if i > 0:
            time.sleep(_NOMINATIM_MIN_INTERVAL_SEC)
        loc = _geocode_one(geolocator, q, restrict_china=False)
        if loc is not None:
            return (float(loc.latitude), float(loc.longitude))
    return (None, None)


def _render_batch_site_map(summary_df: pd.DataFrame) -> None:
    """在批量汇总结果上绘制绿（通过）/红（拦截或失败）点位图。"""
    if summary_df.empty or "地址" not in summary_df.columns:
        st.info("当前汇总表无地址列，无法绘制地图。")
        return

    if "决策建议" not in summary_df.columns:
        st.info("当前汇总表无决策列，无法区分颜色。")
        return

    addr_series = summary_df["地址"].fillna("").astype(str).str.strip()
    dec_series = summary_df["决策建议"].astype(str).str.strip()

    seen_addr: set[str] = set()
    addrs_in_order: list[str] = []
    for addr in addr_series:
        if not addr or addr in seen_addr:
            continue
        seen_addr.add(addr)
        addrs_in_order.append(addr)

    rows: list[tuple[float, float, bool]] = []
    failed_addr: list[str] = []

    with st.spinner("正在解析点位坐标（已缓存的地址不会重复请求 Nominatim）…"):
        for addr in addrs_in_order:
            lat, lon = _geocode_address(addr)
            if lat is None or lon is None:
                failed_addr.append(addr)
                continue
            mask = addr_series == addr
            # 同一地址多行：任一为「准予拓店」则标绿，否则标红（含拒绝、处理失败）
            is_green = bool((dec_series[mask] == "准予拓店").any())
            rows.append((lat, lon, is_green))

    if not rows:
        st.warning(
            "未能解析到任何有效坐标（可能受网络或 Nominatim 限制影响）。"
            + (f" 失败地址示例：{failed_addr[:3]}…" if failed_addr else "")
        )
        return

    center_lat = sum(p[0] for p in rows) / len(rows)
    center_lon = sum(p[1] for p in rows) / len(rows)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    for lat, lon, approved in rows:
        color = "#27ae60" if approved else "#e74c3c"
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(
                "准予拓店（expansion_approved）"
                if approved
                else "未准予拓店（拦截、拒绝或处理失败）",
                max_width=260,
            ),
        ).add_to(m)

    # 缩放到所有点
    if len(rows) == 1:
        m.location = [rows[0][0], rows[0][1]]
        m.zoom_start = 13
    else:
        sw = [min(p[0] for p in rows), min(p[1] for p in rows)]
        ne = [max(p[0] for p in rows), max(p[1] for p in rows)]
        m.fit_bounds([sw, ne], padding=(24, 24))

    legend_html = """
    <div style="position: fixed; bottom: 36px; left: 36px; z-index: 1000;
         background: white; padding: 10px 12px; border-radius: 6px;
         box-shadow: 0 1px 4px rgba(0,0,0,0.25); font-size: 13px;">
      <div><span style="color:#27ae60;font-weight:bold;">●</span> 准予拓店</div>
      <div><span style="color:#e74c3c;font-weight:bold;">●</span> 拒绝/拦截</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    components.html(m._repr_html_(), height=520, scrolling=False)
    if failed_addr:
        st.caption(f"以下地址未解析到坐标（已跳过）：共 {len(failed_addr)} 条。")


st.set_page_config(page_title="智能选址决策系统", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>智能选址决策系统 v1.0</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("访问验证")
    expected_access_token = str(st.secrets.get("ACCESS_TOKEN", "")).strip()
    access_token_input = st.text_input(
        "请输入访问令牌",
        type="password",
        key="access_token_input",
        help="令牌由管理员在 Streamlit secrets 中配置（ACCESS_TOKEN）。",
    )
    if not expected_access_token:
        st.error("后台未配置 ACCESS_TOKEN，请先在 secrets 中设置后再使用。")
        st.stop()
    if access_token_input != expected_access_token:
        st.warning("请输入正确的令牌以开启决策系统")
        st.stop()

    mode = st.radio(
        "评估模式",
        ["单店评估", "批量评估"],
        horizontal=True,
    )
    st.divider()
    if mode == "单店评估":
        st.header("点位参数")
        site_image = st.file_uploader(
            "上传店址现场图片（招租告示/街道实景）",
            type=["jpg", "jpeg", "png", "webp"],
            help="用于 AI 提取面积、联系方式、竞品与客群线索。",
        )
        if st.button("解析现场图片", use_container_width=True):
            if site_image is None:
                st.warning("请先上传现场图片。")
            else:
                with st.spinner("正在调用 Gemini 解析现场图片…"):
                    image_result = analyze_site_image(site_image)
                if image_result.get("ok"):
                    st.session_state["site_image_analysis"] = image_result
                    area_v = image_result.get("store_area_sqm")
                    if area_v not in (None, ""):
                        try:
                            st.session_state["store_area_sqm_input"] = float(area_v)
                        except Exception:
                            pass
                    phone_v = str(image_result.get("contact_phone", "")).strip()
                    if phone_v:
                        st.session_state["site_contact"] = phone_v
                    st.success("图片解析完成，已尝试自动回填面积与联系方式。")
                else:
                    st.error(str(image_result.get("error") or "图片解析失败"))

        address = st.text_input("地址", placeholder="例如：上海市静安区南京西路某号")
        _site_contact = st.text_input(
            "现场联系方式（可自动识别）",
            key="site_contact",
            placeholder="例如：13800000000",
        )
        monthly_rent_cny = st.number_input("月租金（元）", min_value=0, value=20_000, step=500)
        estimated_daily_cups = st.number_input("预估日出杯量（杯）", min_value=0, value=400, step=10)
        avg_ticket_cny = st.number_input("平均客单价（元）", min_value=0.01, value=25.0, step=0.5)
        full_time_staff_count = st.number_input("单店全职员工数（人）", min_value=0, value=3, step=1)
        store_area_sqm = st.number_input(
            "门店面积（平米）",
            min_value=0.01,
            value=60.0,
            step=1.0,
            key="store_area_sqm_input",
            help="SiteInput 必填，用于模型与政策校验。",
        )
    else:
        st.info(
            "批量模式：请在主区域上传 **.xlsx** 或 **.csv**。"
            "表头需含：地址、门店面积、月租金、预估日出杯量、平均客单价、单店全职员工数（可用中英文别名）。"
        )

if mode == "单店评估":
    if st.button("开始评估", type="primary"):
        if not (address or "").strip():
            st.warning("请先在侧边栏填写地址。")
        else:
            try:
                site = SiteInput(
                    address=address.strip(),
                    store_area_sqm=float(store_area_sqm),
                    monthly_rent_cny=int(monthly_rent_cny),
                    estimated_daily_cups=int(estimated_daily_cups),
                    avg_ticket_cny=float(avg_ticket_cny),
                    full_time_staff_count=int(full_time_staff_count),
                )
            except ValidationError as e:
                st.error("输入校验失败：" + str(e))
            else:
                with st.spinner("正在评估（情报检索、双智能体辩论、红线校验）…"):
                    result = calculate_site_roi(site)
                single_conclusion = (
                    "准予拓店"
                    if result.expansion_approved
                    else (result.risk_block_reason or "拒绝拓店（原因未提供）")
                )
                notify_ok = send_enterprise_notification(
                    _notification_summary(
                        site.address,
                        float(result.estimated_net_margin_pct),
                        single_conclusion,
                    )
                )
                email_ok = send_assessment_email(_email_payload(site, result))
                if notify_ok:
                    st.caption("通知已发送给管理者。")
                else:
                    st.caption("通知发送失败或未配置 Webhook。")
                if email_ok:
                    st.caption("评估邮件已发送。")
                else:
                    st.caption("评估邮件发送失败或未配置 SMTP。")

                st.subheader("核心指标")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(
                        "预估月营业额",
                        f"{result.estimated_monthly_revenue_cny:,.2f} 元",
                    )
                with m2:
                    st.metric(
                        "预估净利润率",
                        f"{result.estimated_net_margin_pct:.2f}%",
                    )

                rent_y, labor_y, material_y, net_y = _monthly_cost_breakdown(
                    site, result.estimated_monthly_revenue_cny
                )

                st.subheader("成本结构（交互式饼图）")
                if net_y < 0:
                    st.caption(
                        f"当前预估净利润 **{net_y:,.2f} 元** 为负；饼图仅展示租金、人工、原材料三项占比，"
                        "避免在饼图中使用负值切片。"
                    )
                fig = _build_cost_pie_fig(rent_y, labor_y, material_y, net_y)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("决策信号")
                if result.expansion_approved:
                    st.success("准予拓店：通过当前模型与红线校验。")
                else:
                    st.error(result.risk_block_reason or "拒绝拓店（原因未提供）")

                st.subheader("辩论过程")
                col_a, col_b = st.columns(2)
                with col_a:
                    with st.expander("拓店部（Promoter）", expanded=False):
                        st.markdown(result.promoter_opinion or "（无）")
                with col_b:
                    with st.expander("风控部（Critic）", expanded=False):
                        st.markdown(result.critic_opinion or "（无）")

                st.subheader("市场情报")
                st.text(result.market_intelligence or "（无）")

                st.subheader("竞品情报看板")
                with st.spinner(
                    "正在侦察全网竞品情报（美团/点评/抖音/京东视角的公开摘录 + 模型清洗）…"
                ):
                    spy_df, spy_hint = build_competitor_intel_table(site.address)
                if spy_df.empty:
                    st.markdown(
                        spy_hint
                        or "暂未生成竞品情报表。请确认已配置 **TAVILY_API_KEY** 与 "
                        "**DEEPSEEK_API_KEY**（或 **OPENAI_API_KEY**），并稍后重试。"
                    )
                else:
                    st.dataframe(spy_df, use_container_width=True, height=360)
                st.caption(
                    "说明：本看板基于联网搜索公开网页摘录经大模型结构化整理，"
                    "不等同于各平台官方实时排行或内部数据接口。"
                )

                img_info = st.session_state.get("site_image_analysis")
                if isinstance(img_info, dict) and img_info.get("ok"):
                    st.subheader("现场图片考察补充")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"联系方式：{img_info.get('contact_phone') or '未识别'}")
                        st.write(
                            f"面积线索：{img_info.get('store_area_sqm') or '未识别'} 平米"
                        )
                        st.write(
                            "租金要求："
                            + str(img_info.get("rent_requirement_text") or "未识别")
                        )
                        st.write(
                            "转让费用："
                            + str(img_info.get("transfer_fee_text") or "未识别")
                        )
                    with c2:
                        brands = img_info.get("competitor_brands") or []
                        if isinstance(brands, list):
                            btxt = "、".join(str(x) for x in brands if str(x).strip())
                        else:
                            btxt = str(brands)
                        st.write(f"可见竞品品牌：{btxt or '未识别'}")
                        st.write(
                            "客群质量判断："
                            + str(img_info.get("crowd_quality") or "未识别")
                        )
                        st.caption(str(img_info.get("summary") or ""))

else:
    st.subheader("批量评估")
    uploaded = st.file_uploader(
        "上传点位表（.xlsx / .xls / .csv）",
        type=["xlsx", "xls", "csv"],
        help="每行一个点位；列名需可映射到：地址、门店面积、月租金、日出杯量、客单价、员工数。",
    )
    run_batch = st.button("开始批量评估", type="primary")

    if run_batch:
        if uploaded is None:
            st.warning("请先上传表格文件。")
        else:
            try:
                df_in = _read_uploaded_table(uploaded)
            except Exception as e:
                st.error(f"读取文件失败：{e}")
            else:
                try:
                    colmap = _resolve_input_columns(df_in)
                except ValueError as e:
                    st.error(str(e))
                    st.dataframe(df_in.head(20), use_container_width=True)
                else:
                    n = len(df_in)
                    if n == 0:
                        st.warning("表格为空。")
                    else:
                        rows_out: list[dict[str, Any]] = []
                        notify_success = 0
                        notify_failed = 0
                        email_success = 0
                        email_failed = 0
                        with st.status("批量评估进行中", expanded=True) as run_status:
                            progress = st.progress(0)
                            for idx, (_, row) in enumerate(df_in.iterrows()):
                                progress.progress((idx + 1) / n)
                                run_status.update(
                                    label=f"正在评估第 {idx + 1} / {n} 个点位"
                                )
                                addr_display = ""
                                try:
                                    site = _row_to_site(row, colmap)
                                    addr_display = site.address
                                    result = calculate_site_roi(site)
                                    decision = (
                                        "准予拓店"
                                        if result.expansion_approved
                                        else "拒绝拓店"
                                    )
                                    reason = result.risk_block_reason or ""
                                    intel = result.market_intelligence or ""
                                    rows_out.append(
                                        {
                                            "地址": addr_display,
                                            "利润率%": round(
                                                result.estimated_net_margin_pct, 2
                                            ),
                                            "决策建议": decision,
                                            "拦截原因": reason,
                                            "全网情报摘要": intel,
                                            "预估月营业额": round(
                                                result.estimated_monthly_revenue_cny,
                                                2,
                                            ),
                                            "竞品数": result.competitor_count,
                                            "拓店部意见": result.promoter_opinion
                                            or "",
                                            "风控部意见": result.critic_opinion
                                            or "",
                                        }
                                    )
                                    row_conclusion = (
                                        "准予拓店"
                                        if result.expansion_approved
                                        else (
                                            result.risk_block_reason or "拒绝拓店"
                                        )
                                    )
                                    notify_ok = send_enterprise_notification(
                                        _notification_summary(
                                            addr_display,
                                            float(result.estimated_net_margin_pct),
                                            row_conclusion,
                                        )
                                    )
                                    if notify_ok:
                                        notify_success += 1
                                    else:
                                        notify_failed += 1
                                    email_ok = send_assessment_email(
                                        _email_payload(site, result)
                                    )
                                    if email_ok:
                                        email_success += 1
                                    else:
                                        email_failed += 1
                                except Exception as ex:
                                    try:
                                        raw_addr = row[colmap["address"]]
                                        addr_fail = (
                                            ""
                                            if pd.isna(raw_addr)
                                            else str(raw_addr).strip()
                                        )
                                    except Exception:
                                        addr_fail = addr_display
                                    rows_out.append(
                                        {
                                            "地址": addr_fail or addr_display,
                                            "利润率%": None,
                                            "决策建议": "处理失败",
                                            "拦截原因": str(ex),
                                            "全网情报摘要": "",
                                            "预估月营业额": None,
                                            "竞品数": None,
                                            "拓店部意见": "",
                                            "风控部意见": "",
                                        }
                                    )
                            progress.progress(1.0)
                            run_status.update(
                                label=f"已完成 {n} 个点位", state="complete"
                            )

                        summary_df = pd.DataFrame(rows_out)
                        st.session_state["batch_summary_for_map"] = summary_df

                        st.success(f"已完成 {n} 个点位的评估。")
                        st.subheader("结果汇总")
                        st.dataframe(summary_df, use_container_width=True, height=420)

                        st.subheader("点位空间分布图")
                        _render_batch_site_map(summary_df)
                        st.caption(
                            f"通知推送统计：成功 {notify_success} 条，失败/未配置 {notify_failed} 条。"
                        )
                        st.caption(
                            f"邮件发送统计：成功 {email_success} 条，失败/未配置 {email_failed} 条。"
                        )

                        csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="下载汇总表（CSV）",
                            data=csv_bytes,
                            file_name="选址批量评估汇总.csv",
                            mime="text/csv",
                        )

                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                            summary_df.to_excel(
                                writer, index=False, sheet_name="评估汇总"
                            )
                        buf.seek(0)
                        st.download_button(
                            label="下载汇总表（Excel）",
                            data=buf.getvalue(),
                            file_name="选址批量评估汇总.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

    elif st.session_state.get("batch_summary_for_map") is not None:
        sdf = st.session_state["batch_summary_for_map"]
        st.subheader("最近一次批量评估结果")
        st.caption("交互导致页面刷新后仍可查看上次汇总与地图；重新点击「开始批量评估」将覆盖。")
        st.dataframe(sdf, use_container_width=True, height=420)
        st.subheader("点位空间分布图")
        _render_batch_site_map(sdf)
        csv_bytes = sdf.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下载汇总表（CSV）",
            data=csv_bytes,
            file_name="选址批量评估汇总.csv",
            mime="text/csv",
        )
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            sdf.to_excel(writer, index=False, sheet_name="评估汇总")
        buf.seek(0)
        st.download_button(
            label="下载汇总表（Excel）",
            data=buf.getvalue(),
            file_name="选址批量评估汇总.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
