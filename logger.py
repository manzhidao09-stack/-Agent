"""评估报告：结构化 Markdown 落盘至 evaluations/。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from schema import SiteAssessment, SiteInput
from usage_context import get_llm_usage_chars

_ROOT = Path(__file__).resolve().parent
_EVAL_DIR = _ROOT / "evaluations"

# 粗算：1,000,000 Tokens ≈ 1 元人民币（可按业务调整）
_TOKENS_PER_CNY = 1_000_000
# 字数 → Token 的粗略折算（中英混合场景下的经验比例，仅作展示）
_CHARS_PER_TOKEN_EST = 2.0


def _safe_filename_fragment(text: str, max_len: int = 48) -> str:
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", text.strip())
    s = s.strip("._") or "unknown"
    return s[:max_len]


def save_report(
    assessment: SiteAssessment,
    *,
    site_input: SiteInput | None = None,
    llm_chars_override: int | None = None,
) -> Path:
    """
    将一次完整评估写入 evaluations/ 下 Markdown 文件。
    文件名：日期 + 地址（已做文件名安全处理）。
    site_input：原始经营数据；若省略则报告中该节标注为未提供。
    llm_chars_override：若不传，则读取 usage_context 中本次累计字符数。
    """
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    date_part = datetime.now().strftime("%Y-%m-%d")
    addr_part = (
        _safe_filename_fragment(site_input.address)
        if site_input is not None
        else _safe_filename_fragment(assessment.market_intelligence[:32])
    )
    base = f"{date_part}_{addr_part}.md"
    path = _EVAL_DIR / base
    if path.exists():
        path = _EVAL_DIR / f"{date_part}_{addr_part}_{datetime.now().strftime('%H%M%S')}.md"

    chars = llm_chars_override if llm_chars_override is not None else get_llm_usage_chars()
    tokens_est = max(1, int(chars / _CHARS_PER_TOKEN_EST))
    cost_cny = tokens_est / _TOKENS_PER_CNY

    lines: list[str] = [
        "# 连锁门店选址评估报告",
        "",
        f"- **生成时间**：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 一、原始经营数据（输入）",
        "",
    ]

    if site_input is not None:
        lines.extend(
            [
                f"- **点位地址**：{site_input.address}",
                f"- **门店面积**：{site_input.store_area_sqm} 平米",
                f"- **月租金**：{site_input.monthly_rent_cny} 元",
                f"- **预估日出杯量**：{site_input.estimated_daily_cups} 杯",
                f"- **平均客单价**：{site_input.avg_ticket_cny} 元",
                f"- **单店全职员工数**：{site_input.full_time_staff_count} 人",
                "",
            ]
        )
    else:
        lines.extend(["*（本次未传入 SiteInput 快照，原始经营数据略）*", ""])

    lines.extend(
        [
            "## 二、抓取到的全网情报",
            "",
            assessment.market_intelligence.strip() or "（空）",
            "",
            "## 三、【辩论全过程】",
            "",
            "### 拓店总监（Promoter Agent）完整发言",
            "",
            assessment.promoter_opinion.strip() or "（空）",
            "",
            "### 风控总监（Critic Agent）完整发言",
            "",
            assessment.critic_opinion.strip() or "（空）",
            "",
            "## 四、最终决策与红线拦截",
            "",
            f"- **是否准予拓店**：{'是' if assessment.expansion_approved else '否'}",
            f"- **风险/拦截说明**：{assessment.risk_block_reason or '（无）'}",
            "",
            "### 模型侧关键财务与竞品指标（供复核）",
            "",
            f"- 预估月营业额：{assessment.estimated_monthly_revenue_cny:,.2f} 元",
            f"- 预估净利润率：{assessment.estimated_net_margin_pct:.2f}%",
            f"- 周边竞品数：{assessment.competitor_count}",
            "",
            "## 五、本次评估 LLM 成本粗估",
            "",
            "| 项目 | 数值 |",
            "| --- | --- |",
            f"| 本次累计 LLM 相关字数（请求+返回文本粗计） | {chars} 字 |",
            f"| 折合 Tokens（粗算：字数 ÷ {_CHARS_PER_TOKEN_EST:g}） | 约 {tokens_est} Tokens |",
            f"| 计价假设 | 1 元 / {_TOKENS_PER_CNY:,} Tokens |",
            f"| **估算成本（人民币）** | **{cost_cny:.8f} 元** |",
            "",
            "*说明：字数为本地统计的提示与返回文本长度之和，非厂商计费账单；Token 折算仅供内部量级参考。*",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

