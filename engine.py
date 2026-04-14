"""选址 ROI 粗算引擎：由 SiteInput 推导 SiteAssessment。"""

from __future__ import annotations

import tools
from agents import run_debate
from guardrails import verify_safety
from search_service import get_real_world_context
from usage_context import reset_llm_usage

from schema import SiteAssessment, SiteInput

_DAYS_PER_MONTH = 30
_LABOR_PER_FTE_CNY = 6000
_MATERIAL_RATE = 0.35
_MARGIN_THRESHOLD_PCT = 15.0
_COMPETITOR_DENSITY_MAX = 15

# 与 search_service 产出的精简摘要对齐；命中则触发情报风控（摘要经 OpenAI 兼容 API 基于 Tavily 提炼）。
_INTEL_NEGATIVE_KEYWORDS = (
    "倒闭",
    "修路",
    "人流下滑",
    "竞争惨烈",
    "撤铺",
)


def _intel_risk_hits(text: str) -> list[str]:
    return [kw for kw in _INTEL_NEGATIVE_KEYWORDS if kw in text]


def calculate_site_roi(input_data: SiteInput) -> SiteAssessment:
    reset_llm_usage()
    market_intelligence = get_real_world_context(input_data.address)
    competitor_count = tools.get_competitor_density(input_data.address)

    def _apply_guardrails(assessment: SiteAssessment) -> SiteAssessment:
        return verify_safety(
            assessment,
            market_intelligence,
            input_data.monthly_rent_cny,
        )

    revenue = (
        input_data.estimated_daily_cups
        * input_data.avg_ticket_cny
        * _DAYS_PER_MONTH
    )
    labor = input_data.full_time_staff_count * _LABOR_PER_FTE_CNY
    material = revenue * _MATERIAL_RATE
    total_cost = input_data.monthly_rent_cny + labor + material
    net_profit = revenue - total_cost

    if revenue > 0:
        net_margin_pct = (net_profit / revenue) * 100.0
    else:
        net_margin_pct = 0.0

    promoter_opinion, critic_opinion = run_debate(
        input_data,
        revenue,
        net_margin_pct,
        competitor_count,
        market_intelligence,
    )

    if competitor_count > _COMPETITOR_DENSITY_MAX:
        return _apply_guardrails(
            SiteAssessment(
                estimated_monthly_revenue_cny=revenue,
                estimated_net_margin_pct=net_margin_pct,
                competitor_count=competitor_count,
                market_intelligence=market_intelligence,
                promoter_opinion=promoter_opinion,
                critic_opinion=critic_opinion,
                expansion_approved=False,
                risk_block_reason=(
                    f"周边竞品密度过高（{competitor_count}家），竞争过于激烈"
                ),
            )
        )

    if net_margin_pct < _MARGIN_THRESHOLD_PCT:
        margin_str = f"{net_margin_pct:.2f}"
        return _apply_guardrails(
            SiteAssessment(
                estimated_monthly_revenue_cny=revenue,
                estimated_net_margin_pct=net_margin_pct,
                competitor_count=competitor_count,
                market_intelligence=market_intelligence,
                promoter_opinion=promoter_opinion,
                critic_opinion=critic_opinion,
                expansion_approved=False,
                risk_block_reason=f"预估净利润率{margin_str}%低于公司15%的红线",
            )
        )

    hits = _intel_risk_hits(market_intelligence)
    if hits:
        return _apply_guardrails(
            SiteAssessment(
                estimated_monthly_revenue_cny=revenue,
                estimated_net_margin_pct=net_margin_pct,
                competitor_count=competitor_count,
                market_intelligence=market_intelligence,
                promoter_opinion=promoter_opinion,
                critic_opinion=critic_opinion,
                expansion_approved=False,
                risk_block_reason=(
                    "受实时市场情报风险拦截（基于 DeepSeek/OpenAI 兼容摘要）"
                    f"（触达：{'、'.join(hits)}）"
                ),
            )
        )

    return _apply_guardrails(
        SiteAssessment(
            estimated_monthly_revenue_cny=revenue,
            estimated_net_margin_pct=net_margin_pct,
            competitor_count=competitor_count,
            market_intelligence=market_intelligence,
            promoter_opinion=promoter_opinion,
            critic_opinion=critic_opinion,
            expansion_approved=True,
            risk_block_reason=None,
        )
    )


if __name__ == "__main__":
    sample = SiteInput(
        address="上海市静安区南京西路某号",
        store_area_sqm=60.0,
        monthly_rent_cny=20_000,
        estimated_daily_cups=500,
        avg_ticket_cny=15.0,
        full_time_staff_count=3,
    )
    result = calculate_site_roi(sample)
    print(result.model_dump())
