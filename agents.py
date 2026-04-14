"""多智能体：拓店总监 vs 风控总监（基于同一套数据与情报的对抗性意见）。"""

from __future__ import annotations

from search_service import llm_chat
from schema import SiteInput

_PROMOTER_SYSTEM = """你是连锁茶饮品牌的「拓店总监」，决策风格积极但需可落地。
你必须体现品牌经营逻辑：重视「品牌曝光度」（临街展示、商圈势能、可识别性）与「外卖配送覆盖面」（配送半径内人口/写字楼密度、骑手友好度、线上可见订单潜力）。
你要主动从情报与数据中挖掘「增量机会」：例如周边是否有即将开业或筹备中的商场/综合体、新建或规划中的地铁出入口、道路改造后的人流再分配等（无依据时写「情报未体现」勿编造）。
输出格式只允许使用下方用户消息规定的 Markdown 骨架，勿写开场白或套话结语。"""

_CRITIC_SYSTEM = """你是总部的「风控总监」，决策风格极度审慎。
在给出任何结论前，你必须先做思维链分析（CoT），且三个维度缺一不可：
1）「固定成本风险」：租金刚性、人工与原材料占比、盈亏平衡点敏感性等；
2）「竞品渗透率」：周边同类门店密度、差异化空间、价格战风险（输入中的竞品数为对 500 米内同类密度的业务代理指标）；
3）「人流质量」：客群与时段、目的性客流 vs 过路客、情报中与人流相关的信号。

硬性负面样本（必须遵守）：若「租售比（租金占月营业额）> 40%」且「周边同类竞品在 500 米尺度上达到 3 家以上」（以输入字段「周边竞品数」≥3 为工程代理），则你在「风控结论」三条中必须整体采用极度保守口径，明确提示不宜轻率拓店或须大幅降租/提量后再议，并在思维链中显式写出已触发该规则。
若未同时满足上述两条件，仍须完成三项思维链后再输出结论，但不必套用极度保守模板。

输出格式只允许使用下方用户消息规定的 Markdown 骨架，勿写开场白或套话结语。"""

# 竞品数为粗粒度代理：「3 家以上」→ 周边竞品数 ≥ 3
_ULTRA_CONSERVATIVE_COMPETITOR_THRESHOLD = 3

_PROMOTER_USER = """以下为该点位的结构化输入与全网情报摘要。请严格按下列 **Markdown 骨架** 输出（保留标题行；无内容可写「暂无」勿删节）：

## 结构化解题（拓店）

### 品牌曝光度
（1–3 句，可引用地址/商圈/情报）

### 外卖配送覆盖面
（1–3 句）

### 增量机会
（即将开业商场、新地铁口、道路/商圈变化等；无则写「情报未体现」）

## 核心优势（恰好 3 条）
1. …
2. …
3. …

---
【数据与情报】
{context}
"""

_CRITIC_USER = """以下为该点位的结构化输入与全网情报摘要。请严格按下列 **Markdown 骨架** 输出（保留标题行；无内容可写「待核实」勿删节）：

## 思维链分析

### 固定成本风险
（若干句）

### 竞品渗透率
（若干句）

### 人流质量
（若干句）

## 风控结论（恰好 3 条）
1. …
2. …
3. …

---
【数据与情报】
{context}

【触发判定（供你引用，勿改数字）】
- 租售比 = 租金占月营业额百分比；是否大于 40%：{rent_over_40}
- 周边竞品数：{competitor_count}；是否满足「3 家及以上」（竞品数 ≥ {_threshold}）：{competitor_hit}
- 极度保守规则是否同时满足：{rule_fired}（若为「是」，风控结论三条必须极度保守并在思维链中点名本规则）
"""

_FALLBACK = "（双智能体意见暂不可用：请配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。）"


def _build_context(
    input_data: SiteInput,
    revenue_cny: float,
    net_margin_pct: float,
    competitor_count: int,
    market_intelligence: str,
) -> str:
    rent_ratio = (
        (input_data.monthly_rent_cny / revenue_cny * 100.0)
        if revenue_cny > 0
        else None
    )
    rr = f"{rent_ratio:.1f}%" if rent_ratio is not None else "无法计算（营业额为 0）"
    return (
        f"【点位地址】{input_data.address}\n"
        f"【门店面积】{input_data.store_area_sqm} 平米\n"
        f"【月租金】{input_data.monthly_rent_cny} 元\n"
        f"【预估日出杯量】{input_data.estimated_daily_cups} 杯\n"
        f"【平均客单价】{input_data.avg_ticket_cny} 元\n"
        f"【单店全职员工数】{input_data.full_time_staff_count} 人\n"
        f"【预估月营业额】{revenue_cny:,.2f} 元\n"
        f"【预估净利润率】{net_margin_pct:.2f}%\n"
        f"【租金占营业额比（租售比）】{rr}\n"
        f"【周边竞品数（500 米密度之业务代理）】{competitor_count}\n"
        f"【全网情报摘要】\n{market_intelligence}\n"
    )


def _critic_trigger_flags(
    revenue_cny: float,
    monthly_rent_cny: int,
    competitor_count: int,
) -> tuple[str, str, str, str]:
    if revenue_cny <= 0:
        rent_over_40 = "否（营业额为 0，无法计算租售比）"
        competitor_hit = "是" if competitor_count >= _ULTRA_CONSERVATIVE_COMPETITOR_THRESHOLD else "否"
        rule_fired = "否"
        return rent_over_40, str(competitor_count), competitor_hit, rule_fired
    ratio_pct = (monthly_rent_cny / revenue_cny) * 100.0
    over = ratio_pct > 40.0
    comp_hit = competitor_count >= _ULTRA_CONSERVATIVE_COMPETITOR_THRESHOLD
    rent_over_40 = "是" if over else "否"
    competitor_hit = "是" if comp_hit else "否"
    rule_fired = "是" if (over and comp_hit) else "否"
    return rent_over_40, str(competitor_count), competitor_hit, rule_fired


def promoter_agent(context: str) -> str:
    """拓店总监：结构化 Markdown（品牌曝光、外卖覆盖、增量机会 + 3 条优势）。"""
    text = llm_chat(
        [
            {"role": "system", "content": _PROMOTER_SYSTEM},
            {"role": "user", "content": _PROMOTER_USER.format(context=context)},
        ],
        temperature=0.45,
        max_tokens=1100,
    )
    return text if text else _FALLBACK


def critic_agent(
    context: str,
    *,
    rent_over_40: str,
    competitor_count: str,
    competitor_hit: str,
    rule_fired: str,
) -> str:
    """风控总监：CoT + 负面样本规则 + 3 条结论，结构化 Markdown。"""
    text = llm_chat(
        [
            {"role": "system", "content": _CRITIC_SYSTEM},
            {
                "role": "user",
                "content": _CRITIC_USER.format(
                    context=context,
                    rent_over_40=rent_over_40,
                    competitor_count=competitor_count,
                    competitor_hit=competitor_hit,
                    rule_fired=rule_fired,
                    _threshold=_ULTRA_CONSERVATIVE_COMPETITOR_THRESHOLD,
                ),
            },
        ],
        temperature=0.35,
        max_tokens=1100,
    )
    return text if text else _FALLBACK


def run_debate(
    input_data: SiteInput,
    revenue_cny: float,
    net_margin_pct: float,
    competitor_count: int,
    market_intelligence: str,
) -> tuple[str, str]:
    """先后调用两角色，返回 (拓店部意见, 风控部意见)。"""
    ctx = _build_context(
        input_data,
        revenue_cny,
        net_margin_pct,
        competitor_count,
        market_intelligence,
    )
    r40, cc, ch, fired = _critic_trigger_flags(
        revenue_cny,
        input_data.monthly_rent_cny,
        competitor_count,
    )
    return promoter_agent(ctx), critic_agent(
        ctx,
        rent_over_40=r40,
        competitor_count=cc,
        competitor_hit=ch,
        rule_fired=fired,
    )
