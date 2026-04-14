"""一票否决：policy.json 红线 + 租金占比硬拦截。"""

from __future__ import annotations

import json
from pathlib import Path

from schema import SiteAssessment

_POLICY_PATH = Path(__file__).resolve().parent / "policy.json"
_HARD_PREFIX = "【硬核红线拦截】"


def _default_policy() -> dict:
    return {
        "blacklisted_keywords": ["拆迁", "违建", "产权纠纷", "无产证", "消防不达标"],
        "max_rent_ratio": 0.45,
        "min_area_requirement": 15,
    }


def _load_policy() -> dict:
    if not _POLICY_PATH.is_file():
        return _default_policy()
    try:
        raw = _POLICY_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return _default_policy()
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return _default_policy()


def verify_safety(
    assessment: SiteAssessment,
    intelligence: str,
    monthly_rent_cny: int,
) -> SiteAssessment:
    """
    基于 policy.json 与租金占比做最终拦截；若触发，expansion_approved=False，
    risk_block_reason 以「【硬核红线拦截】」开头。
    intelligence：用于黑名单关键词匹配（可与 assessment.market_intelligence 一致）。
    monthly_rent_cny：月租金，用于 租金/月营业额 占比。
    """
    policy = _load_policy()
    keywords = list(policy.get("blacklisted_keywords") or [])
    max_ratio = float(policy.get("max_rent_ratio", 0.45))

    guard_parts: list[str] = []

    kw_hits = [k for k in keywords if k and k in intelligence]
    if kw_hits:
        guard_parts.append(f"全网情报触达黑名单关键词：{'、'.join(kw_hits)}")

    revenue = float(assessment.estimated_monthly_revenue_cny)
    if revenue > 0:
        rent_ratio = monthly_rent_cny / revenue
        if rent_ratio > max_ratio:
            guard_parts.append(
                f"预估租金占月营业额 {rent_ratio * 100:.1f}%"
                f"，超过政策上限 {max_ratio * 100:.0f}%"
            )
    elif monthly_rent_cny > 0:
        guard_parts.append(
            "月营业额为零或无法计算，租金仍为正，租金占比视为超过红线"
        )

    if not guard_parts:
        return assessment

    guard_text = _HARD_PREFIX + "；".join(guard_parts)
    if not assessment.expansion_approved and (
        assessment.risk_block_reason
        and assessment.risk_block_reason.strip()
    ):
        merged = f"{guard_text}｜引擎原判：{assessment.risk_block_reason.strip()}"
    else:
        merged = guard_text

    return assessment.model_copy(
        update={
            "expansion_approved": False,
            "risk_block_reason": merged,
        }
    )
