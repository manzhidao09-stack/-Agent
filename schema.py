"""连锁门店选址评估 — 业务数据模型（Pydantic）。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SiteInput(BaseModel):
    """点位输入信息：选址侧采集或填报的经营与物业基础参数。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    address: str = Field(
        ...,
        min_length=1,
        description="点位地址：用于调用地图类服务估算周边竞品密度等。",
    )
    store_area_sqm: float = Field(
        ...,
        gt=0,
        description="门店面积(平米)：租赁或使用面积，用于坪效与容量粗算。",
    )
    monthly_rent_cny: int = Field(
        ...,
        ge=0,
        description="月租金(元)：含税与否以业务约定为准，须与财务口径一致。",
    )
    estimated_daily_cups: int = Field(
        ...,
        ge=0,
        description="预估日出杯量(杯)：按工作日/周末可另做场景模型，此处为综合日均杯数。",
    )
    avg_ticket_cny: float = Field(
        ...,
        gt=0,
        description="平均客单价(元)：单笔订单均值，用于营业额推算。",
    )
    full_time_staff_count: int = Field(
        ...,
        ge=0,
        description="单店全职员工数(人)：不含兼职时填 0；人力成本与合规评估的输入。",
    )


class SiteAssessment(BaseModel):
    """系统评估结果：模型或规则引擎输出的财务与风控结论。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    estimated_monthly_revenue_cny: float = Field(
        ...,
        ge=0,
        description="预估月营业额(元)：由杯量、客单价、营业天数等推导的月度收入估计。",
    )
    estimated_net_margin_pct: float = Field(
        ...,
        description="预估净利润率(百分比)：如 12.5 表示 12.5%，可为负表示预计亏损。",
    )
    competitor_count: int = Field(
        ...,
        ge=0,
        description="周边竞品门店数量：根据点位地址调用地图/密度工具抓取或模拟得到。",
    )
    market_intelligence: str = Field(
        ...,
        description="全网实时情报：Tavily 等搜索返回的周边茶饮与商圈摘要文本。",
    )
    promoter_opinion: str = Field(
        ...,
        description="拓店部（Promoter Agent）基于数据与情报列出的核心优势。",
    )
    critic_opinion: str = Field(
        ...,
        description="风控部（Critic Agent）列出的潜在风险与死穴。",
    )
    expansion_approved: bool = Field(
        ...,
        description="是否准予拓店：True 表示通过评估可进入下一流程；False 表示拦截或待复议。",
    )
    risk_block_reason: Optional[str] = Field(
        None,
        description="风控拦截原因：未拦截时为 None；被拦截时必须填写具体规则或人工备注。",
    )

    @model_validator(mode="after")
    def require_block_reason_when_not_approved(self) -> SiteAssessment:
        if not self.expansion_approved:
            if self.risk_block_reason is None or not self.risk_block_reason.strip():
                raise ValueError("未准予拓店时，风控拦截原因不能为空。")
        return self
