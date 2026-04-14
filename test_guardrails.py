from schema import SiteAssessment
from guardrails import verify_safety

# 模拟引擎认为可拓店，但情报里含政策黑名单词（拆迁）
mock_assessment = SiteAssessment(
    estimated_monthly_revenue_cny=200_000,
    estimated_net_margin_pct=25.0,
    competitor_count=5,
    market_intelligence="该地段人流极旺，但属于违章建筑，近期面临拆迁风险。",
    promoter_opinion="（测试占位）",
    critic_opinion="（测试占位）",
    expansion_approved=True,
    risk_block_reason=None,
)

print("--- 拦截前状态 ---")
print(f"审批结果: {mock_assessment.expansion_approved}")

# 租金 3 万 / 营业额 20 万 = 15%，不触发租金红线；由黑名单「拆迁」拦截
monthly_rent_cny = 30_000
final_result = verify_safety(
    mock_assessment,
    mock_assessment.market_intelligence,
    monthly_rent_cny,
)

print("\n--- 拦截后状态 ---")
print(f"审批结果: {final_result.expansion_approved}")
print(f"拦截原因: {final_result.risk_block_reason}")
