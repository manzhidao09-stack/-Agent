"""示例点位数据：验证 SiteInput 对典型「健康店 / 亏损店」输入的校验行为。"""

from __future__ import annotations

from pydantic import ValidationError

from schema import SiteInput

# 健康店：面积适中、租金占比合理、日出杯与客单尚可、人力配置正常
HEALTHY_SITE_RAW = {
    "address": "杭州市西湖区文三路",
    "store_area_sqm": 65.0,
    "monthly_rent_cny": 28000,
    "estimated_daily_cups": 420,
    "avg_ticket_cny": 32.0,
    "full_time_staff_count": 5,
}

# 亏损店：输入层面仍合法（高租、低杯量），便于后续规则引擎算出差评估
LOSS_SITE_RAW = {
    "address": "北京市朝阳区某商圈",
    "store_area_sqm": 45.0,
    "monthly_rent_cny": 55000,
    "estimated_daily_cups": 80,
    "avg_ticket_cny": 22.0,
    "full_time_staff_count": 4,
}


def main() -> None:
    for label, raw in (
        ("健康店", HEALTHY_SITE_RAW),
        ("亏损店", LOSS_SITE_RAW),
    ):
        try:
            site = SiteInput.model_validate(raw)
        except ValidationError as e:
            print(f"{label}: 校验失败\n{e}")
            raise
        print(f"{label}: SiteInput 校验通过 -> {site.model_dump()}")


if __name__ == "__main__":
    main()
