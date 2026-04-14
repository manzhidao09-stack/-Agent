"""外部能力模拟：地图竞品密度等。"""

from __future__ import annotations

import random


def get_competitor_density(address: str) -> int:
    """模拟地图 API：按地址返回周边竞品门店数量（随机，仅用于演示）。"""
    if "上海" in address:
        return random.randint(10, 20)
    return random.randint(0, 10)
