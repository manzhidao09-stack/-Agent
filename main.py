"""交互式连锁门店选址评估入口。"""

from __future__ import annotations

import sys
from typing import Optional

from pydantic import ValidationError

from engine import calculate_site_roi
from logger import save_report
from schema import SiteInput

_EXIT = "exit"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass


def _read_line(prompt: str) -> Optional[str]:
    """返回 None 表示用户要求退出。"""
    line = input(prompt).strip()
    if line.lower() == _EXIT:
        return None
    return line


def _prompt_int(label: str) -> Optional[int]:
    while True:
        raw = _read_line(f"{label}（输入 {_EXIT} 退出）：")
        if raw is None:
            return None
        try:
            return int(raw, 10)
        except ValueError:
            print("请输入有效的整数，例如 20000。若输错了可以重新输入。")


def _prompt_float(label: str) -> Optional[float]:
    while True:
        raw = _read_line(f"{label}（输入 {_EXIT} 退出）：")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            print("请输入有效的数字，例如 65 或 32.5。若输错了可以重新输入。")


def _prompt_address() -> Optional[str]:
    while True:
        raw = _read_line(f"请输入点位地址（输入 {_EXIT} 退出）：")
        if raw is None:
            return None
        if raw == "":
            print("地址不能为空，请重新输入一行文字描述位置。")
            continue
        return raw


def _build_site_input() -> tuple[Optional[SiteInput], bool]:
    """返回 (点位或 None, 是否用户主动退出)。校验失败返回 (None, False) 以便外层继续循环。"""
    addr = _prompt_address()
    if addr is None:
        return None, True
    area = _prompt_float("请输入门店面积(平米)")
    if area is None:
        return None, True
    rent = _prompt_int("请输入月租金(元)")
    if rent is None:
        return None, True
    cups = _prompt_int("请输入预估日出杯量(杯)")
    if cups is None:
        return None, True
    ticket = _prompt_float("请输入平均客单价(元)")
    if ticket is None:
        return None, True
    staff = _prompt_int("请输入单店全职员工数(人)")
    if staff is None:
        return None, True

    try:
        site = SiteInput(
            address=addr,
            store_area_sqm=area,
            monthly_rent_cny=rent,
            estimated_daily_cups=cups,
            avg_ticket_cny=ticket,
            full_time_staff_count=staff,
        )
        return site, False
    except ValidationError as e:
        print("当前数值不满足业务规则（例如面积、客单价须大于 0，人数与租金不能为负）。")
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "")
            if loc:
                print(f"  · {loc}: {msg}")
            else:
                print(f"  · {msg}")
        print("请从新的一轮开始，逐项核对后再试。\n")
        return None, False


def main() -> None:
    _enable_windows_ansi()
    print("连锁门店选址评估（输入 exit 可随时退出程序）\n")

    while True:
        site, user_exit = _build_site_input()
        if user_exit:
            print("已退出。")
            break
        if site is None:
            continue

        result = calculate_site_roi(site)
        margin = result.estimated_net_margin_pct
        print(f"[检测到周边竞品数量] {result.competitor_count}")
        print(f"【全网实时情报】\n{result.market_intelligence}\n")
        print(f"【拓店部意见】\n{result.promoter_opinion}\n")
        print(f"【风控部意见】\n{result.critic_opinion}\n")

        if result.expansion_approved:
            print(
                f"{_GREEN}【准予拓店】{_RESET} "
                f"预估净利润率：{margin:.2f}%"
            )
        else:
            reason = result.risk_block_reason or "未提供拦截原因"
            print(f"{_RED}【拒绝拓店】{_RESET} {reason}")

        try:
            report_path = save_report(result, site_input=site)
            print(f"评估报告已归档至 ./evaluations/{report_path.name}")
        except OSError:
            print("评估报告写入失败，请检查 evaluations 目录权限或磁盘空间。")
        print()


if __name__ == "__main__":
    main()
