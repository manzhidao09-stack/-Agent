"""评估结果企业通知服务：通过 Webhook 推送到管理侧。"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

_WEBHOOK_SECRET_KEY = "ENTERPRISE_WEBHOOK_URL"
_TIMEOUT_SEC = 10


def _webhook_url() -> str:
    """仅从 Streamlit secrets 读取 Webhook URL，严禁硬编码。"""
    return str(st.secrets.get(_WEBHOOK_SECRET_KEY, "")).strip()


def send_enterprise_notification(summary_text: str) -> bool:
    """
    向企业 Webhook 发送评估摘要。
    返回 True 表示发送成功，False 表示失败（不抛异常，不中断主流程）。
    """
    url = _webhook_url()
    if not url:
        return False

    payload: dict[str, Any] = {
        "text": summary_text,
        "msg_type": "text",
        "content": {"text": summary_text},
    }
    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT_SEC)
        return 200 <= resp.status_code < 300
    except Exception:
        return False

