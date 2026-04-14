"""评估报告邮件服务：将结果以 HTML 发送给管理者。"""

from __future__ import annotations

import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import streamlit as st

_SMTP_HOST_KEY = "SMTP_HOST"
_SMTP_PORT_KEY = "SMTP_PORT"
_SMTP_USER_KEY = "SMTP_USER"
_SMTP_PASSWORD_KEY = "SMTP_PASSWORD"
_SMTP_FROM_KEY = "SMTP_FROM"
_SMTP_TO_KEY = "SMTP_TO"
_SMTP_USE_TLS_KEY = "SMTP_USE_TLS"


def _to_plain_dict(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if hasattr(data, "model_dump"):
        return data.model_dump()  # type: ignore[no-any-return]
    if hasattr(data, "dict"):
        return data.dict()  # type: ignore[no-any-return]
    return {}


def _pick(d: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "1", "yes", "y", "是"):
            return True
        if t in ("false", "0", "no", "n", "否"):
            return False
    return default


def _assessment_fields(assessment_data: Any) -> dict[str, Any]:
    d = _to_plain_dict(assessment_data)
    address = str(_pick(d, "address", "地址", default="未知地址")).strip() or "未知地址"
    approved = _as_bool(_pick(d, "expansion_approved", default=False), default=False)
    block_reason = str(
        _pick(d, "risk_block_reason", "拦截原因", "核心结论", default="")
    ).strip()
    decision = "准予拓店" if approved else (block_reason or "红线拦截")
    margin_pct = _as_float(_pick(d, "estimated_net_margin_pct", "利润率%", default=0.0))
    roi_pct = _as_float(_pick(d, "roi_pct", "ROI", "roi", default=margin_pct))
    revenue = _as_float(_pick(d, "estimated_monthly_revenue_cny", "预估月营业额", default=0.0))
    rent = _as_float(_pick(d, "monthly_rent_cny", "月租金", default=0.0))
    rent_sales_ratio = (rent / revenue * 100.0) if revenue > 0 else 0.0
    promoter = str(_pick(d, "promoter_opinion", "拓店部意见", default="（无）"))
    critic = str(_pick(d, "critic_opinion", "风控部意见", default="（无）"))
    return {
        "address": address,
        "approved": approved,
        "decision": decision,
        "block_reason": block_reason,
        "margin_pct": margin_pct,
        "roi_pct": roi_pct,
        "rent_sales_ratio": rent_sales_ratio,
        "promoter": promoter,
        "critic": critic,
    }


def _subject(info: dict[str, Any]) -> str:
    return f"【柠季选址告警】{info['address']} - {info['decision']}"


def _html_body(info: dict[str, Any]) -> str:
    block_html = ""
    if not info["approved"]:
        reason = html.escape(info["block_reason"] or "触发红线规则")
        block_html = (
            "<p style='margin-top:12px;'>"
            "<span style='color:#c62828;font-weight:700;'>"
            f"红线拦截原因：{reason}"
            "</span></p>"
        )
    return f"""
    <html>
      <body style="font-family: Arial, Helvetica, sans-serif; color: #222;">
        <h3 style="margin-bottom: 8px;">选址评估自动报告</h3>
        <p><b>地址：</b>{html.escape(str(info["address"]))}</p>
        <p><b>最终建议：</b>{html.escape(str(info["decision"]))}</p>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; margin-top: 10px;">
          <tr style="background: #f6f8fa;">
            <th align="left">核心财务指标</th>
            <th align="left">数值</th>
          </tr>
          <tr>
            <td>ROI</td>
            <td>{info["roi_pct"]:.2f}%</td>
          </tr>
          <tr>
            <td>利润率</td>
            <td>{info["margin_pct"]:.2f}%</td>
          </tr>
          <tr>
            <td>租售比</td>
            <td>{info["rent_sales_ratio"]:.2f}%</td>
          </tr>
        </table>
        <h4 style="margin-top: 16px; margin-bottom: 6px;">拓店部意见</h4>
        <div style="white-space: pre-wrap; line-height: 1.5;">{html.escape(str(info["promoter"]))}</div>
        <h4 style="margin-top: 16px; margin-bottom: 6px;">风控部意见</h4>
        <div style="white-space: pre-wrap; line-height: 1.5;">{html.escape(str(info["critic"]))}</div>
        {block_html}
      </body>
    </html>
    """


def send_assessment_email(assessment_data: Any) -> bool:
    """
    发送评估报告邮件（HTML）。
    安全性：仅使用 st.secrets 读取 SMTP 配置；失败返回 False，不影响页面展示。
    """
    try:
        smtp_host = str(st.secrets.get(_SMTP_HOST_KEY, "")).strip()
        smtp_port = int(st.secrets.get(_SMTP_PORT_KEY, 587))
        smtp_user = str(st.secrets.get(_SMTP_USER_KEY, "")).strip()
        smtp_password = str(st.secrets.get(_SMTP_PASSWORD_KEY, "")).strip()
        smtp_from = str(st.secrets.get(_SMTP_FROM_KEY, smtp_user)).strip()
        smtp_to_raw = str(st.secrets.get(_SMTP_TO_KEY, "")).strip()
        use_tls = _as_bool(st.secrets.get(_SMTP_USE_TLS_KEY, True), default=True)

        recipients = [x.strip() for x in smtp_to_raw.split(",") if x.strip()]
        if not smtp_host or not smtp_user or not smtp_password or not recipients:
            st.error("邮件发送失败：SMTP 配置不完整，请检查 st.secrets。")
            return False

        info = _assessment_fields(assessment_data)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = _subject(info)
        msg["From"] = smtp_from
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(_html_body(info), "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, recipients, msg.as_string())
        return True
    except Exception as e:
        st.error(f"邮件发送失败：{e}")
        return False

