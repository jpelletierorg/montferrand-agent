from __future__ import annotations

import html
import os
from datetime import datetime
from urllib.parse import quote_plus

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from montferrand_agent.crm import (
    JobCardResult,
    TenantCrmMissingError,
    get_tenant_crm_by_key,
)
from montferrand_agent.tenant import phone_to_filename

router = APIRouter()


def _public_base_url() -> str:
    base_url = os.getenv("MONTFERRAND_BASE_URL", "").strip()
    if base_url:
        return base_url.rstrip("/")

    host = os.getenv("MONTFERRAND_HOST", "").strip()
    if host:
        host = host.replace("https://", "").replace("http://", "")
        return f"https://{host.rstrip('/')}"

    raise RuntimeError(
        "Set MONTFERRAND_BASE_URL or MONTFERRAND_HOST to build plumber job-card links."
    )


def job_card_url_for(twilio_number: str, token: str) -> str:
    tenant_key = phone_to_filename(twilio_number)
    return f"{_public_base_url()}/p/{tenant_key}/{token}"


def job_card_token_ttl_hours() -> int:
    raw = os.getenv("MONTFERRAND_JOB_CARD_TOKEN_TTL_HOURS", "72").strip() or "72"
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "MONTFERRAND_JOB_CARD_TOKEN_TTL_HOURS must be an integer."
        ) from exc
    return max(value, 1)


def _fmt_dt(value: str | None) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    return dt.strftime("%Y-%m-%d %H:%M")


def _window_text(card: JobCardResult) -> str:
    start = _fmt_dt(card.scheduled_start)
    end = _fmt_dt(card.scheduled_end)
    if start and end:
        return f"{start} to {end}"
    return start or end or "Schedule pending"


def _status_label(status: str) -> str:
    labels = {
        "booked": "Booked",
        "en_route": "En route",
        "arrived": "Arrived",
        "completed": "Completed",
        "follow_up_needed": "Follow-up needed",
        "cancelled": "Cancelled",
    }
    return labels.get(status, status.replace("_", " ").title())


def _redirect_to_card(tenant_key: str, token: str, notice: str) -> RedirectResponse:
    encoded = quote_plus(notice)
    return RedirectResponse(
        url=f"/p/{tenant_key}/{token}?notice={encoded}",
        status_code=303,
    )


async def _load_job_card(tenant_key: str, token: str) -> JobCardResult | None:
    try:
        crm = get_tenant_crm_by_key(tenant_key)
    except TenantCrmMissingError:
        return None
    return await crm.get_job_card_by_token(token)


def _error_page(message: str, *, status_code: int) -> HTMLResponse:
    safe_message = html.escape(message)
    return HTMLResponse(
        content=(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Job card unavailable</title>"
            "<style>body{font-family:Georgia,serif;background:#f4efe6;color:#1f1a17;"
            "margin:0;padding:24px}main{max-width:640px;margin:0 auto;background:#fff;"
            "border:1px solid #d8c8b8;border-radius:18px;padding:24px;box-shadow:0 10px 30px rgba(0,0,0,.08)}"
            "h1{margin:0 0 12px;font-size:1.5rem}p{line-height:1.5}</style></head>"
            f"<body><main><h1>Job card unavailable</h1><p>{safe_message}</p></main></body></html>"
        ),
        status_code=status_code,
    )


def _render_notes(card: JobCardResult) -> str:
    if not card.recent_notes:
        return "<p class='muted'>No notes yet.</p>"
    items = []
    for note in card.recent_notes:
        items.append(
            "<li>"
            f"<strong>{html.escape(note.author_kind.title())}</strong>"
            f" <span class='muted'>{html.escape(_fmt_dt(note.created_at))}</span>"
            f"<div>{html.escape(note.body)}</div>"
            "</li>"
        )
    return "<ul class='list'>" + "".join(items) + "</ul>"


def _render_history(card: JobCardResult) -> str:
    if not card.recent_jobs:
        return "<p class='muted'>No prior jobs on file.</p>"
    items = []
    for job in card.recent_jobs:
        location = (
            f" - {html.escape(job.service_location)}" if job.service_location else ""
        )
        when = html.escape(_fmt_dt(job.scheduled_start))
        items.append(
            "<li>"
            f"<strong>{when or 'Date unknown'}</strong>"
            f" <span class='pill'>{html.escape(_status_label(job.status))}</span>"
            f"<div>{html.escape(job.issue_summary)}{location}</div>"
            "</li>"
        )
    return "<ul class='list'>" + "".join(items) + "</ul>"


def _job_card_html(
    card: JobCardResult, tenant_key: str, token: str, notice: str = ""
) -> str:
    safe_notice = html.escape(notice)
    customer_name = html.escape(card.customer_name or "Unknown customer")
    customer_phone = html.escape(card.customer_phone)
    service_location = html.escape(card.service_location)
    issue_summary = html.escape(card.issue_summary)
    plumber_notes = html.escape(card.plumber_notes)
    access_notes = html.escape(card.access_notes)
    customer_summary = html.escape(card.customer_summary)
    window_text = html.escape(_window_text(card))
    tel_href = html.escape(f"tel:{card.customer_phone}") if card.customer_phone else ""
    maps_href = html.escape(
        f"https://maps.google.com/?q={quote_plus(card.service_location)}"
    )
    status_label = html.escape(_status_label(card.status))

    notice_block = f"<div class='notice'>{safe_notice}</div>" if notice else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Montferrand Job Card</title>
  <style>
    :root {{
      --paper:#faf6ef;
      --ink:#231c17;
      --muted:#6f655c;
      --line:#dbc8b2;
      --accent:#b55d2c;
      --accent-deep:#8c431a;
      --card:#fffdfa;
      --ok:#1d6b43;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:linear-gradient(180deg,#efe5d6 0%,#f8f4ee 55%,#efe7dc 100%); color:var(--ink); font-family:Georgia,"Iowan Old Style",serif; }}
    main {{ max-width:760px; margin:0 auto; padding:20px 16px 40px; }}
    .hero {{ background:var(--card); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 10px 34px rgba(46,31,17,.08); }}
    h1 {{ margin:0 0 8px; font-size:1.9rem; line-height:1.05; }}
    h2 {{ margin:0 0 10px; font-size:1.05rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
    p {{ margin:0; line-height:1.5; }}
    .grid {{ display:grid; gap:14px; margin-top:16px; }}
    .section {{ background:var(--card); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 8px 24px rgba(46,31,17,.05); }}
    .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#f3e2cf; color:var(--accent-deep); font-size:.9rem; margin-left:8px; }}
    .muted {{ color:var(--muted); }}
    .notice {{ margin:0 0 14px; background:#eef8f1; border:1px solid #b8d8c1; color:var(--ok); padding:12px 14px; border-radius:14px; }}
    .actions {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
    button {{ width:100%; border:0; border-radius:14px; padding:13px 14px; background:var(--accent); color:#fff; font:inherit; font-weight:700; }}
    button.secondary {{ background:#d7c2aa; color:var(--ink); }}
    a.cta {{ display:inline-block; text-decoration:none; padding:12px 14px; border-radius:14px; background:#1f1a17; color:#fff; margin-right:8px; }}
    .list {{ margin:0; padding-left:18px; display:grid; gap:10px; }}
    textarea {{ width:100%; min-height:120px; border:1px solid var(--line); border-radius:14px; padding:12px; font:inherit; background:#fff; }}
    label.checkbox {{ display:flex; gap:8px; align-items:flex-start; margin:12px 0 14px; }}
    @media (max-width: 640px) {{ .actions {{ grid-template-columns:1fr; }} h1 {{ font-size:1.6rem; }} }}
  </style>
</head>
<body>
  <main>
    {notice_block}
    <section class="hero">
      <h1>{customer_name}</h1>
      <p><strong>{window_text}</strong><span class="pill">{status_label}</span></p>
      <p class="muted" style="margin-top:8px">{service_location or "No address recorded."}</p>
    </section>

    <div class="grid">
      <section class="section">
        <h2>Dispatch</h2>
        <p>{issue_summary or "No issue summary recorded."}</p>
        <p class="muted" style="margin-top:8px">{plumber_notes or "No plumber notes recorded yet."}</p>
      </section>

      <section class="section">
        <h2>Customer</h2>
        <p>{customer_phone or "No phone recorded."}</p>
        <p class="muted" style="margin-top:8px">{customer_summary or "No CRM summary yet."}</p>
        <div style="margin-top:14px">
          {f'<a class="cta" href="{tel_href}">Call customer</a>' if tel_href else ""}
          <a class="cta" href="{maps_href}" target="_blank" rel="noreferrer">Open in maps</a>
        </div>
      </section>

      <section class="section">
        <h2>Access notes</h2>
        <p>{access_notes or "No access notes recorded."}</p>
      </section>

      <section class="section">
        <h2>Status</h2>
        <form class="actions" method="post" action="/p/{tenant_key}/{token}/status">
          <button type="submit" name="status" value="en_route">En route</button>
          <button type="submit" name="status" value="arrived">Arrived</button>
          <button class="secondary" type="submit" name="status" value="booked">Reset to booked</button>
          <button class="secondary" type="submit" name="status" value="follow_up_needed">Needs follow-up</button>
        </form>
      </section>

      <section class="section">
        <h2>Closeout</h2>
        <form method="post" action="/p/{tenant_key}/{token}/closeout">
          <textarea name="closeout_text" placeholder="What did you find? What did you do? What should happen next?"></textarea>
          <label class="checkbox"><input type="checkbox" name="follow_up_needed" value="1"> <span>Mark this job as needing follow-up instead of completed.</span></label>
          <button type="submit">Save closeout</button>
        </form>
      </section>

      <section class="section">
        <h2>Recent notes</h2>
        {_render_notes(card)}
      </section>

      <section class="section">
        <h2>History</h2>
        {_render_history(card)}
      </section>
    </div>
  </main>
</body>
</html>"""


@router.get("/p/{tenant_key}/{token}", response_class=HTMLResponse)
async def plumber_job_card(
    tenant_key: str,
    token: str,
    notice: str = "",
) -> HTMLResponse:
    card = await _load_job_card(tenant_key, token)
    if card is None:
        return _error_page("This job link is invalid.", status_code=404)
    if not card.success:
        return _error_page(
            card.message or "This job link is unavailable.", status_code=410
        )
    return HTMLResponse(_job_card_html(card, tenant_key, token, notice))


@router.post("/p/{tenant_key}/{token}/status")
async def update_plumber_job_status(
    tenant_key: str,
    token: str,
    status: str = Form(...),
) -> RedirectResponse:
    card = await _load_job_card(tenant_key, token)
    if card is None or not card.success or card.job_id is None:
        return _redirect_to_card(tenant_key, token, "This job link is no longer valid.")
    crm = get_tenant_crm_by_key(tenant_key)
    await crm.update_job_status(card.job_id, status)
    await crm.add_job_note(
        card.job_id,
        f"Status changed to {_status_label(status)}.",
        author_kind="plumber",
    )
    return _redirect_to_card(
        tenant_key, token, f"Status updated to {_status_label(status)}."
    )


@router.post("/p/{tenant_key}/{token}/closeout")
async def closeout_plumber_job(
    tenant_key: str,
    token: str,
    closeout_text: str = Form(""),
    follow_up_needed: str | None = Form(None),
) -> RedirectResponse:
    card = await _load_job_card(tenant_key, token)
    if card is None or not card.success or card.job_id is None:
        return _redirect_to_card(tenant_key, token, "This job link is no longer valid.")
    crm = get_tenant_crm_by_key(tenant_key)
    await crm.close_job(
        card.job_id,
        closeout_text,
        follow_up_needed=follow_up_needed is not None,
    )
    message = "Closeout saved."
    if follow_up_needed is not None:
        message = "Closeout saved and follow-up flagged."
    return _redirect_to_card(tenant_key, token, message)
