from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from montferrand_agent.crm import TenantCrmMissingError, get_tenant_crm
from montferrand_agent.plumber_router import job_card_token_ttl_hours, job_card_url_for

NEXT_WORK_ITEM_COMMANDS = {
    "next",
    "what's next",
    "whats next",
    "what is next",
    "prochain",
    "prochaine",
    "suivant",
    "ensuite",
}


class NextWorkItemResult(BaseModel):
    success: bool = Field(description="Whether the next work-item lookup succeeded.")
    has_upcoming_job: bool = Field(
        description="Whether there is an upcoming active service call."
    )
    message: str = Field(default="")
    job_id: int | None = None
    card_url: str = ""
    customer_name: str = ""
    service_location: str = ""
    issue_summary: str = ""
    scheduled_start: str | None = None
    scheduled_end: str | None = None


def normalize_boss_command(text: str) -> str:
    return " ".join(text.lower().split())


def _format_job_window(start_iso: str | None, end_iso: str | None) -> str:
    if not start_iso:
        return "schedule pending"

    try:
        start_dt = datetime.fromisoformat(start_iso)
        if end_iso:
            end_dt = datetime.fromisoformat(end_iso)
            return (
                f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%H:%M')}"
            )
        return start_dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        if end_iso:
            return f"{start_iso} to {end_iso}"
        return start_iso


async def get_next_work_item(twilio_number: str) -> NextWorkItemResult:
    try:
        crm = get_tenant_crm(twilio_number)
    except TenantCrmMissingError:
        return NextWorkItemResult(
            success=False,
            has_upcoming_job=False,
            message="Le dossier CRM de ce tenant n'est pas disponible pour le moment.",
        )

    next_job = await crm.get_next_open_job()
    if next_job is None or not next_job.success or next_job.job_id is None:
        return NextWorkItemResult(
            success=True,
            has_upcoming_job=False,
            message="Aucun rendez-vous actif a venir pour le moment.",
        )

    token = await crm.issue_job_token(
        next_job.job_id,
        ttl_hours=job_card_token_ttl_hours(),
    )

    try:
        link = job_card_url_for(twilio_number, token)
    except RuntimeError:
        link = ""

    summary = (
        f"Next: {_format_job_window(next_job.scheduled_start, next_job.scheduled_end)}, "
        f"{next_job.customer_name or 'Unknown customer'}, "
        f"{next_job.service_location or 'address missing'}. "
        f"Issue: {next_job.issue_summary or 'service call'}."
    )
    message = f"{summary} Open card: {link}" if link else summary
    return NextWorkItemResult(
        success=True,
        has_upcoming_job=True,
        message=message,
        job_id=next_job.job_id,
        card_url=link,
        customer_name=next_job.customer_name,
        service_location=next_job.service_location,
        issue_summary=next_job.issue_summary,
        scheduled_start=next_job.scheduled_start,
        scheduled_end=next_job.scheduled_end,
    )


async def maybe_handle_next_work_item_command(
    twilio_number: str,
    body: str,
    *,
    is_boss: bool,
) -> str | None:
    if not is_boss:
        return None
    if normalize_boss_command(body) not in NEXT_WORK_ITEM_COMMANDS:
        return None
    result = await get_next_work_item(twilio_number)
    return result.message
