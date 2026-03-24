"""FastAPI webhook server for the Montferrand booking agent.

Endpoints:

    POST /sms            — Twilio incoming SMS webhook
    POST /admin/tenants  — Upsert a tenant configuration (bearer token auth)
    GET  /health         — Health check

Architecture note — async SMS replies:

    The SMS webhook does NOT return the agent's reply inline via TwiML.
    Instead it returns an empty TwiML response immediately (so Twilio
    never times out), and launches a background task that:

    1. Acquires a per-conversation lock (prevents race conditions)
    2. Calls ``process_message()`` (which can take arbitrarily long)
    3. Sends the reply via the Twilio REST API (``_send_sms``)

    If Twilio retries a webhook (e.g., because of a previous timeout or
    5xx), the ``MessageSid`` dedup logic detects the duplicate and skips
    processing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from montferrand_agent.conversation import (
    ConversationTraceEvent,
    ConversationError,
    conversation_key_for_sms,
    process_message,
    reset_tenant,
)
from montferrand_agent.crm import (
    ensure_tenant_crm,
    TenantCrmError,
    provision_tenant_crm,
)
from montferrand_agent.ops import (
    ensure_ops_db,
    mark_delivery_status,
    mark_duplicate,
    mark_missing_tenant,
    mark_outbound_accepted,
    mark_outbound_attempted,
    mark_outbound_failed,
    mark_processing_failed,
    mark_processing_started,
    mark_processing_succeeded,
    new_message_id,
    record_inbound_received,
    record_processing_trace,
    run_readiness_checks,
)
from montferrand_agent.next_work_item import maybe_handle_next_work_item_command
from montferrand_agent.tenant import (
    TenantConfig,
    TenantNotFoundError,
    load_tenant_config,
    save_tenant_config,
    tenant_exists,
)
from montferrand_agent.plumber_router import (
    router as plumber_router,
)

logger = logging.getLogger(__name__)


def _log_ops_event(event: str, **fields: object) -> None:
    """Emit a compact JSON log line for operational events."""

    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# MessageSid deduplication
# ---------------------------------------------------------------------------

_DEDUP_TTL = 60.0  # seconds to remember a MessageSid

_seen_sids: dict[str, float] = {}  # MessageSid -> monotonic timestamp


def _is_duplicate(message_sid: str) -> bool:
    """Return True if this MessageSid was already seen within the TTL.

    Also prunes expired entries on each call to keep memory bounded.
    """
    now = time.monotonic()

    # Prune expired entries
    expired = [sid for sid, ts in _seen_sids.items() if now - ts > _DEDUP_TTL]
    for sid in expired:
        del _seen_sids[sid]

    if message_sid in _seen_sids:
        return True

    _seen_sids[message_sid] = now
    return False


# ---------------------------------------------------------------------------
# In-flight task tracking for graceful shutdown
# ---------------------------------------------------------------------------

_inflight_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _track_task(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """Add a background task to the tracked set; auto-remove on completion."""
    _inflight_tasks.add(task)
    task.add_done_callback(_inflight_tasks.discard)


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Application lifespan — drain in-flight tasks on shutdown."""
    _validate_twilio_webhook_configuration()
    ensure_ops_db()
    yield
    # Graceful shutdown: wait for all background tasks to complete
    if _inflight_tasks:
        logger.info(
            "Waiting for %d in-flight task(s) to complete...",
            len(_inflight_tasks),
        )
        await asyncio.gather(*_inflight_tasks, return_exceptions=True)
        logger.info("All in-flight tasks completed.")


app = FastAPI(
    title="Montferrand Agent",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)
app.include_router(plumber_router)


# ---------------------------------------------------------------------------
# Twilio REST API — outbound SMS via httpx
# ---------------------------------------------------------------------------


def _status_callback_url() -> str | None:
    """Return the configured Twilio status callback URL, if any."""

    url = os.environ.get("MONTFERRAND_TWILIO_STATUS_CALLBACK_URL", "").strip()
    return url or None


async def _send_sms(to: str, from_: str, body: str) -> str | None:
    """Send an SMS via the Twilio REST API.

    Uses httpx (async) to POST to the Twilio Messages endpoint.

    Raises:
        RuntimeError: If TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN is not set.
        httpx.HTTPStatusError: If the Twilio API returns an error.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()

    if not account_sid or not auth_token:
        raise RuntimeError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must both be set "
            "to send outbound SMS."
        )

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data = {"To": to, "From": from_, "Body": body}
    callback_url = _status_callback_url()
    if callback_url:
        data["StatusCallback"] = callback_url

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            auth=(account_sid, auth_token),
            data=data,
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("sid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_twiml() -> Response:
    """Return an empty TwiML response (no message)."""
    twiml = MessagingResponse()
    return Response(content=str(twiml), media_type="application/xml")


def _twiml_response(text: str) -> Response:
    """Build a TwiML XML response containing a single SMS message."""
    twiml = MessagingResponse()
    twiml.message(text)
    return Response(content=str(twiml), media_type="application/xml")


async def _maybe_handle_plumber_command(
    twilio_number: str,
    body: str,
    *,
    is_boss: bool,
) -> str | None:
    reply = await maybe_handle_next_work_item_command(
        twilio_number,
        body,
        is_boss=is_boss,
    )
    if reply is None:
        return None
    if "Open card:" not in reply:
        logger.warning("Public base URL is not configured; plumber link omitted")
    return reply


# ---------------------------------------------------------------------------
# Twilio signature validation
# ---------------------------------------------------------------------------


def _validate_twilio_webhook_configuration() -> None:
    """Crash on startup unless webhook signature validation is configured."""

    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if token:
        return

    raise RuntimeError("TWILIO_AUTH_TOKEN must be set before starting the server.")


def _get_twilio_validator() -> RequestValidator:
    """Build a Twilio RequestValidator from the configured auth token."""

    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if token:
        return RequestValidator(token)

    raise RuntimeError(
        "TWILIO_AUTH_TOKEN is not configured for webhook signature validation."
    )


async def _validate_twilio_signature(request: Request) -> None:
    """FastAPI dependency that validates the Twilio request signature."""
    try:
        validator = _get_twilio_validator()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    signature = request.headers.get("X-Twilio-Signature", "")
    # Behind a reverse proxy (e.g., Fly.io), request.url uses http://
    # but Twilio signed against the public https:// URL.
    url = str(request.url.replace(scheme="https"))
    form = dict(await request.form())

    if not validator.validate(url, form, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


# ---------------------------------------------------------------------------
# Admin bearer token auth
# ---------------------------------------------------------------------------


async def _validate_admin_token(request: Request) -> None:
    """FastAPI dependency that validates the admin bearer token."""
    expected = os.environ.get("MONTFERRAND_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="MONTFERRAND_ADMIN_TOKEN is not configured on the server",
        )

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = auth[len("Bearer ") :]
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token")


# ---------------------------------------------------------------------------
# POST /sms — Twilio incoming SMS webhook
# ---------------------------------------------------------------------------


async def _handle_sms(
    message_id: str,
    twilio_number: str,
    from_number: str,
    body: str,
    tenant_profile: str,
    *,
    is_boss: bool = False,
) -> None:
    """Background task: process the message and send the reply via Twilio API.

    This runs after the webhook has already returned an empty TwiML to
    Twilio.  Errors are logged and an error SMS is sent to the sender.
    """
    key = conversation_key_for_sms(twilio_number, from_number)
    mark_processing_started(message_id, is_boss=is_boss)
    _log_ops_event(
        "processing_started",
        message_id=message_id,
        conversation_id=key,
        is_boss=is_boss,
        twilio_number=twilio_number,
        from_number=from_number,
    )

    def trace_observer(event: ConversationTraceEvent) -> None:
        details = asdict(event)
        record_processing_trace(
            message_id,
            event_kind=f"trace_{event.kind}",
            summary=event.summary,
            details=details,
        )
        if event.kind in {"warning", "turn_finished"}:
            _log_ops_event(
                f"trace_{event.kind}",
                message_id=message_id,
                conversation_id=key,
                summary=event.summary,
                details=details,
            )

    try:
        helper_reply = await _maybe_handle_plumber_command(
            twilio_number,
            body,
            is_boss=is_boss,
        )
        if helper_reply is not None:
            reply = helper_reply
        else:
            result = await process_message(
                key,
                body,
                tenant_profile=tenant_profile,
                twilio_number=twilio_number,
                is_boss=is_boss,
                customer_phone=None if is_boss else from_number,
                trace_observer=trace_observer,
            )
            reply = result.message
        mark_processing_succeeded(message_id, reply)
        _log_ops_event(
            "processing_succeeded",
            message_id=message_id,
            conversation_id=key,
            reply_preview=reply[:160],
        )
    except ConversationError as exc:
        logger.error("Agent error for %s: %s", key, exc)
        reply = "Une erreur est survenue. Veuillez réessayer dans quelques instants."
        mark_processing_failed(message_id, str(exc), reply)
        _log_ops_event(
            "processing_failed",
            message_id=message_id,
            conversation_id=key,
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error for %s: %s", key, exc)
        reply = "Une erreur est survenue. Veuillez réessayer dans quelques instants."
        mark_processing_failed(message_id, str(exc), reply)
        _log_ops_event(
            "processing_failed_unexpected",
            message_id=message_id,
            conversation_id=key,
            error=str(exc),
        )

    mark_outbound_attempted(message_id, reply)
    try:
        outbound_sid = await _send_sms(to=from_number, from_=twilio_number, body=reply)
        mark_outbound_accepted(message_id, outbound_sid)
        _log_ops_event(
            "outbound_accepted",
            message_id=message_id,
            conversation_id=key,
            outbound_message_sid=outbound_sid,
        )
    except Exception as exc:
        logger.exception("Failed to send SMS reply to %s: %s", from_number, exc)
        mark_outbound_failed(message_id, str(exc))
        _log_ops_event(
            "outbound_failed",
            message_id=message_id,
            conversation_id=key,
            error=str(exc),
        )


@app.post("/sms", dependencies=[Depends(_validate_twilio_signature)])
async def sms_webhook(request: Request) -> Response:
    """Handle an incoming SMS from Twilio.

    Returns an empty TwiML response immediately (no message).  The actual
    agent reply is sent asynchronously via the Twilio REST API in a
    background task — so the LLM can take as long as it needs.
    """
    form = await request.form()

    twilio_number: str = form.get("To", "")  # type: ignore[assignment]
    from_number: str = form.get("From", "")  # type: ignore[assignment]
    body: str = form.get("Body", "")  # type: ignore[assignment]
    message_sid: str = form.get("MessageSid", "")  # type: ignore[assignment]

    if not twilio_number or not from_number:
        raise HTTPException(status_code=400, detail="Missing To or From")

    conversation_id = conversation_key_for_sms(twilio_number, from_number)
    message_id = new_message_id(message_sid or None)
    record_inbound_received(
        message_id=message_id,
        message_sid=message_sid or None,
        conversation_id=conversation_id,
        twilio_number=twilio_number,
        from_number=from_number,
        body=body,
    )
    _log_ops_event(
        "sms_received",
        message_id=message_id,
        message_sid=message_sid or None,
        conversation_id=conversation_id,
        twilio_number=twilio_number,
        from_number=from_number,
    )

    # Dedup: skip if we already saw this MessageSid
    if message_sid and _is_duplicate(message_sid):
        logger.info("Duplicate MessageSid %s — skipping", message_sid)
        mark_duplicate(message_id, message_sid)
        _log_ops_event(
            "sms_duplicate",
            message_id=message_id,
            message_sid=message_sid,
            conversation_id=conversation_id,
        )
        return _empty_twiml()

    # Load tenant config (sync and fast — can respond inline on error)
    try:
        config = load_tenant_config(twilio_number)
    except TenantNotFoundError:
        logger.error("No tenant config for %s", twilio_number)
        mark_missing_tenant(message_id, twilio_number)
        _log_ops_event(
            "tenant_missing",
            message_id=message_id,
            twilio_number=twilio_number,
            conversation_id=conversation_id,
        )
        return _twiml_response(
            "Desolé, ce service n'est pas configuré. "
            "Veuillez contacter l'entreprise directement."
        )

    boss = from_number in config.boss_numbers

    # Launch background task and return immediately
    task = asyncio.create_task(
        _handle_sms(
            message_id,
            twilio_number,
            from_number,
            body,
            config.profile,
            is_boss=boss,
        )
    )
    _track_task(task)

    return _empty_twiml()


@app.post("/twilio/status", dependencies=[Depends(_validate_twilio_signature)])
async def twilio_status_callback(request: Request) -> Response:
    """Persist outbound delivery updates from Twilio status callbacks."""

    form = await request.form()
    outbound_message_sid: str = form.get("MessageSid", "")  # type: ignore[assignment]
    delivery_status: str = form.get("MessageStatus", "")  # type: ignore[assignment]
    error_code: str = form.get("ErrorCode", "")  # type: ignore[assignment]
    error_message: str = form.get("ErrorMessage", "")  # type: ignore[assignment]

    if not outbound_message_sid:
        raise HTTPException(status_code=400, detail="Missing MessageSid")

    matched = mark_delivery_status(
        outbound_message_sid,
        delivery_status,
        error_code=error_code or None,
        error_message=error_message or None,
    )
    _log_ops_event(
        "twilio_delivery_status",
        outbound_message_sid=outbound_message_sid,
        delivery_status=delivery_status,
        matched=matched,
        error_code=error_code or None,
    )
    return _empty_twiml()


# ---------------------------------------------------------------------------
# POST /admin/tenants — Upsert tenant config
# ---------------------------------------------------------------------------


class TenantUpsertRequest(BaseModel):
    """Request body for creating/updating a tenant."""

    twilio_number: str
    tenant_profile: str
    boss_numbers: list[str] = []


@app.post(
    "/admin/tenants",
    status_code=201,
    dependencies=[Depends(_validate_admin_token)],
)
async def upsert_tenant(payload: TenantUpsertRequest) -> dict[str, str]:
    """Create or update a tenant's configuration."""
    from montferrand_agent.calendar import ensure_tenant_calendar

    exists = tenant_exists(payload.twilio_number)
    config = TenantConfig(
        phone=payload.twilio_number,
        profile=payload.tenant_profile,
        boss_numbers=payload.boss_numbers,
    )
    ensure_tenant_calendar(payload.twilio_number)
    try:
        if exists:
            crm_path = ensure_tenant_crm(payload.twilio_number)
        else:
            crm_path = provision_tenant_crm(payload.twilio_number)
    except TenantCrmError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    path = save_tenant_config(config)
    logger.info(
        "Tenant upserted: %s -> %s (crm=%s)",
        payload.twilio_number,
        path,
        crm_path,
    )
    return {"status": "ok", "path": str(path), "crm_path": str(crm_path)}


# ---------------------------------------------------------------------------
# DELETE /admin/tenants/{twilio_number}/conversations
# ---------------------------------------------------------------------------


@app.delete(
    "/admin/tenants/{twilio_number}/conversations",
    dependencies=[Depends(_validate_admin_token)],
)
async def delete_tenant_conversations(twilio_number: str) -> dict[str, object]:
    """Delete all conversation data and reset the tenant calendar and CRM."""
    count = reset_tenant(twilio_number)
    logger.info(
        "Reset tenant %s: deleted %d conversation(s) and reset calendar and CRM",
        twilio_number,
        count,
    )
    return {
        "status": "ok",
        "deleted": count,
        "message": (
            f"Deleted {count} conversation(s) and reset the calendar and CRM for "
            f"{twilio_number}."
        ),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def _build_readiness_payload() -> tuple[int, dict[str, object]]:
    checks = run_readiness_checks()
    ok = all(check.ok for check in checks)
    payload = {
        "status": "ok" if ok else "degraded",
        "checks": [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in checks
        ],
    }
    return (200 if ok else 503, payload)


@app.get("/livez")
async def livez() -> dict[str, str]:
    """Liveness probe: process is up."""

    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    """Readiness probe: config, storage, and local dependencies are usable."""

    status_code, payload = _build_readiness_payload()
    return JSONResponse(status_code=status_code, content=payload)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}
