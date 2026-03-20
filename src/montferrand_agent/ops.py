"""Operations store, readiness checks, and incident queries."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from montferrand_agent.config import (
    calendars_dir,
    conversations_dir,
    data_dir,
    ops_db_path,
    tenant_db_root_dir,
    tenants_dir,
)
from montferrand_agent.llm_backend import resolve_backend

_UNSET = object()


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class OpsMessageRecord:
    message_id: str
    message_sid: str | None
    conversation_id: str
    twilio_number: str
    from_number: str
    is_boss: bool | None
    inbound_body: str
    last_stage: str
    reply_body: str | None
    error_text: str | None
    outbound_message_sid: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class OpsEventRecord:
    id: int
    message_id: str
    event_kind: str
    summary: str
    details: dict[str, Any]
    created_at: str


def new_message_id(message_sid: str | None = None) -> str:
    """Return a durable internal message identifier."""

    if message_sid:
        return message_sid
    return f"local-{uuid.uuid4().hex}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    path = ops_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sms_messages (
            message_id TEXT PRIMARY KEY,
            message_sid TEXT,
            conversation_id TEXT NOT NULL,
            twilio_number TEXT NOT NULL,
            from_number TEXT NOT NULL,
            is_boss INTEGER,
            inbound_body TEXT NOT NULL,
            last_stage TEXT NOT NULL,
            reply_body TEXT,
            error_text TEXT,
            outbound_message_sid TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sms_messages_sid
            ON sms_messages(message_sid);

        CREATE INDEX IF NOT EXISTS idx_sms_messages_outbound_sid
            ON sms_messages(outbound_message_sid);

        CREATE INDEX IF NOT EXISTS idx_sms_messages_lookup
            ON sms_messages(twilio_number, from_number, updated_at DESC);

        CREATE TABLE IF NOT EXISTS sms_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(message_id) REFERENCES sms_messages(message_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sms_events_message_id
            ON sms_events(message_id, id);
        """
    )


def ensure_ops_db() -> Path:
    """Ensure the ops SQLite database exists and return its path."""

    conn = _connect()
    conn.close()
    return ops_db_path()


def _append_event(
    conn: sqlite3.Connection,
    message_id: str,
    event_kind: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO sms_events(message_id, event_kind, summary, details_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (message_id, event_kind, summary, json.dumps(details or {}), now),
    )


def _update_message(
    conn: sqlite3.Connection,
    message_id: str,
    *,
    last_stage: str | object = _UNSET,
    is_boss: bool | None | object = _UNSET,
    reply_body: str | None | object = _UNSET,
    error_text: str | None | object = _UNSET,
    outbound_message_sid: str | None | object = _UNSET,
) -> None:
    sets = ["updated_at = ?"]
    values: list[Any] = [_utc_now_iso()]

    if last_stage is not _UNSET:
        sets.append("last_stage = ?")
        values.append(last_stage)
    if is_boss is not _UNSET:
        sets.append("is_boss = ?")
        if is_boss is None:
            values.append(None)
        else:
            values.append(int(cast(bool, is_boss)))
    if reply_body is not _UNSET:
        sets.append("reply_body = ?")
        values.append(reply_body)
    if error_text is not _UNSET:
        sets.append("error_text = ?")
        values.append(error_text)
    if outbound_message_sid is not _UNSET:
        sets.append("outbound_message_sid = ?")
        values.append(outbound_message_sid)

    values.append(message_id)
    conn.execute(
        f"UPDATE sms_messages SET {', '.join(sets)} WHERE message_id = ?",
        values,
    )


def record_inbound_received(
    *,
    message_id: str,
    message_sid: str | None,
    conversation_id: str,
    twilio_number: str,
    from_number: str,
    body: str,
) -> None:
    """Persist the inbound webhook receipt before background processing."""

    now = _utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sms_messages(
                message_id,
                message_sid,
                conversation_id,
                twilio_number,
                from_number,
                is_boss,
                inbound_body,
                last_stage,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?, 'received', ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                message_sid = COALESCE(sms_messages.message_sid, excluded.message_sid)
            """,
            (
                message_id,
                message_sid,
                conversation_id,
                twilio_number,
                from_number,
                body,
                now,
                now,
            ),
        )
        _append_event(
            conn,
            message_id,
            "received",
            "Inbound Twilio webhook received.",
            {
                "message_sid": message_sid,
                "conversation_id": conversation_id,
                "twilio_number": twilio_number,
                "from_number": from_number,
            },
        )


def mark_duplicate(message_id: str, message_sid: str) -> None:
    with _connect() as conn:
        _update_message(conn, message_id, last_stage="duplicate")
        _append_event(
            conn,
            message_id,
            "duplicate",
            "Duplicate MessageSid skipped.",
            {"message_sid": message_sid},
        )


def mark_missing_tenant(message_id: str, twilio_number: str) -> None:
    with _connect() as conn:
        _update_message(conn, message_id, last_stage="tenant_missing")
        _append_event(
            conn,
            message_id,
            "tenant_missing",
            "No tenant config exists for the inbound Twilio number.",
            {"twilio_number": twilio_number},
        )


def mark_processing_started(message_id: str, *, is_boss: bool) -> None:
    with _connect() as conn:
        _update_message(
            conn, message_id, last_stage="processing_started", is_boss=is_boss
        )
        _append_event(
            conn,
            message_id,
            "processing_started",
            "Background SMS processing started.",
            {"is_boss": is_boss},
        )


def record_processing_trace(
    message_id: str,
    *,
    event_kind: str,
    summary: str,
    details: dict[str, Any],
) -> None:
    with _connect() as conn:
        _append_event(conn, message_id, event_kind, summary, details)


def mark_processing_succeeded(message_id: str, reply_body: str) -> None:
    with _connect() as conn:
        _update_message(
            conn,
            message_id,
            last_stage="processing_succeeded",
            reply_body=reply_body,
            error_text=None,
        )
        _append_event(
            conn,
            message_id,
            "processing_succeeded",
            "Agent produced a reply.",
            {"reply_preview": reply_body[:160]},
        )


def mark_processing_failed(message_id: str, error_text: str, reply_body: str) -> None:
    with _connect() as conn:
        _update_message(
            conn,
            message_id,
            last_stage="processing_failed",
            reply_body=reply_body,
            error_text=error_text,
        )
        _append_event(
            conn,
            message_id,
            "processing_failed",
            "Agent processing failed; fallback reply selected.",
            {"error_text": error_text, "reply_preview": reply_body[:160]},
        )


def mark_outbound_attempted(message_id: str, reply_body: str) -> None:
    with _connect() as conn:
        _update_message(
            conn, message_id, last_stage="outbound_attempted", reply_body=reply_body
        )
        _append_event(
            conn,
            message_id,
            "outbound_attempted",
            "Attempting outbound Twilio SMS send.",
            {"reply_preview": reply_body[:160]},
        )


def mark_outbound_accepted(message_id: str, outbound_message_sid: str | None) -> None:
    with _connect() as conn:
        _update_message(
            conn,
            message_id,
            last_stage="outbound_accepted",
            outbound_message_sid=outbound_message_sid,
        )
        _append_event(
            conn,
            message_id,
            "outbound_accepted",
            "Twilio accepted the outbound SMS request.",
            {"outbound_message_sid": outbound_message_sid},
        )


def mark_outbound_failed(message_id: str, error_text: str) -> None:
    with _connect() as conn:
        _update_message(
            conn, message_id, last_stage="outbound_failed", error_text=error_text
        )
        _append_event(
            conn,
            message_id,
            "outbound_failed",
            "Outbound Twilio SMS send failed.",
            {"error_text": error_text},
        )


def mark_delivery_status(
    outbound_message_sid: str,
    delivery_status: str,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT message_id FROM sms_messages WHERE outbound_message_sid = ?",
            (outbound_message_sid,),
        ).fetchone()
        if row is None:
            return False

        message_id = str(row["message_id"])
        normalized = delivery_status.strip().lower() or "unknown"
        error_text = error_message or error_code
        _update_message(
            conn,
            message_id,
            last_stage=f"delivery_{normalized}",
            error_text=error_text,
        )
        _append_event(
            conn,
            message_id,
            "delivery_status",
            f"Twilio delivery status updated to {normalized}.",
            {
                "outbound_message_sid": outbound_message_sid,
                "delivery_status": normalized,
                "error_code": error_code,
                "error_message": error_message,
            },
        )
        return True


def _message_from_row(row: sqlite3.Row) -> OpsMessageRecord:
    raw_is_boss = row["is_boss"]
    return OpsMessageRecord(
        message_id=str(row["message_id"]),
        message_sid=row["message_sid"],
        conversation_id=str(row["conversation_id"]),
        twilio_number=str(row["twilio_number"]),
        from_number=str(row["from_number"]),
        is_boss=None if raw_is_boss is None else bool(raw_is_boss),
        inbound_body=str(row["inbound_body"]),
        last_stage=str(row["last_stage"]),
        reply_body=row["reply_body"],
        error_text=row["error_text"],
        outbound_message_sid=row["outbound_message_sid"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _event_from_row(row: sqlite3.Row) -> OpsEventRecord:
    return OpsEventRecord(
        id=int(row["id"]),
        message_id=str(row["message_id"]),
        event_kind=str(row["event_kind"]),
        summary=str(row["summary"]),
        details=json.loads(row["details_json"]),
        created_at=str(row["created_at"]),
    )


def get_message_timeline(
    *,
    message_sid: str | None = None,
    message_id: str | None = None,
) -> tuple[OpsMessageRecord, list[OpsEventRecord]] | None:
    """Return a message record and ordered timeline events."""

    if not message_sid and not message_id:
        raise ValueError("message_sid or message_id is required")

    query = (
        "SELECT * FROM sms_messages WHERE message_sid = ? "
        "ORDER BY updated_at DESC LIMIT 1"
    )
    value = message_sid
    if message_id is not None:
        query = "SELECT * FROM sms_messages WHERE message_id = ? LIMIT 1"
        value = message_id

    with _connect() as conn:
        row = conn.execute(query, (value,)).fetchone()
        if row is None:
            return None

        message = _message_from_row(row)
        events = [
            _event_from_row(event_row)
            for event_row in conn.execute(
                "SELECT * FROM sms_events WHERE message_id = ? ORDER BY id ASC",
                (message.message_id,),
            ).fetchall()
        ]
        return message, events


def find_messages(
    *,
    twilio_number: str | None = None,
    from_number: str | None = None,
    limit: int = 10,
) -> list[OpsMessageRecord]:
    """Return recent inbound messages filtered by tenant/customer."""

    clauses: list[str] = []
    params: list[Any] = []
    if twilio_number:
        clauses.append("twilio_number = ?")
        params.append(twilio_number)
    if from_number:
        clauses.append("from_number = ?")
        params.append(from_number)

    query = "SELECT * FROM sms_messages"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        return [
            _message_from_row(row) for row in conn.execute(query, params).fetchall()
        ]


def run_readiness_checks() -> list[ReadinessCheck]:
    """Run local dependency and storage checks for operator diagnostics."""

    checks: list[ReadinessCheck] = []

    try:
        root = data_dir()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=root, prefix="readyz-", delete=True
        ) as probe:
            probe.write(b"ok")
            probe.flush()
        checks.append(ReadinessCheck("data_dir", True, f"Writable: {root}"))
    except Exception as exc:
        checks.append(ReadinessCheck("data_dir", False, str(exc)))

    for name, directory in (
        ("tenants_dir", tenants_dir),
        ("conversations_dir", conversations_dir),
        ("calendars_dir", calendars_dir),
        ("tenant_db_root_dir", tenant_db_root_dir),
    ):
        try:
            path = directory()
            path.mkdir(parents=True, exist_ok=True)
            checks.append(ReadinessCheck(name, True, str(path)))
        except Exception as exc:
            checks.append(ReadinessCheck(name, False, str(exc)))

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if account_sid and auth_token:
        checks.append(
            ReadinessCheck(
                "twilio_credentials", True, "Outbound and webhook auth configured."
            )
        )
    elif auth_token:
        checks.append(
            ReadinessCheck(
                "twilio_credentials",
                False,
                "Webhook auth configured, but TWILIO_ACCOUNT_SID is missing.",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                "twilio_credentials",
                False,
                "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN are not configured.",
            )
        )

    try:
        backend = resolve_backend()
        checks.append(
            ReadinessCheck(
                "llm_backend",
                True,
                f"{backend.spec.provider}:{backend.spec.model_name}",
            )
        )
    except Exception as exc:
        checks.append(ReadinessCheck("llm_backend", False, str(exc)))

    try:
        path = ensure_ops_db()
        checks.append(ReadinessCheck("ops_db", True, str(path)))
    except Exception as exc:
        checks.append(ReadinessCheck("ops_db", False, str(exc)))

    return checks
