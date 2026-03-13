"""Calendar management for the Montferrand booking agent.

Stores calendar events as individual ``.ics`` files in a vdir_ directory
structure — one directory per tenant, one file per event.  This is the
same layout used by khal, vdirsyncer, and CalDAV servers, so tenants
can later sync their calendars to Google Calendar or any CalDAV service
with zero code changes.

Directory layout::

    $MONTFERRAND_DATA_DIR/calendars/{tenant_hash}/
        {uid}.ics
        {uid}.ics

.. _vdir: https://vdirsyncer.readthedocs.io/en/latest/vdir.html
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

import icalendar
from zoneinfo import ZoneInfo

from montferrand_agent.config import calendars_dir
from montferrand_agent.tenant import phone_to_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TZ = ZoneInfo("America/Montreal")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def _tenant_calendar_dir(twilio_number: str) -> Path:
    """Return (and create if needed) the vdir for a tenant's calendar."""
    d = calendars_dir() / phone_to_filename(twilio_number)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _parse_dt(date_str: str, time_str: str) -> datetime:
    """Combine an ISO date and HH:MM time into a tz-aware datetime.

    Args:
        date_str: ISO format date, e.g. ``'2026-03-16'``.
        time_str: Time in ``HH:MM`` format, e.g. ``'09:00'``.
    """
    d = date.fromisoformat(date_str)
    parts = time_str.split(":")
    t = time(int(parts[0]), int(parts[1]))
    return datetime.combine(d, t, tzinfo=_TZ)


def _dt_to_str(dt: datetime) -> str:
    """Format a datetime as ``'YYYY-MM-DD HH:MM'``."""
    return dt.strftime("%Y-%m-%d %H:%M")


def _ensure_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (default to _TZ)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_TZ)
    return dt


# ---------------------------------------------------------------------------
# .ics file helpers
# ---------------------------------------------------------------------------


def _build_ics(
    uid: str,
    start: datetime,
    end: datetime,
    summary: str,
    description: str,
) -> bytes:
    """Build a minimal VCALENDAR with one VEVENT."""
    cal = icalendar.Calendar()
    cal.add("prodid", "-//Montferrand Agent//EN")
    cal.add("version", "2.0")

    event = icalendar.Event()
    event.add("uid", uid)
    event.add("dtstart", start)
    event.add("dtend", end)
    event.add("summary", summary)
    event.add("description", description)
    event.add("dtstamp", datetime.now(tz=_TZ))
    event.add("created", datetime.now(tz=_TZ))

    cal.add_component(event)
    return cal.to_ical()


def _parse_ics(data: bytes) -> dict | None:
    """Parse a ``.ics`` file and return event fields, or None on error.

    Returns a dict with keys: uid, start, end, summary, description.
    """
    try:
        cal = icalendar.Calendar.from_ical(data)
    except Exception:
        return None

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        uid = str(component.get("uid", ""))
        dt_start = component.get("dtstart")
        dt_end = component.get("dtend")

        if not uid or dt_start is None:
            return None

        start = dt_start.dt if hasattr(dt_start, "dt") else dt_start
        # Handle date-only (all-day) events
        if isinstance(start, date) and not isinstance(start, datetime):
            start = datetime.combine(start, time(0, 0), tzinfo=_TZ)
        start = _ensure_aware(start)

        if dt_end is not None:
            end = dt_end.dt if hasattr(dt_end, "dt") else dt_end
            if isinstance(end, date) and not isinstance(end, datetime):
                end = datetime.combine(end, time(23, 59), tzinfo=_TZ)
            end = _ensure_aware(end)
        else:
            end = start + timedelta(hours=1)

        summary = str(component.get("summary", ""))
        description = str(component.get("description", ""))

        return {
            "uid": uid,
            "start": start,
            "end": end,
            "summary": summary,
            "description": description,
        }

    return None


# ---------------------------------------------------------------------------
# Disk reads
# ---------------------------------------------------------------------------


def _read_all_events(twilio_number: str) -> list[dict]:
    """Read and parse all .ics files for a tenant."""
    vdir = _tenant_calendar_dir(twilio_number)
    events: list[dict] = []

    for ics_path in vdir.glob("*.ics"):
        data = ics_path.read_bytes()
        event = _parse_ics(data)
        if event is not None:
            events.append(event)
        else:
            logger.warning("Skipping unparseable .ics file: %s", ics_path)

    return events


def _has_overlap(
    events: list[dict],
    start: datetime,
    end: datetime,
    exclude_uid: str | None = None,
) -> dict | None:
    """Return the first overlapping event, or None if no overlap."""
    for ev in events:
        if ev["uid"] == exclude_uid:
            continue
        if ev["start"] < end and ev["end"] > start:
            return ev
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reset_calendar(twilio_number: str) -> int:
    """Delete all calendar events for a tenant.

    Removes the tenant's vdir directory.
    Returns the number of ``.ics`` files deleted.
    """
    import shutil

    vdir = calendars_dir() / phone_to_filename(twilio_number)

    count = 0
    if vdir.exists():
        count = sum(1 for f in vdir.iterdir() if f.suffix == ".ics")
        shutil.rmtree(vdir)

    return count


def list_events(twilio_number: str, from_date: str, to_date: str) -> str:
    """List all booked events in a date range.

    Returns a JSON string with event details, or a message if no events.
    """
    try:
        range_start = date.fromisoformat(from_date)
        range_end = date.fromisoformat(to_date)
    except ValueError as exc:
        return f"ERROR: Invalid date format: {exc}. Use ISO format like '2026-03-16'."

    all_events = _read_all_events(twilio_number)

    matches = []
    for ev in all_events:
        ev_start_date = ev["start"].date()
        ev_end_date = ev["end"].date()
        if ev_start_date <= range_end and ev_end_date >= range_start:
            matches.append(
                {
                    "uid": ev["uid"],
                    "start": _dt_to_str(ev["start"]),
                    "end": _dt_to_str(ev["end"]),
                    "summary": ev["summary"],
                    "description": ev["description"],
                }
            )

    if not matches:
        return "No events found in this date range."

    # Sort by start time
    matches.sort(key=lambda e: e["start"])
    return json.dumps(matches, ensure_ascii=False, indent=2)


def create_event(
    twilio_number: str,
    date_str: str,
    start_time: str,
    end_time: str,
    summary: str,
    description: str = "",
) -> str:
    """Book a new service call on the calendar.

    Returns a confirmation string or a conflict error.
    """
    try:
        start = _parse_dt(date_str, start_time)
        end = _parse_dt(date_str, end_time)
    except (ValueError, IndexError) as exc:
        return (
            f"ERROR: Invalid date/time format: {exc}. "
            f"Use date like '2026-03-16' and time like '09:00'."
        )

    if end <= start:
        return "ERROR: End time must be after start time."

    all_events = _read_all_events(twilio_number)

    # Overlap check
    conflict = _has_overlap(all_events, start, end)
    if conflict is not None:
        return (
            f"CONFLICT: This time slot overlaps with an existing booking: "
            f"'{conflict['summary']}' on {_dt_to_str(conflict['start'])} to "
            f"{_dt_to_str(conflict['end'])}. Please choose a different time."
        )

    # Create the event
    uid = uuid.uuid4().hex
    ics_data = _build_ics(uid, start, end, summary, description)

    vdir = _tenant_calendar_dir(twilio_number)
    ics_path = vdir / f"{uid}.ics"
    ics_path.write_bytes(ics_data)

    return (
        f"BOOKED: Event '{summary}' created on {date_str} from "
        f"{start_time} to {end_time}. UID: {uid}"
    )


def delete_event(twilio_number: str, uid: str) -> str:
    """Cancel a booked service call by its UID.

    Returns a confirmation string or an error if not found.
    """
    all_events = _read_all_events(twilio_number)
    ev = next((e for e in all_events if e["uid"] == uid), None)

    if ev is None:
        return f"ERROR: No event found with UID '{uid}'."

    summary = ev["summary"]
    date_str = ev["start"].strftime("%Y-%m-%d")

    # Delete the file
    vdir = _tenant_calendar_dir(twilio_number)
    ics_path = vdir / f"{uid}.ics"
    if ics_path.exists():
        ics_path.unlink()

    return f"DELETED: Event '{summary}' on {date_str} has been cancelled."


def modify_event(
    twilio_number: str,
    uid: str,
    date_str: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    summary: str | None = None,
    description: str | None = None,
) -> str:
    """Reschedule or update a booked service call.

    Only non-None fields are updated.  Returns a confirmation string,
    a conflict error, or a not-found error.
    """
    all_events = _read_all_events(twilio_number)
    ev = next((e for e in all_events if e["uid"] == uid), None)

    if ev is None:
        return f"ERROR: No event found with UID '{uid}'."

    # Compute new values (fall back to existing)
    new_summary = summary if summary is not None else ev["summary"]
    new_description = description if description is not None else ev["description"]

    # Date/time: only recompute if any date/time field was provided
    time_changed = any(x is not None for x in (date_str, start_time, end_time))

    if time_changed:
        d = date_str if date_str is not None else ev["start"].strftime("%Y-%m-%d")
        st = start_time if start_time is not None else ev["start"].strftime("%H:%M")
        et = end_time if end_time is not None else ev["end"].strftime("%H:%M")

        try:
            new_start = _parse_dt(d, st)
            new_end = _parse_dt(d, et)
        except (ValueError, IndexError) as exc:
            return (
                f"ERROR: Invalid date/time format: {exc}. "
                f"Use date like '2026-03-16' and time like '09:00'."
            )

        if new_end <= new_start:
            return "ERROR: End time must be after start time."

        # Overlap check (exclude self)
        conflict = _has_overlap(all_events, new_start, new_end, exclude_uid=uid)
        if conflict is not None:
            return (
                f"CONFLICT: This time slot overlaps with an existing booking: "
                f"'{conflict['summary']}' on {_dt_to_str(conflict['start'])} to "
                f"{_dt_to_str(conflict['end'])}. Please choose a different time."
            )
    else:
        new_start = ev["start"]
        new_end = ev["end"]

    # Write updated .ics
    ics_data = _build_ics(uid, new_start, new_end, new_summary, new_description)
    vdir = _tenant_calendar_dir(twilio_number)
    ics_path = vdir / f"{uid}.ics"
    ics_path.write_bytes(ics_data)

    return (
        f"UPDATED: Event '{new_summary}' now scheduled for "
        f"{new_start.strftime('%Y-%m-%d')} {new_start.strftime('%H:%M')}-"
        f"{new_end.strftime('%H:%M')}. UID: {uid}"
    )
