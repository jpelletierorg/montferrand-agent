"""Tenant-scoped calendar management for the Montferrand booking agent.

This module follows the same pattern as the sample agent in
``/Users/jopela/Projects/agents/calendar``: calendar access is wrapped in a
backend object, and each agent run receives exactly one backend instance via
deps. That keeps tool calls isolated to a single tenant calendar.

Calendar storage uses a vdir-compatible layout:

    $MONTFERRAND_DATA_DIR/calendars/{tenant_hash}/
        {uid}.ics
        {uid}.ics
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal, cast

import icalendar
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from montferrand_agent.config import calendars_dir
from montferrand_agent.tenant import phone_to_filename

logger = logging.getLogger(__name__)

_TZ = ZoneInfo("America/Montreal")
_TENANT_LOCKS: dict[str, threading.Lock] = {}
_BLOCK_EVENT_SUMMARY_PREFIX = "BLOCK -"


@dataclass(frozen=True)
class _ServiceWindowTemplate:
    label: str
    start: time
    end: time
    capacity: int


_SERVICE_WINDOWS = (
    _ServiceWindowTemplate("morning", time(9, 0), time(12, 0), 2),
    _ServiceWindowTemplate("afternoon", time(13, 0), time(17, 0), 3),
)


def _current_time() -> datetime:
    """Return the current tenant-local time."""

    return datetime.now(tz=_TZ)


def _is_service_day(day: date) -> bool:
    """Return True when service windows exist on this date."""

    return day.weekday() < 5


def _window_bounds(
    day: date, template: _ServiceWindowTemplate
) -> tuple[datetime, datetime]:
    """Return aware start/end datetimes for a service window."""

    return (
        datetime.combine(day, template.start, tzinfo=_TZ),
        datetime.combine(day, template.end, tzinfo=_TZ),
    )


def _service_window_for(
    start: datetime, end: datetime
) -> _ServiceWindowTemplate | None:
    """Return the matching configured service window, if any."""

    local_start = _ensure_aware(start).astimezone(_TZ)
    local_end = _ensure_aware(end).astimezone(_TZ)
    if local_start.date() != local_end.date():
        return None

    for template in _SERVICE_WINDOWS:
        expected_start, expected_end = _window_bounds(local_start.date(), template)
        if local_start == expected_start and local_end == expected_end:
            return template
    return None


def _iter_service_days(range_start: date, range_end: date):
    day = range_start
    while day <= range_end:
        if _is_service_day(day):
            yield day
        day += timedelta(days=1)


class CalendarEvent(BaseModel):
    """A single calendar event stored for a tenant."""

    uid: str = Field(description="Stable event identifier used for updates.")
    start_iso: str = Field(description="Event start time in ISO 8601 format.")
    end_iso: str = Field(description="Event end time in ISO 8601 format.")
    event_kind: Literal["service_call", "block"] = Field(
        default="service_call",
        description="Whether this event is a customer service call or a block.",
    )
    summary: str = Field(description="Short human-readable event title.")
    customer_name: str = Field(
        default="",
        description="Customer full name for service calls.",
    )
    customer_phone: str = Field(
        default="",
        description="Reachable customer phone number for service calls.",
    )
    location: str = Field(
        default="",
        description="Service address for the visit.",
    )
    plumber_notes: str = Field(
        default="",
        description="Plumber-facing issue notes and diagnostic context.",
    )
    description: str = Field(
        default="",
        description="Optional human-readable event notes about the job.",
    )


class AvailabilityWindow(BaseModel):
    """One bookable service window."""

    date: str = Field(description="Window date in ISO format.")
    start_time: str = Field(description="Window start time in HH:MM.")
    end_time: str = Field(description="Window end time in HH:MM.")
    label: str = Field(description="Human-friendly window label.")
    remaining_capacity: int = Field(
        description="Remaining service-call slots in this window.",
    )


class AvailabilityResult(BaseModel):
    """Structured response for availability lookups."""

    success: bool = Field(description="Whether the date range was valid.")
    message: str = Field(description="Short explanation of the availability result.")
    windows: list[AvailabilityWindow] = Field(
        default_factory=list,
        description="Available service windows in ascending chronological order.",
    )


class ListEventsResult(BaseModel):
    """Structured response for listing a tenant's events."""

    success: bool = Field(description="Whether the date range was valid.")
    message: str = Field(description="Short explanation of the listing result.")
    events: list[CalendarEvent] = Field(
        default_factory=list,
        description="Events in ascending start-time order for the requested range.",
    )


class CalendarMutationResult(BaseModel):
    """Structured response for create, update, and delete operations."""

    success: bool = Field(description="Whether the mutation succeeded.")
    status: Literal[
        "created",
        "updated",
        "deleted",
        "conflict",
        "not_found",
        "invalid_input",
        "forbidden",
    ] = Field(description="Machine-friendly outcome of the calendar operation.")
    message: str = Field(description="Human-readable explanation of the outcome.")
    event: CalendarEvent | None = Field(
        default=None,
        description="The affected event when the operation succeeds.",
    )
    conflicting_event: CalendarEvent | None = Field(
        default=None,
        description="The event that blocks the requested slot when status is conflict.",
    )


@dataclass(frozen=True)
class _StoredEvent:
    uid: str
    start: datetime
    end: datetime
    event_kind: Literal["service_call", "block"]
    summary: str
    customer_name: str
    customer_phone: str
    location: str
    plumber_notes: str
    description: str

    def to_model(self) -> CalendarEvent:
        return CalendarEvent(
            uid=self.uid,
            start_iso=self.start.isoformat(),
            end_iso=self.end.isoformat(),
            event_kind=self.event_kind,
            summary=self.summary,
            customer_name=self.customer_name,
            customer_phone=self.customer_phone,
            location=self.location,
            plumber_notes=self.plumber_notes,
            description=self.description,
        )


def _get_tenant_lock(twilio_number: str) -> threading.Lock:
    tenant_key = phone_to_filename(twilio_number)
    lock = _TENANT_LOCKS.get(tenant_key)
    if lock is None:
        lock = threading.Lock()
        _TENANT_LOCKS[tenant_key] = lock
    return lock


def _calendar_dir_for(twilio_number: str) -> Path:
    return calendars_dir() / phone_to_filename(twilio_number)


def _parse_dt(date_str: str, time_str: str) -> datetime:
    d = date.fromisoformat(date_str)
    parts = time_str.split(":")
    t = time(int(parts[0]), int(parts[1]))
    return datetime.combine(d, t, tzinfo=_TZ)


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_TZ)
    return dt


def _build_ics(
    uid: str,
    start: datetime,
    end: datetime,
    event_kind: Literal["service_call", "block"],
    summary: str,
    customer_name: str,
    customer_phone: str,
    location: str,
    plumber_notes: str,
    description: str,
) -> bytes:
    cal = icalendar.Calendar()
    cal.add("prodid", "-//Montferrand Agent//EN")
    cal.add("version", "2.0")

    event = icalendar.Event()
    event.add("uid", uid)
    event.add("dtstart", start)
    event.add("dtend", end)
    event.add("x-montferrand-event-kind", event_kind)
    event.add("summary", summary)
    event.add("x-montferrand-customer-name", customer_name)
    event.add("x-montferrand-customer-phone", customer_phone)
    event.add("location", location)
    event.add("x-montferrand-plumber-notes", plumber_notes)
    event.add("description", description)
    event.add("dtstamp", datetime.now(tz=_TZ))
    event.add("created", datetime.now(tz=_TZ))

    cal.add_component(event)
    return cal.to_ical()


def _parse_ics(data: bytes) -> _StoredEvent | None:
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

        event_kind_raw = str(component.get("x-montferrand-event-kind", "service_call"))
        if event_kind_raw == "block":
            event_kind: Literal["service_call", "block"] = "block"
        else:
            event_kind = "service_call"

        return _StoredEvent(
            uid=uid,
            start=start,
            end=end,
            event_kind=event_kind,
            summary=str(component.get("summary", "")),
            customer_name=str(component.get("x-montferrand-customer-name", "")),
            customer_phone=str(component.get("x-montferrand-customer-phone", "")),
            location=str(component.get("location", "")),
            plumber_notes=str(component.get("x-montferrand-plumber-notes", "")),
            description=str(component.get("description", "")),
        )

    return None


def _has_overlap(
    events: list[_StoredEvent],
    start: datetime,
    end: datetime,
    exclude_uid: str | None = None,
) -> _StoredEvent | None:
    for event in events:
        if event.uid == exclude_uid:
            continue
        if event.start < end and event.end > start:
            return event
    return None


@dataclass
class TenantCalendarBackend:
    """Calendar backend scoped to a single tenant."""

    twilio_number: str
    directory: Path
    lock: threading.Lock

    @classmethod
    def for_tenant(cls, twilio_number: str) -> TenantCalendarBackend:
        return cls(
            twilio_number=twilio_number,
            directory=_calendar_dir_for(twilio_number),
            lock=_get_tenant_lock(twilio_number),
        )

    def ensure_exists(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    def _read_events(self) -> list[_StoredEvent]:
        self.ensure_exists()
        events: list[_StoredEvent] = []
        for ics_path in self.directory.glob("*.ics"):
            event = _parse_ics(ics_path.read_bytes())
            if event is None:
                logger.warning("Skipping unparseable .ics file: %s", ics_path)
                continue
            events.append(event)
        return events

    def _events_in_range(
        self, range_start: date, range_end: date
    ) -> list[_StoredEvent]:
        matches = []
        for event in self._read_events():
            if event.start.date() <= range_end and event.end.date() >= range_start:
                matches.append(event)
        matches.sort(key=lambda event: event.start)
        return matches

    def _events_in_window(
        self,
        start: datetime,
        end: datetime,
        *,
        kind: Literal["service_call", "block"] | None = None,
        exclude_uid: str | None = None,
    ) -> list[_StoredEvent]:
        matches: list[_StoredEvent] = []
        for event in self._read_events():
            if exclude_uid is not None and event.uid == exclude_uid:
                continue
            if kind is not None and event.event_kind != kind:
                continue
            if event.start < end and event.end > start:
                matches.append(event)
        matches.sort(key=lambda event: event.start)
        return matches

    def list_events(
        self,
        from_date: str,
        to_date: str,
        include_past: bool = False,
        recent_past_hours: int = 0,
    ) -> ListEventsResult:
        try:
            range_start = date.fromisoformat(from_date)
            range_end = date.fromisoformat(to_date)
        except ValueError as exc:
            return ListEventsResult(
                success=False,
                message=(
                    f"Invalid date format: {exc}. Use ISO format like '2026-03-16'."
                ),
            )

        matches = self._events_in_range(range_start, range_end)
        if not include_past:
            now = _current_time()
            cutoff = now - timedelta(hours=recent_past_hours)
            matches = [event for event in matches if event.end >= cutoff]

        match_models = [event.to_model() for event in matches]
        if not match_models:
            return ListEventsResult(
                success=True,
                message=(
                    "No events found in this date range."
                    if include_past
                    else "No upcoming events found in this date range."
                ),
                events=[],
            )
        return ListEventsResult(
            success=True,
            message=(
                f"Found {len(match_models)} event(s) in this date range."
                if include_past
                else f"Found {len(match_models)} upcoming event(s) in this date range."
            ),
            events=match_models,
        )

    def list_available_windows(
        self, from_date: str, to_date: str
    ) -> AvailabilityResult:
        """Return bookable service windows in the requested date range."""

        try:
            range_start = date.fromisoformat(from_date)
            range_end = date.fromisoformat(to_date)
        except ValueError as exc:
            return AvailabilityResult(
                success=False,
                message=(
                    f"Invalid date format: {exc}. Use ISO format like '2026-03-16'."
                ),
            )

        now = _current_time()
        windows: list[AvailabilityWindow] = []

        for day in _iter_service_days(range_start, range_end):
            for template in _SERVICE_WINDOWS:
                window_start, window_end = _window_bounds(day, template)
                if window_end <= now:
                    continue

                overlapping_blocks = self._events_in_window(
                    window_start,
                    window_end,
                    kind="block",
                )
                if overlapping_blocks:
                    continue

                overlapping_calls = self._events_in_window(
                    window_start,
                    window_end,
                    kind="service_call",
                )
                remaining_capacity = template.capacity - len(overlapping_calls)
                if remaining_capacity <= 0:
                    continue

                windows.append(
                    AvailabilityWindow(
                        date=day.isoformat(),
                        start_time=template.start.strftime("%H:%M"),
                        end_time=template.end.strftime("%H:%M"),
                        label=(
                            f"{day.isoformat()} {template.start.strftime('%H:%M')}-"
                            f"{template.end.strftime('%H:%M')}"
                        ),
                        remaining_capacity=remaining_capacity,
                    )
                )

        if not windows:
            return AvailabilityResult(
                success=True,
                message="No available service windows found in this date range.",
                windows=[],
            )

        return AvailabilityResult(
            success=True,
            message=f"Found {len(windows)} available service window(s).",
            windows=windows,
        )

    def create_service_call(
        self,
        date_str: str,
        start_time: str,
        end_time: str,
        summary: str,
        customer_name: str,
        customer_phone: str,
        location: str,
        plumber_notes: str,
    ) -> CalendarMutationResult:
        try:
            start = _parse_dt(date_str, start_time)
            end = _parse_dt(date_str, end_time)
        except (ValueError, IndexError) as exc:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message=(
                    f"Invalid date/time format: {exc}. Use date like '2026-03-16' "
                    f"and time like '09:00'."
                ),
            )

        if end <= start:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="End time must be after start time.",
            )

        service_window = _service_window_for(start, end)
        if service_window is None:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message=(
                    "Service calls must use a supported service window: "
                    "09:00-12:00 or 13:00-17:00."
                ),
            )

        if end <= _current_time():
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="Cannot book a service window that is already in the past.",
            )

        clean_summary = " ".join(summary.split()).strip()
        clean_customer_name = " ".join(customer_name.split()).strip()
        clean_customer_phone = " ".join(customer_phone.split()).strip()
        clean_location = " ".join(location.split()).strip()
        clean_plumber_notes = plumber_notes.strip()

        if not clean_location:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="Location is required to create a booking.",
            )

        if not clean_customer_name:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="Customer name is required to create a booking.",
            )

        if not clean_customer_phone:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="Customer phone is required to create a booking.",
            )

        if not clean_plumber_notes:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="Plumber notes are required to create a booking.",
            )

        with self.lock:
            overlapping_blocks = self._events_in_window(start, end, kind="block")
            if overlapping_blocks:
                block = overlapping_blocks[0]
                return CalendarMutationResult(
                    success=False,
                    status="conflict",
                    message=(
                        "This service window is blocked: "
                        f"'{block.summary}' from {block.start:%Y-%m-%d %H:%M} "
                        f"to {block.end:%H:%M}."
                    ),
                    conflicting_event=block.to_model(),
                )

            overlapping_calls = self._events_in_window(start, end, kind="service_call")
            if len(overlapping_calls) >= service_window.capacity:
                return CalendarMutationResult(
                    success=False,
                    status="conflict",
                    message=(
                        "This service window is already full. Choose another "
                        "available service window."
                    ),
                    conflicting_event=overlapping_calls[0].to_model(),
                )

            uid = uuid.uuid4().hex
            ics_path = self.ensure_exists() / f"{uid}.ics"
            ics_path.write_bytes(
                _build_ics(
                    uid,
                    start,
                    end,
                    "service_call",
                    clean_summary,
                    clean_customer_name,
                    clean_customer_phone,
                    clean_location,
                    clean_plumber_notes,
                    clean_plumber_notes,
                )
            )

        event = CalendarEvent(
            uid=uid,
            start_iso=start.isoformat(),
            end_iso=end.isoformat(),
            event_kind="service_call",
            summary=clean_summary,
            customer_name=clean_customer_name,
            customer_phone=clean_customer_phone,
            location=clean_location,
            plumber_notes=clean_plumber_notes,
            description=clean_plumber_notes,
        )
        return CalendarMutationResult(
            success=True,
            status="created",
            message=(
                f"Event '{clean_summary}' created on {date_str} from {start_time} "
                f"to {end_time}."
            ),
            event=event,
        )

    def create_block(
        self,
        date_str: str,
        start_time: str,
        end_time: str,
        summary: str,
        description: str = "",
    ) -> CalendarMutationResult:
        """Create a blocking event that removes availability."""

        try:
            start = _parse_dt(date_str, start_time)
            end = _parse_dt(date_str, end_time)
        except (ValueError, IndexError) as exc:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message=(
                    f"Invalid date/time format: {exc}. Use date like '2026-03-16' "
                    f"and time like '09:00'."
                ),
            )

        if end <= start:
            return CalendarMutationResult(
                success=False,
                status="invalid_input",
                message="End time must be after start time.",
            )

        clean_summary = (
            " ".join(summary.split()).strip()
            or f"{_BLOCK_EVENT_SUMMARY_PREFIX} indisponible"
        )
        clean_description = description.strip()

        with self.lock:
            uid = uuid.uuid4().hex
            ics_path = self.ensure_exists() / f"{uid}.ics"
            ics_path.write_bytes(
                _build_ics(
                    uid,
                    start,
                    end,
                    "block",
                    clean_summary,
                    "",
                    "",
                    "",
                    clean_description,
                    clean_description,
                )
            )

        event = CalendarEvent(
            uid=uid,
            start_iso=start.isoformat(),
            end_iso=end.isoformat(),
            event_kind="block",
            summary=clean_summary,
            description=clean_description,
        )
        return CalendarMutationResult(
            success=True,
            status="created",
            message=(
                f"Block '{clean_summary}' created on {date_str} from {start_time} "
                f"to {end_time}."
            ),
            event=event,
        )

    def delete_event(self, uid: str) -> CalendarMutationResult:
        with self.lock:
            events = self._read_events()
            event = next((item for item in events if item.uid == uid), None)
            if event is None:
                return CalendarMutationResult(
                    success=False,
                    status="not_found",
                    message=f"No event found with UID '{uid}'.",
                )

            ics_path = self.ensure_exists() / f"{uid}.ics"
            if not ics_path.exists():
                return CalendarMutationResult(
                    success=False,
                    status="not_found",
                    message=f"No event found with UID '{uid}'.",
                )
            ics_path.unlink()

        return CalendarMutationResult(
            success=True,
            status="deleted",
            message=(
                f"Event '{event.summary}' on {event.start:%Y-%m-%d} has been cancelled."
            ),
            event=event.to_model(),
        )

    def modify_event(
        self,
        uid: str,
        date_str: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        summary: str | None = None,
        customer_name: str | None = None,
        customer_phone: str | None = None,
        location: str | None = None,
        plumber_notes: str | None = None,
        description: str | None = None,
    ) -> CalendarMutationResult:
        with self.lock:
            events = self._read_events()
            current = next((item for item in events if item.uid == uid), None)
            if current is None:
                return CalendarMutationResult(
                    success=False,
                    status="not_found",
                    message=f"No event found with UID '{uid}'.",
                )

            new_summary = (
                " ".join(summary.split()).strip() if summary else current.summary
            )
            new_customer_name = (
                " ".join(customer_name.split()).strip()
                if customer_name is not None
                else current.customer_name
            )
            new_customer_phone = (
                " ".join(customer_phone.split()).strip()
                if customer_phone is not None
                else current.customer_phone
            )
            new_location = (
                " ".join(location.split()).strip()
                if location is not None
                else current.location
            )
            new_plumber_notes = (
                plumber_notes.strip()
                if plumber_notes is not None
                else current.plumber_notes
            )
            new_description = (
                description.strip()
                if description is not None
                else (
                    new_plumber_notes
                    if current.event_kind == "service_call"
                    else current.description
                )
            )

            if current.event_kind == "service_call" and not new_location:
                return CalendarMutationResult(
                    success=False,
                    status="invalid_input",
                    message="Location is required to keep a booking.",
                )

            if current.event_kind == "service_call" and not new_customer_name:
                return CalendarMutationResult(
                    success=False,
                    status="invalid_input",
                    message="Customer name is required to keep a booking.",
                )

            if current.event_kind == "service_call" and not new_customer_phone:
                return CalendarMutationResult(
                    success=False,
                    status="invalid_input",
                    message="Customer phone is required to keep a booking.",
                )

            if current.event_kind == "service_call" and not new_plumber_notes:
                return CalendarMutationResult(
                    success=False,
                    status="invalid_input",
                    message="Plumber notes are required to keep a booking.",
                )

            time_changed = any(
                value is not None for value in (date_str, start_time, end_time)
            )
            if time_changed:
                d = date_str or current.start.strftime("%Y-%m-%d")
                st = start_time or current.start.strftime("%H:%M")
                et = end_time or current.end.strftime("%H:%M")
                try:
                    new_start = _parse_dt(d, st)
                    new_end = _parse_dt(d, et)
                except (ValueError, IndexError) as exc:
                    return CalendarMutationResult(
                        success=False,
                        status="invalid_input",
                        message=(
                            f"Invalid date/time format: {exc}. Use date like "
                            f"'2026-03-16' and time like '09:00'."
                        ),
                    )
                if new_end <= new_start:
                    return CalendarMutationResult(
                        success=False,
                        status="invalid_input",
                        message="End time must be after start time.",
                    )

                if current.event_kind == "service_call":
                    service_window = _service_window_for(new_start, new_end)
                    if service_window is None:
                        return CalendarMutationResult(
                            success=False,
                            status="invalid_input",
                            message=(
                                "Service calls must use a supported service window: "
                                "09:00-12:00 or 13:00-17:00."
                            ),
                        )
                    if new_end <= _current_time():
                        return CalendarMutationResult(
                            success=False,
                            status="invalid_input",
                            message="Cannot move a service call into the past.",
                        )

                if current.event_kind == "service_call":
                    overlapping_blocks = self._events_in_window(
                        new_start,
                        new_end,
                        kind="block",
                    )
                    if overlapping_blocks:
                        return CalendarMutationResult(
                            success=False,
                            status="conflict",
                            message="This service window is blocked.",
                            conflicting_event=overlapping_blocks[0].to_model(),
                        )

                    overlapping_calls = self._events_in_window(
                        new_start,
                        new_end,
                        kind="service_call",
                        exclude_uid=uid,
                    )
                    service_window = _service_window_for(new_start, new_end)
                    assert service_window is not None
                    if len(overlapping_calls) >= service_window.capacity:
                        return CalendarMutationResult(
                            success=False,
                            status="conflict",
                            message="This service window is already full.",
                            conflicting_event=overlapping_calls[0].to_model(),
                        )
            else:
                new_start = current.start
                new_end = current.end

            ics_path = self.ensure_exists() / f"{uid}.ics"
            ics_path.write_bytes(
                _build_ics(
                    uid,
                    new_start,
                    new_end,
                    current.event_kind,
                    new_summary,
                    new_customer_name,
                    new_customer_phone,
                    new_location,
                    new_plumber_notes,
                    new_description,
                )
            )

        event = CalendarEvent(
            uid=uid,
            start_iso=new_start.isoformat(),
            end_iso=new_end.isoformat(),
            event_kind=current.event_kind,
            summary=new_summary,
            customer_name=new_customer_name,
            customer_phone=new_customer_phone,
            location=new_location,
            plumber_notes=new_plumber_notes,
            description=new_description,
        )
        return CalendarMutationResult(
            success=True,
            status="updated",
            message=(
                f"Event '{new_summary}' now runs on {new_start:%Y-%m-%d} from "
                f"{new_start:%H:%M} to {new_end:%H:%M}."
            ),
            event=event,
        )

    def modify_own_service_call(
        self,
        uid: str,
        expected_customer_phone: str,
        *,
        date_str: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        summary: str | None = None,
        customer_name: str | None = None,
        location: str | None = None,
        plumber_notes: str | None = None,
    ) -> CalendarMutationResult:
        current = next((item for item in self._read_events() if item.uid == uid), None)
        if current is None:
            return CalendarMutationResult(
                success=False,
                status="not_found",
                message=f"No event found with UID '{uid}'.",
            )
        if current.customer_phone != expected_customer_phone:
            return CalendarMutationResult(
                success=False,
                status="forbidden",
                message="This booking belongs to another customer.",
            )
        return self.modify_event(
            uid,
            date_str,
            start_time,
            end_time,
            summary,
            customer_name,
            expected_customer_phone,
            location,
            plumber_notes,
            None,
        )

    def delete_own_service_call(
        self,
        uid: str,
        expected_customer_phone: str,
    ) -> CalendarMutationResult:
        current = next((item for item in self._read_events() if item.uid == uid), None)
        if current is None:
            return CalendarMutationResult(
                success=False,
                status="not_found",
                message=f"No event found with UID '{uid}'.",
            )
        if current.customer_phone != expected_customer_phone:
            return CalendarMutationResult(
                success=False,
                status="forbidden",
                message="This booking belongs to another customer.",
            )
        return self.delete_event(uid)

    def list_customer_events(
        self,
        customer_phone: str,
        from_date: str,
        to_date: str,
        *,
        include_past: bool = False,
        recent_past_hours: int = 0,
    ) -> ListEventsResult:
        result = self.list_events(
            from_date,
            to_date,
            include_past=include_past,
            recent_past_hours=recent_past_hours,
        )
        return ListEventsResult(
            success=result.success,
            message=result.message,
            events=[
                event
                for event in result.events
                if event.customer_phone == customer_phone
            ],
        )

    def reset(self) -> int:
        with self.lock:
            count = 0
            if self.directory.exists():
                count = sum(
                    1 for path in self.directory.iterdir() if path.suffix == ".ics"
                )
                shutil.rmtree(self.directory)
            self.ensure_exists()
        return count


def get_tenant_calendar(twilio_number: str) -> TenantCalendarBackend:
    """Return a calendar backend scoped to one tenant."""

    return TenantCalendarBackend.for_tenant(twilio_number)


def ensure_tenant_calendar(twilio_number: str) -> Path:
    """Create the on-disk calendar directory for a tenant if needed."""

    return get_tenant_calendar(twilio_number).ensure_exists()


def reset_calendar(twilio_number: str) -> int:
    """Delete all calendar events for a tenant and return the file count."""

    return get_tenant_calendar(twilio_number).reset()


def list_events(
    twilio_number: str,
    from_date: str,
    to_date: str,
    include_past: bool = False,
    recent_past_hours: int = 0,
) -> ListEventsResult:
    """Compatibility wrapper around the tenant backend list operation."""

    return get_tenant_calendar(twilio_number).list_events(
        from_date,
        to_date,
        include_past=include_past,
        recent_past_hours=recent_past_hours,
    )


def list_available_windows(
    twilio_number: str,
    from_date: str,
    to_date: str,
) -> AvailabilityResult:
    """Compatibility wrapper around tenant availability lookup."""

    return get_tenant_calendar(twilio_number).list_available_windows(from_date, to_date)


def create_service_call(
    twilio_number: str,
    date_str: str,
    start_time: str,
    end_time: str,
    summary: str,
    customer_name: str,
    customer_phone: str,
    location: str,
    plumber_notes: str,
) -> CalendarMutationResult:
    """Compatibility wrapper around the tenant backend service-call create."""

    return get_tenant_calendar(twilio_number).create_service_call(
        date_str,
        start_time,
        end_time,
        summary,
        customer_name,
        customer_phone,
        location,
        plumber_notes,
    )


def create_block(
    twilio_number: str,
    date_str: str,
    start_time: str,
    end_time: str,
    summary: str,
    description: str = "",
) -> CalendarMutationResult:
    """Compatibility wrapper around the tenant backend block create."""

    return get_tenant_calendar(twilio_number).create_block(
        date_str,
        start_time,
        end_time,
        summary,
        description,
    )


def delete_event(twilio_number: str, uid: str) -> CalendarMutationResult:
    """Compatibility wrapper around the tenant backend delete operation."""

    return get_tenant_calendar(twilio_number).delete_event(uid)


def modify_event(
    twilio_number: str,
    uid: str,
    date_str: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    summary: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    location: str | None = None,
    plumber_notes: str | None = None,
    description: str | None = None,
) -> CalendarMutationResult:
    """Compatibility wrapper around the tenant backend modify operation."""

    return get_tenant_calendar(twilio_number).modify_event(
        uid,
        date_str,
        start_time,
        end_time,
        summary,
        customer_name,
        customer_phone,
        location,
        plumber_notes,
        description,
    )


def modify_own_service_call(
    twilio_number: str,
    uid: str,
    expected_customer_phone: str,
    *,
    date_str: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    summary: str | None = None,
    customer_name: str | None = None,
    location: str | None = None,
    plumber_notes: str | None = None,
) -> CalendarMutationResult:
    """Compatibility wrapper around customer-scoped service-call modification."""

    return get_tenant_calendar(twilio_number).modify_own_service_call(
        uid,
        expected_customer_phone,
        date_str=date_str,
        start_time=start_time,
        end_time=end_time,
        summary=summary,
        customer_name=customer_name,
        location=location,
        plumber_notes=plumber_notes,
    )


def delete_own_service_call(
    twilio_number: str,
    uid: str,
    expected_customer_phone: str,
) -> CalendarMutationResult:
    """Compatibility wrapper around customer-scoped service-call deletion."""

    return get_tenant_calendar(twilio_number).delete_own_service_call(
        uid,
        expected_customer_phone,
    )
