"""Tests for tenant-scoped calendar backends."""

from datetime import datetime
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

from montferrand_agent import calendar as calendar_module
from montferrand_agent.calendar import ensure_tenant_calendar, get_tenant_calendar

from .conftest import TWILIO_NUMBER

_OTHER_TENANT = "+19998887777"
_LOCATION = "123 rue Test, Longueuil"
_OTHER_LOCATION = "45 avenue Bossuet, Brossard"
_CUSTOMER_NAME = "Jean Tremblay"
_CUSTOMER_PHONE = "+15550000002"


@pytest.fixture(autouse=True)
def _isolate_calendar_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        calendar_module,
        "_current_time",
        lambda: datetime(2026, 3, 15, 12, 0, tzinfo=ZoneInfo("America/Montreal")),
    )


def _event_uid(result) -> str:
    assert result.event is not None
    return result.event.uid


def _event_summary(result) -> str:
    assert result.event is not None
    return result.event.summary


def _event_start_iso(result) -> str:
    assert result.event is not None
    return result.event.start_iso


def _event_location(result) -> str:
    assert result.event is not None
    return result.event.location


def _create_service_call(
    backend,
    date_str: str,
    start_time: str,
    end_time: str,
    summary: str,
    location: str,
    plumber_notes: str,
):
    return backend.create_service_call(
        date_str,
        start_time,
        end_time,
        summary,
        _CUSTOMER_NAME,
        _CUSTOMER_PHONE,
        location,
        plumber_notes,
    )


class TestProvisioning:
    def test_provision_creates_calendar_directory(self):
        path = ensure_tenant_calendar(TWILIO_NUMBER)
        assert path.exists()
        assert path.is_dir()


class TestReset:
    def test_reset_recreates_empty_calendar_directory(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        _create_service_call(
            backend,
            "2026-03-16",
            "09:00",
            "12:00",
            "Service call",
            _LOCATION,
            "desc",
        )
        block = backend.create_block(
            "2026-03-17",
            "13:00",
            "17:00",
            "Blocked",
            "Vacation",
        )
        assert block.success is True

        deleted = backend.reset()

        assert deleted == 2
        assert backend.directory.exists()
        assert backend.directory.is_dir()
        assert list(backend.directory.glob("*.ics")) == []
        assert (
            backend.list_events("2026-03-16", "2026-03-17", include_past=True).events
            == []
        )


class TestListEvents:
    def test_default_list_events_excludes_past_events(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Past job", _LOCATION, "desc past"
        )
        _create_service_call(
            backend,
            "2026-03-19",
            "09:00",
            "12:00",
            "Future job",
            _LOCATION,
            "desc future",
        )

        monkeypatch.setattr(
            calendar_module,
            "_current_time",
            lambda: datetime(2026, 3, 18, 12, 0, tzinfo=ZoneInfo("America/Montreal")),
        )

        result = backend.list_events("2026-03-01", "2026-03-31")

        assert [event.summary for event in result.events] == ["Future job"]
        assert "upcoming" in result.message

    def test_list_events_can_include_past_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Past job", _LOCATION, "desc past"
        )
        _create_service_call(
            backend,
            "2026-03-19",
            "09:00",
            "12:00",
            "Future job",
            _LOCATION,
            "desc future",
        )

        monkeypatch.setattr(
            calendar_module,
            "_current_time",
            lambda: datetime(2026, 3, 18, 12, 0, tzinfo=ZoneInfo("America/Montreal")),
        )

        result = backend.list_events("2026-03-01", "2026-03-31", include_past=True)

        assert [event.summary for event in result.events] == ["Past job", "Future job"]

    def test_empty_calendar_returns_no_events(self):
        result = get_tenant_calendar(TWILIO_NUMBER).list_events(
            "2026-03-16", "2026-03-20"
        )
        assert result.success is True
        assert result.events == []
        assert "No upcoming events" in result.message

    def test_invalid_date_format_returns_error(self):
        result = get_tenant_calendar(TWILIO_NUMBER).list_events(
            "not-a-date", "2026-03-20"
        )
        assert result.success is False
        assert "Invalid date format" in result.message

    def test_returns_events_in_range(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job A", _LOCATION, "desc A"
        )
        _create_service_call(
            backend, "2026-03-17", "13:00", "17:00", "Job B", _LOCATION, "desc B"
        )
        _create_service_call(
            backend, "2026-03-25", "09:00", "12:00", "Job C", _LOCATION, "desc C"
        )

        result = backend.list_events("2026-03-16", "2026-03-20", include_past=True)
        summaries = [event.summary for event in result.events]
        assert summaries == ["Job A", "Job B"]

    def test_returns_event_models_with_uid(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Test", _LOCATION, "desc"
        )

        result = backend.list_events("2026-03-16", "2026-03-16", include_past=True)
        assert result.success is True
        assert len(result.events) == 1
        assert result.events[0].uid == _event_uid(created)
        assert result.events[0].start_iso.startswith("2026-03-16T09:00")
        assert result.events[0].end_iso.startswith("2026-03-16T12:00")
        assert result.events[0].location == _LOCATION


class TestCreateEvent:
    def test_creates_ics_file(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        result = backend.create_service_call(
            "2026-03-16",
            "09:00",
            "12:00",
            "Leak fix",
            _CUSTOMER_NAME,
            _CUSTOMER_PHONE,
            _LOCATION,
            "Kitchen sink",
        )
        assert result.success is True
        assert result.status == "created"
        assert _event_summary(result) == "Leak fix"
        assert _event_location(result) == _LOCATION

        ics_files = list(backend.directory.glob("*.ics"))
        assert len(ics_files) == 1

    def test_overlap_is_rejected(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        result1 = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job 1", _LOCATION, "desc"
        )
        assert result1.success is True

        result2 = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job 2", _LOCATION, "desc"
        )
        assert result2.success is True

        result3 = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job 3", _LOCATION, "desc"
        )
        assert result3.success is False
        assert result3.status == "conflict"
        assert result3.conflicting_event is not None

    def test_adjacent_events_no_overlap(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        result1 = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Morning", _LOCATION, "desc"
        )
        assert result1.success is True

        result2 = _create_service_call(
            backend, "2026-03-16", "13:00", "17:00", "Afternoon", _LOCATION, "desc"
        )
        assert result2.success is True

    def test_end_before_start_is_rejected(self):
        result = get_tenant_calendar(TWILIO_NUMBER).create_service_call(
            "2026-03-16",
            "12:00",
            "09:00",
            "Bad",
            _CUSTOMER_NAME,
            _CUSTOMER_PHONE,
            _LOCATION,
            "desc",
        )
        assert result.success is False
        assert result.status == "invalid_input"

    def test_invalid_date_returns_error(self):
        result = get_tenant_calendar(TWILIO_NUMBER).create_service_call(
            "not-a-date",
            "09:00",
            "12:00",
            "Bad",
            _CUSTOMER_NAME,
            _CUSTOMER_PHONE,
            _LOCATION,
            "desc",
        )
        assert result.success is False
        assert result.status == "invalid_input"

    def test_blank_location_is_rejected(self):
        result = get_tenant_calendar(TWILIO_NUMBER).create_service_call(
            "2026-03-16",
            "09:00",
            "12:00",
            "Bad",
            _CUSTOMER_NAME,
            _CUSTOMER_PHONE,
            "   ",
            "desc",
        )
        assert result.success is False
        assert result.status == "invalid_input"


class TestDeleteEvent:
    def test_delete_existing_event(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "To delete", _LOCATION, "desc"
        )

        delete_result = backend.delete_event(_event_uid(created))
        assert delete_result.success is True
        assert delete_result.status == "deleted"
        assert _event_summary(delete_result) == "To delete"

        listing = backend.list_events("2026-03-16", "2026-03-16")
        assert listing.events == []

    def test_delete_nonexistent_returns_error(self):
        result = get_tenant_calendar(TWILIO_NUMBER).delete_event("nonexistent-uid")
        assert result.success is False
        assert result.status == "not_found"

    def test_deleted_slot_can_be_rebooked(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Original", _LOCATION, "desc"
        )
        backend.delete_event(_event_uid(created))

        result2 = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Replacement", _LOCATION, "desc"
        )
        assert result2.success is True


class TestModifyEvent:
    def test_modify_summary(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Old name", _LOCATION, "desc"
        )

        mod_result = backend.modify_event(_event_uid(created), summary="New name")
        assert mod_result.success is True
        assert mod_result.status == "updated"
        assert _event_summary(mod_result) == "New name"

    def test_modify_time(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job", _LOCATION, "desc"
        )

        mod_result = backend.modify_event(
            _event_uid(created),
            start_time="13:00",
            end_time="17:00",
        )
        assert mod_result.success is True
        assert _event_start_iso(mod_result).startswith("2026-03-16T13:00")

    def test_modify_with_overlap_rejected(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        _create_service_call(
            backend, "2026-03-16", "13:00", "17:00", "Afternoon job", _LOCATION, "desc"
        )
        _create_service_call(
            backend,
            "2026-03-16",
            "13:00",
            "17:00",
            "Afternoon job 2",
            _LOCATION,
            "desc",
        )
        _create_service_call(
            backend,
            "2026-03-16",
            "13:00",
            "17:00",
            "Afternoon job 3",
            _LOCATION,
            "desc",
        )
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Morning job", _LOCATION, "desc"
        )

        mod_result = backend.modify_event(
            _event_uid(created),
            start_time="13:00",
            end_time="17:00",
        )
        assert mod_result.success is False
        assert mod_result.status == "conflict"

    def test_modify_same_time_no_self_conflict(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job", _LOCATION, "desc"
        )

        mod_result = backend.modify_event(_event_uid(created), summary="Updated Job")
        assert mod_result.success is True

    def test_modify_location(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job", _LOCATION, "desc"
        )

        mod_result = backend.modify_event(
            _event_uid(created),
            location=_OTHER_LOCATION,
        )

        assert mod_result.success is True
        assert _event_location(mod_result) == _OTHER_LOCATION

    def test_modify_blank_location_is_rejected(self):
        backend = get_tenant_calendar(TWILIO_NUMBER)
        created = _create_service_call(
            backend, "2026-03-16", "09:00", "12:00", "Job", _LOCATION, "desc"
        )

        mod_result = backend.modify_event(_event_uid(created), location="   ")
        assert mod_result.success is False
        assert mod_result.status == "invalid_input"

    def test_modify_nonexistent_returns_error(self):
        result = get_tenant_calendar(TWILIO_NUMBER).modify_event(
            "nonexistent-uid", summary="Nope"
        )
        assert result.success is False
        assert result.status == "not_found"


class TestTenantIsolation:
    def test_tenant_isolation(self):
        backend_a = get_tenant_calendar(TWILIO_NUMBER)
        backend_b = get_tenant_calendar(_OTHER_TENANT)
        _create_service_call(
            backend_a, "2026-03-16", "09:00", "12:00", "Tenant A job", _LOCATION, "desc"
        )
        _create_service_call(
            backend_b,
            "2026-03-16",
            "09:00",
            "12:00",
            "Tenant B job",
            _OTHER_LOCATION,
            "desc",
        )

        result_a = backend_a.list_events("2026-03-16", "2026-03-16", include_past=True)
        result_b = backend_b.list_events("2026-03-16", "2026-03-16", include_past=True)

        assert [event.summary for event in result_a.events] == ["Tenant A job"]
        assert [event.summary for event in result_b.events] == ["Tenant B job"]
        assert [event.location for event in result_a.events] == [_LOCATION]
        assert [event.location for event in result_b.events] == [_OTHER_LOCATION]

    def test_no_cross_tenant_overlap(self):
        backend_a = get_tenant_calendar(TWILIO_NUMBER)
        backend_b = get_tenant_calendar(_OTHER_TENANT)

        result1 = _create_service_call(
            backend_a, "2026-03-16", "09:00", "12:00", "Tenant A", _LOCATION, "desc"
        )
        assert result1.success is True

        result2 = _create_service_call(
            backend_b,
            "2026-03-16",
            "09:00",
            "12:00",
            "Tenant B",
            _OTHER_LOCATION,
            "desc",
        )
        assert result2.success is True
