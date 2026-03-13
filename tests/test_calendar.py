"""Tests for calendar.py — vdir calendar storage.

All tests are pure logic — no LLM calls.  Each test uses a temporary
directory for calendar storage via the MONTFERRAND_DATA_DIR env var.
"""

from pathlib import Path

import pytest

from montferrand_agent.calendar import (
    _tenant_calendar_dir,
    create_event,
    delete_event,
    list_events,
    modify_event,
)

from .conftest import TWILIO_NUMBER

# A second tenant for isolation tests
_OTHER_TENANT = "+19998887777"


@pytest.fixture(autouse=True)
def _isolate_calendar_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point calendar storage to a temp dir."""
    monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# list_events
# ---------------------------------------------------------------------------


class TestListEvents:
    def test_empty_calendar_returns_no_events(self):
        result = list_events(TWILIO_NUMBER, "2026-03-16", "2026-03-20")
        assert "No events" in result

    def test_invalid_date_format_returns_error(self):
        result = list_events(TWILIO_NUMBER, "not-a-date", "2026-03-20")
        assert "ERROR" in result

    def test_returns_events_in_range(self):
        create_event(TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Job A", "desc A")
        create_event(TWILIO_NUMBER, "2026-03-17", "13:00", "17:00", "Job B", "desc B")
        create_event(TWILIO_NUMBER, "2026-03-25", "09:00", "12:00", "Job C", "desc C")

        result = list_events(TWILIO_NUMBER, "2026-03-16", "2026-03-20")
        assert "Job A" in result
        assert "Job B" in result
        assert "Job C" not in result

    def test_returns_json_with_uid(self):
        create_event(TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Test", "desc")
        result = list_events(TWILIO_NUMBER, "2026-03-16", "2026-03-16")
        assert '"uid"' in result
        assert '"start"' in result
        assert '"end"' in result


# ---------------------------------------------------------------------------
# create_event
# ---------------------------------------------------------------------------


class TestCreateEvent:
    def test_creates_ics_file(self, tmp_path: Path):
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Leak fix", "Kitchen sink"
        )
        assert "BOOKED" in result
        assert "Leak fix" in result
        assert "UID:" in result

        # Verify .ics file was created
        vdir = _tenant_calendar_dir(TWILIO_NUMBER)
        ics_files = list(vdir.glob("*.ics"))
        assert len(ics_files) == 1

    def test_overlap_is_rejected(self):
        result1 = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Job 1", "desc"
        )
        assert "BOOKED" in result1

        result2 = create_event(
            TWILIO_NUMBER, "2026-03-16", "10:00", "13:00", "Job 2", "desc"
        )
        assert "CONFLICT" in result2

    def test_adjacent_events_no_overlap(self):
        """Events that end exactly when the next starts should not conflict."""
        result1 = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Morning", "desc"
        )
        assert "BOOKED" in result1

        result2 = create_event(
            TWILIO_NUMBER, "2026-03-16", "12:00", "17:00", "Afternoon", "desc"
        )
        assert "BOOKED" in result2

    def test_end_before_start_is_rejected(self):
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "12:00", "09:00", "Bad", "desc"
        )
        assert "ERROR" in result

    def test_invalid_date_returns_error(self):
        result = create_event(
            TWILIO_NUMBER, "not-a-date", "09:00", "12:00", "Bad", "desc"
        )
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------


class TestDeleteEvent:
    def test_delete_existing_event(self):
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "To delete", "desc"
        )
        uid = result.split("UID: ")[1].strip()

        delete_result = delete_event(TWILIO_NUMBER, uid)
        assert "DELETED" in delete_result
        assert "To delete" in delete_result

        # Event should be gone from listings
        listing = list_events(TWILIO_NUMBER, "2026-03-16", "2026-03-16")
        assert "No events" in listing

    def test_delete_nonexistent_returns_error(self):
        result = delete_event(TWILIO_NUMBER, "nonexistent-uid")
        assert "ERROR" in result

    def test_deleted_slot_can_be_rebooked(self):
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Original", "desc"
        )
        uid = result.split("UID: ")[1].strip()
        delete_event(TWILIO_NUMBER, uid)

        # Same slot should now be available
        result2 = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Replacement", "desc"
        )
        assert "BOOKED" in result2


# ---------------------------------------------------------------------------
# modify_event
# ---------------------------------------------------------------------------


class TestModifyEvent:
    def test_modify_summary(self):
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Old name", "desc"
        )
        uid = result.split("UID: ")[1].strip()

        mod_result = modify_event(TWILIO_NUMBER, uid, summary="New name")
        assert "UPDATED" in mod_result
        assert "New name" in mod_result

    def test_modify_time(self):
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Job", "desc"
        )
        uid = result.split("UID: ")[1].strip()

        mod_result = modify_event(
            TWILIO_NUMBER, uid, start_time="13:00", end_time="17:00"
        )
        assert "UPDATED" in mod_result
        assert "13:00" in mod_result

    def test_modify_with_overlap_rejected(self):
        create_event(
            TWILIO_NUMBER, "2026-03-16", "13:00", "17:00", "Afternoon job", "desc"
        )
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Morning job", "desc"
        )
        uid = result.split("UID: ")[1].strip()

        # Try to move morning job to overlap with afternoon
        mod_result = modify_event(
            TWILIO_NUMBER, uid, start_time="14:00", end_time="16:00"
        )
        assert "CONFLICT" in mod_result

    def test_modify_same_time_no_self_conflict(self):
        """Modifying only the summary should not trigger overlap with itself."""
        result = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Job", "desc"
        )
        uid = result.split("UID: ")[1].strip()

        mod_result = modify_event(TWILIO_NUMBER, uid, summary="Updated Job")
        assert "UPDATED" in mod_result

    def test_modify_nonexistent_returns_error(self):
        result = modify_event(TWILIO_NUMBER, "nonexistent-uid", summary="Nope")
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_tenant_isolation(self):
        """Events from one tenant are invisible to another."""
        create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Tenant A job", "desc"
        )
        create_event(
            _OTHER_TENANT, "2026-03-16", "09:00", "12:00", "Tenant B job", "desc"
        )

        result_a = list_events(TWILIO_NUMBER, "2026-03-16", "2026-03-16")
        result_b = list_events(_OTHER_TENANT, "2026-03-16", "2026-03-16")

        assert "Tenant A job" in result_a
        assert "Tenant B job" not in result_a
        assert "Tenant B job" in result_b
        assert "Tenant A job" not in result_b

    def test_no_cross_tenant_overlap(self):
        """Events in different tenants should not conflict."""
        result1 = create_event(
            TWILIO_NUMBER, "2026-03-16", "09:00", "12:00", "Tenant A", "desc"
        )
        assert "BOOKED" in result1

        result2 = create_event(
            _OTHER_TENANT, "2026-03-16", "09:00", "12:00", "Tenant B", "desc"
        )
        assert "BOOKED" in result2
