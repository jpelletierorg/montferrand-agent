"""Tests for booking calendar functions using frozen LLM-requested args."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.tool_use_fixtures import BOOKING_READY_CREATE_SERVICE_CALL_ARGS

from .conftest import TWILIO_NUMBER


@pytest.fixture(autouse=True)
def _isolate_calendar_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))


def test_create_service_call_with_frozen_llm_args_persists_booking():
    backend = get_tenant_calendar(TWILIO_NUMBER)
    booking_date = (date.today() + timedelta(days=1)).isoformat()

    result = backend.create_service_call(
        date_str=booking_date,
        start_time=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["start_time"],
        end_time=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["end_time"],
        summary=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["summary"],
        customer_name=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["customer_name"],
        customer_phone=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["customer_phone"],
        location=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["location"],
        plumber_notes=BOOKING_READY_CREATE_SERVICE_CALL_ARGS["plumber_notes"],
    )

    assert result.success is True
    assert result.status == "created"
    assert result.event is not None
    assert result.event.summary == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["summary"]
    assert (
        result.event.customer_name
        == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["customer_name"]
    )
    assert (
        result.event.customer_phone
        == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["customer_phone"]
    )
    assert result.event.location == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["location"]
    assert (
        result.event.plumber_notes
        == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["plumber_notes"]
    )

    listing = backend.list_events(booking_date, booking_date, include_past=True)
    assert listing.success is True
    assert len(listing.events) == 1
    assert (
        listing.events[0].summary == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["summary"]
    )
    assert (
        listing.events[0].customer_name
        == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["customer_name"]
    )
    assert (
        listing.events[0].customer_phone
        == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["customer_phone"]
    )
    assert (
        listing.events[0].location == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["location"]
    )
    assert (
        listing.events[0].plumber_notes
        == BOOKING_READY_CREATE_SERVICE_CALL_ARGS["plumber_notes"]
    )
