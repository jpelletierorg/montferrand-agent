from datetime import date

from montferrand_agent.tool_use_fixtures import (
    BOOKING_READY_CREATE_SERVICE_CALL_ARGS,
    BOOKING_READY_DATE_LABEL,
    BOOKING_READY_FIXTURE,
)


def test_booking_ready_fixture_uses_a_future_booking_date():
    booking_date = date.fromisoformat(BOOKING_READY_CREATE_SERVICE_CALL_ARGS["date"])
    assistant_text = str(
        getattr(BOOKING_READY_FIXTURE.history[-1].parts[0], "content", "")
    )

    assert booking_date > date.today()
    assert BOOKING_READY_DATE_LABEL in BOOKING_READY_FIXTURE.latest_user_message
    assert BOOKING_READY_DATE_LABEL in assistant_text
