"""Deterministic message-history fixtures for tool-use evaluation.

These fixtures freeze the conversation right before the booking agent should
decide to use a calendar tool. They are intentionally single-turn and do not
simulate the whole conversation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


_MONTHS_FR = {
    1: "janvier",
    2: "fevrier",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "aout",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "decembre",
}


def _next_fixture_booking_date(today: date | None = None) -> date:
    """Return the next future weekday for booking fixtures."""

    candidate = (today or date.today()) + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _format_french_date(value: date) -> str:
    """Return a simple French date label like '24 mars 2026'."""

    return f"{value.day} {_MONTHS_FR[value.month]} {value.year}"


BOOKING_READY_DATE = _next_fixture_booking_date()
BOOKING_READY_DATE_ISO = BOOKING_READY_DATE.isoformat()
BOOKING_READY_DATE_LABEL = _format_french_date(BOOKING_READY_DATE)


@dataclass(frozen=True)
class ToolUseFixture:
    """A deterministic pre-booking conversation snapshot."""

    name: str
    history: list[ModelMessage]
    latest_user_message: str
    expected_tool_name: str
    expected_location: str


BOOKING_READY_FIXTURE = ToolUseFixture(
    name="booking_ready_create_service_call",
    history=[
        _user("Bonjour, j'ai une fuite sous mon evier de cuisine."),
        _assistant(
            "Bonjour. Afin de comprendre, est-ce que l'eau coule activement en ce moment?"
        ),
        _user("Oui, ca goutte surtout quand j'ouvre le robinet."),
        _assistant(
            "D'accord. Est-ce que les autres drains de la maison fonctionnent normalement?"
        ),
        _user("Oui, tout le reste fonctionne normalement."),
        _assistant(
            f"Ca ressemble a une fuite sous l'evier, a confirmer sur place. On peut passer le {BOOKING_READY_DATE_LABEL} entre 9h et 12h pour environ 180 a 260 $, a confirmer sur place."
        ),
    ],
    latest_user_message=(
        f"Oui, le {BOOKING_READY_DATE_LABEL} entre 9h et 12h me convient. Je m'appelle Jean Tremblay, l'adresse est "
        "123 rue Test, Longueuil, J4K 1A1, et vous pouvez me joindre a ce numero."
    ),
    expected_tool_name="tool_create_service_call",
    expected_location="123 rue Test, Longueuil, J4K 1A1",
)


AVAILABILITY_QUERY_FIXTURE = ToolUseFixture(
    name="availability_query_requires_tool",
    history=[
        _user("Bonjour"),
        _assistant(
            "Bonjour. Quel problème de plomberie souhaitez-vous faire vérifier?"
        ),
    ],
    latest_user_message="Avez-vous de la place vendredi?",
    expected_tool_name="tool_check_availability",
    expected_location="",
)


TOOL_USE_FIXTURES = [BOOKING_READY_FIXTURE, AVAILABILITY_QUERY_FIXTURE]

BOOKING_READY_CREATE_SERVICE_CALL_ARGS = {
    "date": BOOKING_READY_DATE_ISO,
    "start_time": "09:00",
    "end_time": "12:00",
    "summary": "Fuite sous évier de cuisine - Jean Tremblay",
    "customer_name": "Jean Tremblay",
    "customer_phone": "+15550000002",
    "location": "123 rue Test, Longueuil, J4K 1A1",
    "plumber_notes": (
        "Fuite qui goutte sous l’évier de cuisine, surtout lorsqu’on ouvre le "
        "robinet; vérifier les connexions, les flexibles et l’état des pièces "
        "sous l’évier et réparer au besoin."
    ),
}
