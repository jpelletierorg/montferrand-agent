"""Deterministic message-history fixtures for tool-use evaluation.

These fixtures freeze the conversation right before the booking agent should
decide to use a calendar tool. They are intentionally single-turn and do not
simulate the whole conversation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            "Ca ressemble a une fuite sous l'evier, a confirmer sur place. On peut passer le 19 mars 2026 entre 9h et 12h pour environ 180 a 260 $, a confirmer sur place."
        ),
    ],
    latest_user_message=(
        "Oui, le 19 mars 2026 entre 9h et 12h me convient. Je m'appelle Jean Tremblay, l'adresse est "
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
    "date": "2026-03-19",
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
