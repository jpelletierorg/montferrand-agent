"""Conversation manager for the Montferrand booking agent.

Handles message history keyed by conversation ID, with an in-memory cache
backed by NDJSON files on disk.  Exposes a single ``process_message``
entry point that the CLI, eval harness, and SMS webhook all call.

The conversation key is always a plain string — callers decide how to
produce it (random for CLI/evals, deterministic hash for SMS via
``conversation_key_for_sms``).
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Sequence, Union

import pydantic
from pydantic_ai import BinaryContent, UserContent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.usage import RunUsage

from montferrand_agent.agent import (
    PROJECT_ROOT,
    dir_from_env,
    get_agent,
    get_fallback_pricing,
    render_prompt,
)
from montferrand_agent.models import Dialog, Report


class ConversationError(RuntimeError):
    """Raised when a conversation turn cannot be processed."""


# ---------------------------------------------------------------------------
# NDJSON persistence
#
# WARNING: History files grow without bound.  Each conversation appends
# messages forever — the full history IS the customer CRM record.  A
# compaction / summarisation mechanism will be needed once histories
# approach the model's context window limit.  This is acceptable for the
# initial deployment but WILL need revisiting.
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "conversations"

_ModelMessageAdapter = pydantic.TypeAdapter(
    ModelMessage,
    config=pydantic.ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def _data_dir() -> Path:
    """Return the directory for conversation NDJSON files."""
    return dir_from_env("MONTFERRAND_DATA_DIR", _DEFAULT_DATA_DIR)


def _conversation_path(conversation_id: str) -> Path:
    """Return the NDJSON file path for a conversation."""
    return _data_dir() / f"{conversation_id}.ndjson"


def _load_history_from_disk(conversation_id: str) -> list[ModelMessage]:
    """Read a conversation's full history from its NDJSON file."""
    path = _conversation_path(conversation_id)
    if not path.exists():
        return []

    messages: list[ModelMessage] = []
    for line in path.read_bytes().splitlines():
        line = line.strip()
        if line:
            messages.append(_ModelMessageAdapter.validate_json(line))
    return messages


def _append_messages_to_disk(
    conversation_id: str, messages: list[ModelMessage]
) -> None:
    """Append new messages to the conversation's NDJSON file."""
    if not messages:
        return

    path = _conversation_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("ab") as f:
        for msg in messages:
            f.write(_ModelMessageAdapter.dump_json(msg))
            f.write(b"\n")


# ---------------------------------------------------------------------------
# SMS conversation key
# ---------------------------------------------------------------------------


def conversation_key_for_sms(twilio_number: str, from_number: str) -> str:
    """Derive a deterministic conversation key from a phone number pair."""
    raw = (twilio_number + from_number).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Conversation cost tracking
# ---------------------------------------------------------------------------


@dataclass
class ConversationCost:
    """Accumulated cost and usage for a conversation."""

    total_usd: Decimal | None = Decimal(0)
    """Total cost in USD, or None if cost estimation is unavailable."""

    usage: RunUsage = field(default_factory=RunUsage)
    """Accumulated token usage across all turns."""

    @property
    def cost_available(self) -> bool:
        """Whether USD cost estimation worked for this conversation."""
        return self.total_usd is not None


# ---------------------------------------------------------------------------
# In-memory history store
# ---------------------------------------------------------------------------

_histories: dict[str, list[ModelMessage]] = {}
_costs: dict[str, ConversationCost] = {}


def _get_history(conversation_id: str) -> list[ModelMessage]:
    """Return a copy of the message history for a conversation.

    Checks the in-memory cache first, then falls back to disk.
    """
    if conversation_id in _histories:
        return list(_histories[conversation_id])

    # Cache miss — try loading from NDJSON file
    messages = _load_history_from_disk(conversation_id)
    if messages:
        _histories[conversation_id] = messages
    return list(messages)


def _save_history(
    conversation_id: str,
    messages: list[ModelMessage],
    previous_message_count: int,
) -> None:
    """Persist the full message history (in-memory + append new to disk)."""
    _histories[conversation_id] = messages
    new_messages = messages[previous_message_count:]
    _append_messages_to_disk(conversation_id, new_messages)


def get_cost(conversation_id: str) -> ConversationCost:
    """Return the accumulated cost for a conversation."""
    return _costs.get(conversation_id, ConversationCost())


def reset(conversation_id: str) -> None:
    """Clear all state for a conversation (memory and disk)."""
    _histories.pop(conversation_id, None)
    _costs.pop(conversation_id, None)
    path = _conversation_path(conversation_id)
    if path.exists():
        path.unlink()


def new_conversation_id() -> str:
    """Generate a fresh conversation identifier."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _read_image(path: Path) -> BinaryContent:
    """Read a local image file and return it as BinaryContent."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    media_type, _ = mimetypes.guess_type(str(path))
    if media_type is None or not media_type.startswith("image/"):
        media_type = "image/jpeg"

    return BinaryContent(data=path.read_bytes(), media_type=media_type)


# ---------------------------------------------------------------------------
# Turn processing
# ---------------------------------------------------------------------------


def _build_prompt(
    text: str, images: Sequence[Path] | None = None
) -> str | list[UserContent]:
    """Build the user prompt from text and optional image paths.

    Returns a plain string when there are no images (the common case), or
    a list of UserContent parts when images are attached.
    """
    stripped = text.strip()

    if not images:
        return stripped

    parts: list[UserContent] = []
    if stripped:
        parts.append(stripped)
    for image_path in images:
        parts.append(_read_image(image_path))
    return parts


def _estimate_response_cost(message: ModelResponse) -> Decimal | None:
    """Estimate the USD cost for a single model response."""
    try:
        return message.cost().total_price
    except Exception:
        fallback = get_fallback_pricing()
        if fallback is None:
            return None

    input_rate, output_rate = fallback
    return Decimal(
        str(
            message.usage.input_tokens * input_rate / 1_000_000
            + message.usage.output_tokens * output_rate / 1_000_000
        )
    )


def _update_cost(
    conversation_id: str,
    messages: list[ModelMessage],
    previous_message_count: int,
    usage: RunUsage,
) -> None:
    """Accumulate usage and cost for the latest turn."""
    cost = _costs.get(conversation_id, ConversationCost())
    cost.usage = cost.usage + usage

    for message in messages[previous_message_count:]:
        if not isinstance(message, ModelResponse) or cost.total_usd is None:
            continue

        response_cost = _estimate_response_cost(message)
        if response_cost is None:
            cost.total_usd = None
            continue

        cost.total_usd += response_cost

    _costs[conversation_id] = cost


async def _run_booking_agent(
    prompt: str | list[UserContent],
    history: list[ModelMessage],
    instructions: str,
):
    """Run the booking agent with the assembled system prompt."""
    try:
        return await get_agent().run(
            prompt,
            message_history=history,
            instructions=instructions,
        )
    except Exception as exc:
        raise ConversationError(f"Agent call failed: {exc}") from exc


async def process_message(
    conversation_id: str,
    text: str,
    images: Sequence[Path] | None = None,
    *,
    tenant_profile: str,
) -> Union[Dialog, Report]:
    """Run one turn of conversation and return Dialog or Report.

    The *tenant_profile* parameter carries the company-specific text that
    is injected into the master prompt template.  This parameter is
    required — there is no silent fallback.  If a caller does not have
    a tenant profile, that is a configuration error and should crash.

    Raises:
        ConversationError: If the agent call fails.
    """
    history = _get_history(conversation_id)
    prev_msg_count = len(history)
    prompt = _build_prompt(text, images)

    instructions = render_prompt(tenant_profile)
    result = await _run_booking_agent(prompt, history, instructions)

    all_messages = result.all_messages()
    _save_history(conversation_id, all_messages, prev_msg_count)
    _update_cost(conversation_id, all_messages, prev_msg_count, result.usage())

    return result.output
