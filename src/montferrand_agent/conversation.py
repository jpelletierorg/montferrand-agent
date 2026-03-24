"""Conversation manager for the Montferrand booking agent.

Handles message history keyed by conversation ID, with an in-memory cache
backed by NDJSON files on disk.  Exposes a single ``process_message``
entry point that the CLI, eval harness, and SMS webhook all call.

Conversation files are stored in tenant-scoped subdirectories::

    {data_dir}/{tenant_hash}/{conversation_id}.ndjson

where ``tenant_hash`` is derived from the Twilio phone number (same hash
used by ``tenant.py`` for config files).  This structure allows wiping
all conversations for a specific tenant via ``reset_tenant()``.

The conversation key is always a plain string — callers decide how to
produce it (random for CLI/evals, deterministic hash for SMS via
``conversation_key_for_sms``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import shutil
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, Union, cast

import pydantic
from pydantic_graph import End
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai import BinaryContent, UserContent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)

from montferrand_agent.agent import (
    AgentDeps,
    get_agent,
    get_boss_agent,
    get_fallback_pricing,
    render_boss_prompt,
    render_prompt,
)
from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.config import conversations_dir
from montferrand_agent.crm import (
    maybe_get_tenant_crm,
    render_customer_context_for_prompt,
    reset_tenant_crm,
    tenant_crm_dir,
)
from montferrand_agent.models import AgentTurn, BossReply, Dialog, Report
from montferrand_agent.tenant import phone_to_filename, tenant_exists


class ConversationError(RuntimeError):
    """Raised when a conversation turn cannot be processed."""


TraceEventKind = Literal[
    "request_started",
    "request_finished",
    "tool_called",
    "tool_result",
    "tool_retry",
    "warning",
    "turn_finished",
]


@dataclass(frozen=True)
class ConversationTraceEvent:
    """Structured trace event emitted while running one agent turn."""

    kind: TraceEventKind
    summary: str
    at_seconds: float
    request_index: int | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    elapsed_seconds: float | None = None


TraceObserver = Callable[[ConversationTraceEvent], None]


def _emit_trace(observer: TraceObserver | None, event: ConversationTraceEvent) -> None:
    """Send a trace event to the observer if tracing is enabled."""

    if observer is not None:
        observer(event)


def _preview_text(text: str | None, limit: int = 120) -> str:
    """Return a single-line preview clipped for CLI display."""

    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _summarize_request_parts(parts: Sequence[object]) -> str:
    """Return a compact summary of the request parts sent to the model."""

    labels: list[str] = []
    for part in parts:
        if isinstance(part, UserPromptPart):
            if isinstance(part.content, str):
                labels.append(f"user={_preview_text(part.content, limit=80)}")
            else:
                labels.append("user=<multipart>")
        elif isinstance(part, ToolReturnPart):
            labels.append(f"tool-return:{part.tool_name}")
        elif isinstance(part, RetryPromptPart):
            target = part.tool_name or "result"
            labels.append(f"retry:{target}")
        else:
            part_kind = getattr(part, "part_kind", type(part).__name__)
            labels.append(str(part_kind))

    return ", ".join(labels) if labels else "<no parts>"


def _summarize_model_response(response: ModelResponse) -> str:
    """Return a compact summary of the model response for trace output."""

    bits: list[str] = []
    bits.append(f"parts={len(response.parts)}")

    if response.finish_reason:
        bits.append(f"finish={response.finish_reason}")

    tool_names = [call.tool_name for call in response.tool_calls]
    if tool_names:
        bits.append("tools=" + ", ".join(tool_names))

    text = _preview_text(response.text)
    if text:
        bits.append(f'text="{text}"')

    thinking = response.thinking
    if thinking:
        bits.append(f"thinking={len(thinking)} chars")

    return "; ".join(bits) if bits else "<empty response>"


def _summarize_tool_result(result: ToolReturnPart | RetryPromptPart) -> str:
    """Return a short preview of a tool result or retry instruction."""

    if isinstance(result, RetryPromptPart):
        content = result.model_response()
        return _preview_text(content)

    content = result.content
    if isinstance(content, str):
        return _preview_text(content)
    return _preview_text(str(content))


# ---------------------------------------------------------------------------
# NDJSON persistence
#
# WARNING: History files grow without bound.  Each conversation appends
# messages forever — the full history IS the customer CRM record.  A
# compaction / summarisation mechanism will be needed once histories
# approach the model's context window limit.  This is acceptable for the
# initial deployment but WILL need revisiting.
# ---------------------------------------------------------------------------

_ModelMessageAdapter = pydantic.TypeAdapter(
    ModelMessage,
    config=pydantic.ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def _data_dir() -> Path:
    """Return the directory for conversation NDJSON files."""
    return conversations_dir()


def _tenant_data_dir(twilio_number: str) -> Path:
    """Return the tenant-scoped subdirectory for conversation files."""
    return _data_dir() / phone_to_filename(twilio_number)


def _conversation_path(conversation_id: str, twilio_number: str) -> Path:
    """Return the NDJSON file path for a conversation."""
    return _tenant_data_dir(twilio_number) / f"{conversation_id}.ndjson"


def _load_history_from_disk(
    conversation_id: str, twilio_number: str
) -> list[ModelMessage]:
    """Read a conversation's full history from its NDJSON file."""
    path = _conversation_path(conversation_id, twilio_number)
    if not path.exists():
        return []

    messages: list[ModelMessage] = []
    for line_num, line in enumerate(path.read_bytes().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(_ModelMessageAdapter.validate_json(line))
        except Exception:
            logger.warning(
                "Skipping corrupted NDJSON line %d in %s",
                line_num,
                path,
            )
    return messages


def _append_messages_to_disk(
    conversation_id: str, messages: list[ModelMessage], twilio_number: str
) -> None:
    """Append new messages to the conversation's NDJSON file."""
    if not messages:
        return

    path = _conversation_path(conversation_id, twilio_number)
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
_locks: dict[str, asyncio.Lock] = {}


def _get_lock(conversation_id: str) -> asyncio.Lock:
    """Return (or create) the async lock for a conversation.

    Serializes concurrent turns for the same conversation so that
    history reads and writes never interleave.
    """
    if conversation_id not in _locks:
        _locks[conversation_id] = asyncio.Lock()
    return _locks[conversation_id]


def _get_history(conversation_id: str, twilio_number: str) -> list[ModelMessage]:
    """Return a copy of the message history for a conversation.

    Checks the in-memory cache first, then falls back to disk.
    """
    if conversation_id in _histories:
        return list(_histories[conversation_id])

    # Cache miss — try loading from NDJSON file
    messages = _load_history_from_disk(conversation_id, twilio_number)
    if messages:
        _histories[conversation_id] = messages
    return list(messages)


def _save_history(
    conversation_id: str,
    messages: list[ModelMessage],
    previous_message_count: int,
    twilio_number: str,
) -> None:
    """Persist the full message history (in-memory + append new to disk)."""
    _histories[conversation_id] = messages
    new_messages = messages[previous_message_count:]
    _append_messages_to_disk(conversation_id, new_messages, twilio_number)


def get_cost(conversation_id: str) -> ConversationCost:
    """Return the accumulated cost for a conversation."""
    return _costs.get(conversation_id, ConversationCost())


def reset(conversation_id: str, twilio_number: str) -> None:
    """Clear all state for a single conversation (memory and disk)."""
    _histories.pop(conversation_id, None)
    _costs.pop(conversation_id, None)
    _locks.pop(conversation_id, None)
    path = _conversation_path(conversation_id, twilio_number)
    if path.exists():
        path.unlink()


def reset_tenant(twilio_number: str) -> int:
    """Delete all conversation data and reset tenant calendar and CRM state.

    Removes the tenant's conversation subdirectory and all NDJSON files
    in it, plus the tenant's calendar vdir. Also clears any matching
    in-memory state and reprovisions a fresh empty tenant CRM database
    when one exists for the tenant.

    Returns the number of conversation files deleted.
    """
    tenant_dir = _tenant_data_dir(twilio_number)

    # Count files before deletion
    count = 0
    if tenant_dir.exists():
        count = sum(1 for f in tenant_dir.iterdir() if f.suffix == ".ndjson")

        # Clear in-memory state for conversations in this directory
        conversation_ids = [
            f.stem for f in tenant_dir.iterdir() if f.suffix == ".ndjson"
        ]
        for cid in conversation_ids:
            _histories.pop(cid, None)
            _costs.pop(cid, None)
            _locks.pop(cid, None)

        # Remove the entire directory
        shutil.rmtree(tenant_dir)

    # Wipe the tenant's calendar
    get_tenant_calendar(twilio_number).reset()

    # Wipe and reprovision the tenant CRM when this tenant has CRM state.
    if tenant_exists(twilio_number) or tenant_crm_dir(twilio_number).exists():
        reset_tenant_crm(twilio_number)

    return count


def list_conversations(twilio_number: str) -> list[str]:
    """List all conversation IDs for a tenant.

    Returns a list of conversation IDs (NDJSON filename stems).
    """
    tenant_dir = _tenant_data_dir(twilio_number)
    if not tenant_dir.exists():
        return []
    return sorted(f.stem for f in tenant_dir.iterdir() if f.suffix == ".ndjson")


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


async def _run_agent(
    prompt: str | list[UserContent],
    history: list[ModelMessage],
    instructions: str,
    twilio_number: str,
    *,
    is_boss: bool = False,
    customer_phone: str | None = None,
    conversation_id: str | None = None,
    trace_observer: TraceObserver | None = None,
):
    """Run the appropriate agent with the assembled system prompt.

    When *is_boss* is True, uses the boss agent and boss prompt.
    Otherwise uses the customer-facing booking agent.
    """
    calendar = get_tenant_calendar(twilio_number)
    crm = maybe_get_tenant_crm(twilio_number)
    agent_factory = get_boss_agent if is_boss else get_agent
    agent = cast(Any, agent_factory())
    turn_started_at = time.perf_counter()
    request_count = 0
    tool_call_count = 0
    repeated_tool_calls = 0
    tool_started_at: dict[str, float] = {}
    tool_signature_counts: dict[str, int] = {}
    try:
        async with agent.iter(
            prompt,
            message_history=history,
            instructions=instructions,
            deps=AgentDeps(
                calendar=calendar,
                crm=crm,
                customer_phone=customer_phone,
                conversation_id=conversation_id,
                twilio_number=twilio_number,
            ),
        ) as run:
            any_run = cast(Any, run)
            next_node = cast(Any, run.next_node)

            while not isinstance(next_node, End):
                if isinstance(next_node, UserPromptNode):
                    next_node = await any_run.next(next_node)
                    continue

                if isinstance(next_node, ModelRequestNode):
                    request_count += 1
                    request_started_at = time.perf_counter()
                    _emit_trace(
                        trace_observer,
                        ConversationTraceEvent(
                            kind="request_started",
                            summary=_summarize_request_parts(next_node.request.parts),
                            at_seconds=request_started_at - turn_started_at,
                            request_index=request_count,
                        ),
                    )

                    next_node = await any_run.next(next_node)
                    request_elapsed = time.perf_counter() - request_started_at

                    if isinstance(next_node, CallToolsNode):
                        _emit_trace(
                            trace_observer,
                            ConversationTraceEvent(
                                kind="request_finished",
                                summary=_summarize_model_response(
                                    next_node.model_response
                                ),
                                at_seconds=time.perf_counter() - turn_started_at,
                                request_index=request_count,
                                elapsed_seconds=request_elapsed,
                            ),
                        )
                    continue

                if isinstance(next_node, CallToolsNode):
                    async with next_node.stream(run.ctx) as events:
                        async for event in events:
                            now = time.perf_counter()
                            if isinstance(event, FunctionToolCallEvent):
                                tool_call_count += 1
                                tool_name = event.part.tool_name
                                tool_call_id = event.tool_call_id
                                tool_started_at[tool_call_id] = now
                                args_json = event.part.args_as_json_str()
                                args_preview = _preview_text(args_json)
                                signature = f"{tool_name}:{args_json}"
                                tool_signature_counts[signature] = (
                                    tool_signature_counts.get(signature, 0) + 1
                                )

                                _emit_trace(
                                    trace_observer,
                                    ConversationTraceEvent(
                                        kind="tool_called",
                                        summary=(
                                            f"{tool_name}({args_preview})"
                                            if args_preview
                                            else tool_name
                                        ),
                                        at_seconds=now - turn_started_at,
                                        request_index=request_count,
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                    ),
                                )

                                repeat_count = tool_signature_counts[signature]
                                if repeat_count > 1:
                                    repeated_tool_calls += 1
                                    _emit_trace(
                                        trace_observer,
                                        ConversationTraceEvent(
                                            kind="warning",
                                            summary=(
                                                f"repeated tool call x{repeat_count}: "
                                                f"{tool_name}({args_preview})"
                                            ),
                                            at_seconds=now - turn_started_at,
                                            request_index=request_count,
                                            tool_name=tool_name,
                                            tool_call_id=tool_call_id,
                                        ),
                                    )

                            elif isinstance(event, FunctionToolResultEvent):
                                tool_call_id = event.tool_call_id
                                started_at = tool_started_at.pop(tool_call_id, now)
                                elapsed = now - started_at
                                result = event.result
                                kind: TraceEventKind = (
                                    "tool_retry"
                                    if isinstance(result, RetryPromptPart)
                                    else "tool_result"
                                )
                                _emit_trace(
                                    trace_observer,
                                    ConversationTraceEvent(
                                        kind=kind,
                                        summary=_summarize_tool_result(result),
                                        at_seconds=now - turn_started_at,
                                        request_index=request_count,
                                        tool_name=result.tool_name,
                                        tool_call_id=tool_call_id,
                                        elapsed_seconds=elapsed,
                                    ),
                                )

                    next_node = await any_run.next(next_node)
                    if isinstance(next_node, ModelRequestNode):
                        if any(
                            isinstance(part, RetryPromptPart)
                            for part in next_node.request.parts
                        ):
                            summary = (
                                "model output rejected, retrying - "
                                f"{_summarize_request_parts(next_node.request.parts)}"
                            )
                        elif not next_node.request.parts:
                            summary = (
                                "model returned an empty response, retrying the "
                                "same request"
                            )
                        else:
                            summary = ""

                        if summary:
                            _emit_trace(
                                trace_observer,
                                ConversationTraceEvent(
                                    kind="warning",
                                    summary=summary,
                                    at_seconds=time.perf_counter() - turn_started_at,
                                    request_index=request_count,
                                ),
                            )
                    continue

                next_node = await any_run.next(next_node)

            result = cast(Any, run.result)
            assert result is not None
            total_elapsed = time.perf_counter() - turn_started_at
            _emit_trace(
                trace_observer,
                ConversationTraceEvent(
                    kind="turn_finished",
                    summary=(
                        f"{request_count} model request(s), {tool_call_count} tool call(s), "
                        f"{repeated_tool_calls} repeated call warning(s)"
                    ),
                    at_seconds=total_elapsed,
                    elapsed_seconds=total_elapsed,
                ),
            )
            return result
    except Exception as exc:
        raise ConversationError(f"Agent call failed: {exc}") from exc


async def process_message(
    conversation_id: str,
    text: str,
    images: Sequence[Path] | None = None,
    *,
    tenant_profile: str,
    twilio_number: str,
    is_boss: bool = False,
    customer_phone: str | None = None,
    trace_observer: TraceObserver | None = None,
) -> Union[Dialog, Report]:
    """Run one turn of conversation and return Dialog or Report.

    The *tenant_profile* parameter carries the company-specific text that
    is injected into the prompt template.  The *twilio_number* identifies
    which tenant's conversation subdirectory to use for persistence.
    Both parameters are required — there is no silent fallback.

    When *is_boss* is True, the boss agent and prompt are used instead
    of the customer-facing ones.

    Raises:
        ConversationError: If the agent call fails.
    """
    async with _get_lock(conversation_id):
        history = _get_history(conversation_id, twilio_number)
        prev_msg_count = len(history)
        prompt = _build_prompt(text, images)

        try:
            if is_boss:
                instructions = render_boss_prompt(tenant_profile)
            else:
                customer_context = (
                    "- No CRM record is available for this sender phone number."
                )
                if customer_phone:
                    crm = maybe_get_tenant_crm(twilio_number)
                    if crm is not None:
                        crm_context = await crm.get_customer_context_by_phone(
                            customer_phone
                        )
                        customer_context = render_customer_context_for_prompt(
                            crm_context
                        )
                instructions = render_prompt(
                    tenant_profile,
                    customer_crm_context=customer_context,
                )
        except Exception as exc:
            raise ConversationError(
                f"Failed to assemble agent instructions: {exc}"
            ) from exc
        result = await _run_agent(
            prompt,
            history,
            instructions,
            twilio_number,
            is_boss=is_boss,
            customer_phone=customer_phone,
            conversation_id=conversation_id,
            trace_observer=trace_observer,
        )

        all_messages = result.all_messages()
        _save_history(conversation_id, all_messages, prev_msg_count, twilio_number)

        try:
            _update_cost(conversation_id, all_messages, prev_msg_count, result.usage())
        except Exception:
            logger.exception(
                "Failed to update cost for conversation %s (reply still delivered)",
                conversation_id,
            )

        if is_boss:
            boss_reply = BossReply.model_validate(result.output)
            return Dialog(message=boss_reply.message)

        turn = AgentTurn.model_validate(result.output)
        return turn.to_public_result()
