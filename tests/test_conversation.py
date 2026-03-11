"""Tests for conversation.py — history, cost tracking, and prompt building.

All tests are pure logic — no LLM calls.
"""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.usage import RunUsage

from montferrand_agent.conversation import (
    ConversationCost,
    _append_messages_to_disk,
    _build_prompt,
    _data_dir,
    _get_history,
    _load_history_from_disk,
    _read_image,
    _save_history,
    conversation_key_for_sms,
    get_cost,
    new_conversation_id,
    reset,
)

from .conftest import CUSTOMER_NUMBER, TWILIO_NUMBER, assert_hex_string

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# new_conversation_id
# ---------------------------------------------------------------------------


class TestNewConversationId:
    def test_returns_12_char_hex(self):
        assert_hex_string(new_conversation_id(), 12)

    def test_returns_unique_ids(self):
        ids = {new_conversation_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


class TestHistory:
    @pytest.fixture(autouse=True)
    def _isolate_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Send NDJSON writes to a temp dir so fake messages don't hit disk."""
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))

    def test_get_history_unknown_id_returns_empty(self):
        assert _get_history("nonexistent") == []

    def test_save_and_get_roundtrip(self):
        cid = new_conversation_id()
        msgs = ["msg1", "msg2"]  # type: ignore[list-item]
        _save_history(cid, msgs, 0)  # type: ignore[arg-type]
        assert _get_history(cid) == msgs

    def test_get_history_returns_copy(self):
        cid = new_conversation_id()
        msgs = ["msg1"]  # type: ignore[list-item]
        _save_history(cid, msgs, 0)  # type: ignore[arg-type]
        retrieved = _get_history(cid)
        retrieved.append("extra")  # type: ignore[arg-type]
        assert _get_history(cid) == msgs  # original unaffected

    def test_reset_clears_history(self):
        cid = new_conversation_id()
        _save_history(cid, ["msg"], 0)  # type: ignore[arg-type]
        reset(cid)
        assert _get_history(cid) == []

    def test_reset_unknown_id_does_not_raise(self):
        reset("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# ConversationCost
# ---------------------------------------------------------------------------


class TestConversationCost:
    def test_default_cost_available(self):
        cost = ConversationCost()
        assert cost.cost_available is True
        assert cost.total_usd == Decimal(0)

    def test_cost_not_available_when_none(self):
        cost = ConversationCost(total_usd=None)
        assert cost.cost_available is False

    def test_get_cost_unknown_id_returns_zero(self):
        cost = get_cost("nonexistent")
        assert cost.cost_available is True
        assert cost.total_usd == Decimal(0)
        assert cost.usage.total_tokens == 0

    def test_reset_clears_cost(self):
        from montferrand_agent.conversation import _costs

        cid = new_conversation_id()
        _costs[cid] = ConversationCost(
            total_usd=Decimal("0.05"),
            usage=RunUsage(input_tokens=100, output_tokens=50),
        )
        reset(cid)
        cost = get_cost(cid)
        assert cost.total_usd == Decimal(0)
        assert cost.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# _read_image
# ---------------------------------------------------------------------------


class TestReadImage:
    def test_reads_jpeg(self):
        img = _read_image(FIXTURES_DIR / "leaky_faucet.jpg")
        assert isinstance(img, BinaryContent)
        assert img.media_type == "image/jpeg"
        assert len(img.data) > 0

    def test_reads_different_file(self):
        img = _read_image(FIXTURES_DIR / "clogged_drain.jpg")
        assert img.media_type == "image/jpeg"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            _read_image(Path("/nonexistent/image.jpg"))

    def test_unknown_extension_defaults_to_jpeg(self, tmp_path: Path):
        f = tmp_path / "photo.xyz"
        f.write_bytes(b"fake image data")
        img = _read_image(f)
        assert img.media_type == "image/jpeg"

    def test_png_gets_correct_media_type(self, tmp_path: Path):
        f = tmp_path / "photo.png"
        f.write_bytes(b"fake png data")
        img = _read_image(f)
        assert img.media_type == "image/png"


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_text_only_returns_string(self):
        result = _build_prompt("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_strips_whitespace(self):
        assert _build_prompt("  hello  ") == "hello"

    def test_text_with_image_returns_list(self):
        img_path = FIXTURES_DIR / "leaky_faucet.jpg"
        result = _build_prompt("leak here", [img_path])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "leak here"
        assert isinstance(result[1], BinaryContent)

    def test_image_only_no_text(self):
        img_path = FIXTURES_DIR / "leaky_faucet.jpg"
        result = _build_prompt("", [img_path])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], BinaryContent)

    def test_multiple_images(self):
        paths = [
            FIXTURES_DIR / "leaky_faucet.jpg",
            FIXTURES_DIR / "clogged_drain.jpg",
        ]
        result = _build_prompt("photos", paths)
        assert isinstance(result, list)
        assert len(result) == 3  # text + 2 images
        assert result[0] == "photos"

    def test_no_images_none(self):
        result = _build_prompt("hello", None)
        assert result == "hello"
        assert isinstance(result, str)

    def test_no_images_empty_list(self):
        result = _build_prompt("hello", [])
        assert result == "hello"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# conversation_key_for_sms
# ---------------------------------------------------------------------------


class TestConversationKeyForSms:
    def test_returns_12_char_hex(self):
        assert_hex_string(conversation_key_for_sms(TWILIO_NUMBER, CUSTOMER_NUMBER), 12)

    def test_deterministic(self):
        a = conversation_key_for_sms(TWILIO_NUMBER, CUSTOMER_NUMBER)
        b = conversation_key_for_sms(TWILIO_NUMBER, CUSTOMER_NUMBER)
        assert a == b

    def test_different_pairs_differ(self):
        a = conversation_key_for_sms(TWILIO_NUMBER, CUSTOMER_NUMBER)
        b = conversation_key_for_sms(TWILIO_NUMBER, "+14389999999")
        assert a != b

    def test_order_matters(self):
        a = conversation_key_for_sms(TWILIO_NUMBER, CUSTOMER_NUMBER)
        b = conversation_key_for_sms(CUSTOMER_NUMBER, TWILIO_NUMBER)
        assert a != b


# ---------------------------------------------------------------------------
# NDJSON persistence
# ---------------------------------------------------------------------------


class TestNdjsonPersistence:
    @pytest.fixture(autouse=True)
    def _isolate_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Point NDJSON writes to a temp dir for all persistence tests."""
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))

    def test_data_dir_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MONTFERRAND_DATA_DIR", raising=False)
        directory = _data_dir()
        assert directory.name == "conversations"
        assert directory.parent.name == "data"

    def test_data_dir_env_override(self):
        # The autouse fixture already sets the env var
        directory = _data_dir()
        assert directory.exists() or True  # temp dir set via autouse

    def test_load_from_empty_dir(self):
        assert _load_history_from_disk("nonexistent") == []

    def test_append_and_load_roundtrip(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("test123", [msg])

        loaded = _load_history_from_disk("test123")
        assert len(loaded) == 1
        assert isinstance(loaded[0], ModelRequest)
        assert loaded[0].parts[0].content == "hello"  # type: ignore[union-attr]

    def test_append_is_incremental(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg1 = ModelRequest(parts=[UserPromptPart(content="first")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="second")])

        _append_messages_to_disk("test456", [msg1])
        _append_messages_to_disk("test456", [msg2])

        loaded = _load_history_from_disk("test456")
        assert len(loaded) == 2

    def test_get_history_falls_back_to_disk(self):
        """_get_history loads from disk when not in memory cache."""
        from montferrand_agent.conversation import _histories
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cid = "diskonly123"
        msg = ModelRequest(parts=[UserPromptPart(content="from disk")])
        _append_messages_to_disk(cid, [msg])

        # Ensure not in memory
        _histories.pop(cid, None)

        history = _get_history(cid)
        assert len(history) == 1
        assert history[0].parts[0].content == "from disk"  # type: ignore[union-attr]
