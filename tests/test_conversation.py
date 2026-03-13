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
    _tenant_data_dir,
    conversation_key_for_sms,
    get_cost,
    list_conversations,
    new_conversation_id,
    reset,
    reset_tenant,
)

from .conftest import CUSTOMER_NUMBER, TWILIO_NUMBER, assert_hex_string

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# Test tenant number for all persistence tests
_TN = TWILIO_NUMBER


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
        assert _get_history("nonexistent", _TN) == []

    def test_save_and_get_roundtrip(self):
        cid = new_conversation_id()
        msgs = ["msg1", "msg2"]  # type: ignore[list-item]
        _save_history(cid, msgs, 0, _TN)  # type: ignore[arg-type]
        assert _get_history(cid, _TN) == msgs

    def test_get_history_returns_copy(self):
        cid = new_conversation_id()
        msgs = ["msg1"]  # type: ignore[list-item]
        _save_history(cid, msgs, 0, _TN)  # type: ignore[arg-type]
        retrieved = _get_history(cid, _TN)
        retrieved.append("extra")  # type: ignore[arg-type]
        assert _get_history(cid, _TN) == msgs  # original unaffected

    def test_reset_clears_history(self):
        cid = new_conversation_id()
        _save_history(cid, ["msg"], 0, _TN)  # type: ignore[arg-type]
        reset(cid, _TN)
        assert _get_history(cid, _TN) == []

    def test_reset_unknown_id_does_not_raise(self):
        reset("nonexistent", _TN)  # should not raise


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
        reset(cid, _TN)
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

    @pytest.mark.parametrize("images", [None, []])
    def test_no_images_returns_string(self, images):
        result = _build_prompt("hello", images)
        assert result == "hello"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# conversation_key_for_sms
# ---------------------------------------------------------------------------


class TestConversationKeyForSms:
    def test_returns_12_char_hex(self):
        assert_hex_string(conversation_key_for_sms(TWILIO_NUMBER, CUSTOMER_NUMBER), 12)

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

    def test_missing_env_var_crashes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MONTFERRAND_DATA_DIR", raising=False)
        with pytest.raises(RuntimeError, match="MONTFERRAND_DATA_DIR"):
            _data_dir()

    def test_data_dir_returns_conversations_subdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))
        directory = _data_dir()
        assert directory == tmp_path / "conversations"

    def test_load_from_empty_dir(self):
        assert _load_history_from_disk("nonexistent", _TN) == []

    def test_append_and_load_roundtrip(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("test123", [msg], _TN)

        loaded = _load_history_from_disk("test123", _TN)
        assert len(loaded) == 1
        assert isinstance(loaded[0], ModelRequest)
        assert loaded[0].parts[0].content == "hello"  # type: ignore[union-attr]

    def test_append_is_incremental(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg1 = ModelRequest(parts=[UserPromptPart(content="first")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="second")])

        _append_messages_to_disk("test456", [msg1], _TN)
        _append_messages_to_disk("test456", [msg2], _TN)

        loaded = _load_history_from_disk("test456", _TN)
        assert len(loaded) == 2

    def test_get_history_falls_back_to_disk(self):
        """_get_history loads from disk when not in memory cache."""
        from montferrand_agent.conversation import _histories
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cid = "diskonly123"
        msg = ModelRequest(parts=[UserPromptPart(content="from disk")])
        _append_messages_to_disk(cid, [msg], _TN)

        # Ensure not in memory
        _histories.pop(cid, None)

        history = _get_history(cid, _TN)
        assert len(history) == 1
        assert history[0].parts[0].content == "from disk"  # type: ignore[union-attr]

    def test_corrupted_ndjson_line_is_skipped(self, tmp_path: Path):
        """H4 regression: one bad line should not kill the entire history."""
        from montferrand_agent.conversation import (
            _ModelMessageAdapter,
            _conversation_path,
        )
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cid = "corrupt_test"
        msg = ModelRequest(parts=[UserPromptPart(content="good line")])
        _append_messages_to_disk(cid, [msg], _TN)

        # Inject a corrupted line in the middle
        path = _conversation_path(cid, _TN)
        content = path.read_bytes()
        corrupted = content + b"THIS IS NOT VALID JSON\n"
        # Append another valid line
        msg2 = ModelRequest(parts=[UserPromptPart(content="also good")])
        corrupted += _ModelMessageAdapter.dump_json(msg2) + b"\n"
        path.write_bytes(corrupted)

        loaded = _load_history_from_disk(cid, _TN)
        assert len(loaded) == 2
        assert loaded[0].parts[0].content == "good line"  # type: ignore[union-attr]
        assert loaded[1].parts[0].content == "also good"  # type: ignore[union-attr]

    def test_tenant_scoped_directory_structure(self, tmp_path: Path):
        """Conversation files are stored in tenant-scoped subdirectories."""
        from montferrand_agent.conversation import _conversation_path
        from montferrand_agent.tenant import phone_to_filename

        path = _conversation_path("abc123", _TN)
        assert path.parent.name == phone_to_filename(_TN)
        assert path.name == "abc123.ndjson"


# ---------------------------------------------------------------------------
# reset_tenant
# ---------------------------------------------------------------------------


class TestResetTenant:
    @pytest.fixture(autouse=True)
    def _isolate_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Point NDJSON writes to a temp dir."""
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))

    def test_reset_empty_tenant_returns_zero(self):
        assert reset_tenant(_TN) == 0

    def test_reset_deletes_all_conversations(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("conv_a", [msg], _TN)
        _append_messages_to_disk("conv_b", [msg], _TN)
        _append_messages_to_disk("conv_c", [msg], _TN)

        count = reset_tenant(_TN)
        assert count == 3

        # Directory should be gone
        assert not _tenant_data_dir(_TN).exists()

    def test_reset_does_not_affect_other_tenants(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        other_tenant = "+19999999999"
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("conv_a", [msg], _TN)
        _append_messages_to_disk("conv_b", [msg], other_tenant)

        reset_tenant(_TN)

        # Other tenant's data should still exist
        assert _load_history_from_disk("conv_b", other_tenant) != []
        # Our tenant's data should be gone
        assert _load_history_from_disk("conv_a", _TN) == []

    def test_reset_clears_in_memory_state(self):
        from montferrand_agent.conversation import _costs, _histories
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("conv_mem", [msg], _TN)
        _histories["conv_mem"] = [msg]
        _costs["conv_mem"] = ConversationCost(total_usd=Decimal("0.01"))

        reset_tenant(_TN)

        assert "conv_mem" not in _histories
        assert "conv_mem" not in _costs


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------


class TestListConversations:
    @pytest.fixture(autouse=True)
    def _isolate_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))

    def test_empty_tenant(self):
        assert list_conversations(_TN) == []

    def test_lists_conversation_ids(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("aaa111", [msg], _TN)
        _append_messages_to_disk("bbb222", [msg], _TN)

        result = list_conversations(_TN)
        assert sorted(result) == ["aaa111", "bbb222"]
