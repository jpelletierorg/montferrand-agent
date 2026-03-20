"""Tests for the FastAPI webhook server.

Uses FastAPI's TestClient — no real Twilio or LLM calls.
"""

from pathlib import Path
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.conversation import ConversationError
from montferrand_agent.crm import tenant_crm_db_path
from montferrand_agent.models import Dialog
from montferrand_agent.ops import get_message_timeline
from montferrand_agent.server import _is_duplicate, _seen_sids, app
from montferrand_agent.tenant import load_tenant_profile, save_tenant_profile

from .conftest import ADMIN_TOKEN, CUSTOMER_NUMBER, TEST_PROFILE, TWILIO_NUMBER

_PROCESS_MESSAGE = "montferrand_agent.server.process_message"
_SEND_SMS = "montferrand_agent.server._send_sms"
_TWILIO_AUTH_TOKEN = "test-twilio-auth-token"

_SMS_FORM = {
    "To": TWILIO_NUMBER,
    "From": CUSTOMER_NUMBER,
    "Body": "J'ai une fuite",
    "MessageSid": "SM0001",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, isolated_data_dir: Path):
    """Create a test client with Twilio webhook validation enabled."""

    monkeypatch.setenv("TWILIO_AUTH_TOKEN", _TWILIO_AUTH_TOKEN)
    with TestClient(app) as test_client:
        yield test_client


def _twilio_headers(path: str, form: dict[str, str]) -> dict[str, str]:
    """Return a valid Twilio signature header for a test request."""

    validator = RequestValidator(_TWILIO_AUTH_TOKEN)
    signature = validator.compute_signature(f"https://testserver{path}", form)
    return {"X-Twilio-Signature": signature}


def _post_twilio_form(client: TestClient, path: str, form: dict[str, str]):
    """POST form data with a valid Twilio signature."""

    return client.post(path, data=form, headers=_twilio_headers(path, form))


def _wait_until(predicate, *, timeout: float = 1.0, interval: float = 0.01) -> None:
    """Poll until *predicate* returns True or fail the test."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Condition not met before timeout")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_startup_requires_twilio_auth_token(
        self, monkeypatch: pytest.MonkeyPatch, isolated_data_dir: Path
    ):
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)

        with pytest.raises(RuntimeError, match="TWILIO_AUTH_TOKEN must be set"):
            with TestClient(app):
                pass

    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_livez(self, client: TestClient):
        response = client.get("/livez")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_readyz_reports_degraded_without_runtime_config(
        self,
        client: TestClient,
        isolated_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("INCEPTION_API_KEY", raising=False)

        response = client.get("/readyz")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert any(
            check["name"] == "ops_db" and check["ok"] for check in data["checks"]
        )

    def test_readyz_ok_when_local_dependencies_are_configured(
        self,
        client: TestClient,
        isolated_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "auth-token")
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "openrouter")
        monkeypatch.setenv("MONTFERRAND_MODEL", "anthropic/claude-sonnet-4.6")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        response = client.get("/readyz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert all(check["ok"] for check in data["checks"])


# ---------------------------------------------------------------------------
# POST /admin/tenants
# ---------------------------------------------------------------------------


class TestAdminTenants:
    @pytest.fixture(autouse=True)
    def _set_admin_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_ADMIN_TOKEN", ADMIN_TOKEN)

    def test_missing_auth_header(self, client: TestClient):
        response = client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "test"},
        )
        assert response.status_code == 401

    def test_wrong_token(self, client: TestClient):
        response = client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "test"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 403

    def test_no_admin_token_configured(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("MONTFERRAND_ADMIN_TOKEN", raising=False)
        response = client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "test"},
            headers={"Authorization": "Bearer whatever"},
        )
        assert response.status_code == 500

    def test_valid_upsert(
        self, client: TestClient, isolated_tenant_dir: Path, fake_dbmate
    ):
        response = client.post(
            "/admin/tenants",
            json={
                "twilio_number": TWILIO_NUMBER,
                "tenant_profile": TEST_PROFILE,
            },
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert response.status_code == 201
        assert response.json()["status"] == "ok"

        # Verify the file was actually written
        assert load_tenant_profile(TWILIO_NUMBER) == TEST_PROFILE
        assert get_tenant_calendar(TWILIO_NUMBER).directory.exists()
        assert tenant_crm_db_path(TWILIO_NUMBER).exists()

    def test_upsert_overwrites(
        self, client: TestClient, isolated_tenant_dir: Path, fake_dbmate
    ):
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

        client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "v1"},
            headers=headers,
        )
        client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "v2"},
            headers=headers,
        )

        assert load_tenant_profile(TWILIO_NUMBER) == "v2"

    def test_upsert_existing_tenant_with_missing_crm_returns_conflict(
        self, client: TestClient, isolated_tenant_dir: Path, fake_dbmate
    ):
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

        first = client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "v1"},
            headers=headers,
        )
        tenant_crm_db_path(TWILIO_NUMBER).unlink()

        second = client.post(
            "/admin/tenants",
            json={"twilio_number": TWILIO_NUMBER, "tenant_profile": "v2"},
            headers=headers,
        )

        assert first.status_code == 201
        assert second.status_code == 409
        assert "CRM DB is missing" in second.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /admin/tenants/{twilio_number}/conversations
# ---------------------------------------------------------------------------


class TestAdminResetConversations:
    @pytest.fixture(autouse=True)
    def _set_admin_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_ADMIN_TOKEN", ADMIN_TOKEN)

    def test_reset_returns_deleted_count(self, client: TestClient, sms_tenant: Path):
        """DELETE endpoint returns the number of conversations deleted."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        from montferrand_agent.conversation import _append_messages_to_disk

        backend = get_tenant_calendar(TWILIO_NUMBER)
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("conv1", [msg], TWILIO_NUMBER)
        _append_messages_to_disk("conv2", [msg], TWILIO_NUMBER)
        block = backend.create_block(
            "2030-03-17",
            "09:00",
            "12:00",
            "Blocked",
            "Vacation",
        )
        assert block.success is True

        response = client.delete(
            f"/admin/tenants/{TWILIO_NUMBER}/conversations",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["deleted"] == 2
        assert "reset the calendar" in data["message"]
        assert backend.directory.exists()
        assert list(backend.directory.glob("*.ics")) == []

    def test_reset_requires_auth(self, client: TestClient):
        response = client.delete(
            f"/admin/tenants/{TWILIO_NUMBER}/conversations",
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /sms
# ---------------------------------------------------------------------------


class TestSmsWebhook:
    @pytest.fixture(autouse=True)
    def _clear_dedup_state(self):
        """Clear MessageSid dedup cache before each test."""
        _seen_sids.clear()
        yield
        _seen_sids.clear()

    def test_missing_to_or_from(self, client: TestClient):
        form = {"Body": "hello"}
        response = _post_twilio_form(client, "/sms", form)
        assert response.status_code == 400

    def test_missing_twilio_signature_rejected(self, client: TestClient):
        response = client.post("/sms", data=_SMS_FORM)
        assert response.status_code == 403

    def test_invalid_twilio_signature_rejected(self, client: TestClient):
        response = client.post(
            "/sms",
            data=_SMS_FORM,
            headers={"X-Twilio-Signature": "invalid"},
        )
        assert response.status_code == 403

    def test_no_tenant_config(self, client: TestClient, isolated_tenant_dir: Path):
        response = _post_twilio_form(client, "/sms", _SMS_FORM)
        assert response.status_code == 200
        assert "application/xml" in response.headers["content-type"]
        # Should contain a polite error message
        assert "pas configuré" in response.text or "pas config" in response.text

    def test_successful_sms_turn(self, client: TestClient, sms_tenant: Path):
        """Webhook returns empty TwiML; background task sends reply via REST API."""
        mock_dialog = Dialog(message="Bonjour! Comment puis-je vous aider?")

        with (
            patch(
                _PROCESS_MESSAGE,
                new_callable=AsyncMock,
                return_value=mock_dialog,
            ) as mock_pm,
            patch(
                _SEND_SMS, new_callable=AsyncMock, return_value="SM_OUTBOUND_1"
            ) as mock_send,
        ):
            response = _post_twilio_form(client, "/sms", _SMS_FORM)
            _wait_until(lambda: mock_pm.call_count == 1)
            _wait_until(lambda: mock_send.call_count == 1)

            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            # Response is empty TwiML — no <Message> element
            assert "<Response />" in response.text or "<Response/>" in response.text

            # Background task should have called process_message
            mock_pm.assert_called_once()
            call_args = mock_pm.call_args
            assert call_args.args[1] == "J'ai une fuite"
            assert call_args.kwargs["tenant_profile"] == TEST_PROFILE
            assert call_args.kwargs["twilio_number"] == TWILIO_NUMBER

            # Background task should have sent the reply via Twilio REST API
            mock_send.assert_called_once_with(
                to=CUSTOMER_NUMBER,
                from_=TWILIO_NUMBER,
                body="Bonjour! Comment puis-je vous aider?",
            )

        timeline = get_message_timeline(message_sid="SM0001")
        assert timeline is not None
        message, events = timeline
        assert message.last_stage == "outbound_accepted"
        assert message.outbound_message_sid == "SM_OUTBOUND_1"
        assert [event.event_kind for event in events] == [
            "received",
            "processing_started",
            "processing_succeeded",
            "outbound_attempted",
            "outbound_accepted",
        ]

    def test_agent_error_sends_friendly_sms(self, client: TestClient, sms_tenant: Path):
        """When process_message raises, background task sends an error SMS."""
        with (
            patch(
                _PROCESS_MESSAGE,
                new_callable=AsyncMock,
                side_effect=ConversationError("API timeout"),
            ),
            patch(
                _SEND_SMS, new_callable=AsyncMock, return_value="SM_OUTBOUND_2"
            ) as mock_send,
        ):
            form = {**_SMS_FORM, "MessageSid": "SM0002"}
            response = _post_twilio_form(client, "/sms", form)
            _wait_until(lambda: mock_send.call_count == 1)

            assert response.status_code == 200
            # Response is empty TwiML
            assert "<Response />" in response.text or "<Response/>" in response.text

            # Error message sent via REST API
            mock_send.assert_called_once()
            sent_body: str = mock_send.call_args.kwargs["body"]
            assert "erreur" in sent_body.lower() or "réessayer" in sent_body

        timeline = get_message_timeline(message_sid="SM0002")
        assert timeline is not None
        message, events = timeline
        assert message.last_stage == "outbound_accepted"
        assert any(event.event_kind == "processing_failed" for event in events)
        assert message.error_text == "API timeout"

    def test_dedup_skips_duplicate_message_sid(
        self, client: TestClient, sms_tenant: Path
    ):
        """A retried MessageSid should be skipped — no background task launched."""
        with (
            patch(_PROCESS_MESSAGE, new_callable=AsyncMock) as mock_pm,
            patch(
                _SEND_SMS, new_callable=AsyncMock, return_value="SM_OUTBOUND_4"
            ) as mock_send,
        ):
            # First request — should process
            form = {**_SMS_FORM, "MessageSid": "SM_DEDUP_TEST"}
            response1 = _post_twilio_form(client, "/sms", form)
            _wait_until(lambda: mock_pm.call_count == 1)
            assert response1.status_code == 200
            assert mock_pm.call_count == 1

            # Second request with same MessageSid — should skip
            response2 = _post_twilio_form(client, "/sms", form)
            assert response2.status_code == 200
            # process_message should NOT have been called again
            assert mock_pm.call_count == 1

        timeline = get_message_timeline(message_sid="SM_DEDUP_TEST")
        assert timeline is not None
        _message, events = timeline
        assert any(event.event_kind == "duplicate" for event in events)

    def test_twilio_status_callback_updates_delivery_state(
        self, client: TestClient, sms_tenant: Path
    ):
        with (
            patch(
                _PROCESS_MESSAGE,
                new_callable=AsyncMock,
                return_value=Dialog(message="Bonjour!"),
            ),
            patch(_SEND_SMS, new_callable=AsyncMock, return_value="SM_OUTBOUND_3"),
        ):
            form = {**_SMS_FORM, "MessageSid": "SM0003"}
            response = _post_twilio_form(client, "/sms", form)
            _wait_until(
                lambda: (
                    (timeline := get_message_timeline(message_sid="SM0003")) is not None
                    and timeline[0].last_stage == "outbound_accepted"
                )
            )

        assert response.status_code == 200

        callback_form = {
            "MessageSid": "SM_OUTBOUND_3",
            "MessageStatus": "delivered",
        }
        callback = _post_twilio_form(client, "/twilio/status", callback_form)

        assert callback.status_code == 200
        timeline = get_message_timeline(message_sid="SM0003")
        assert timeline is not None
        message, events = timeline
        assert message.last_stage == "delivery_delivered"
        assert any(event.event_kind == "delivery_status" for event in events)


# ---------------------------------------------------------------------------
# _is_duplicate unit tests
# ---------------------------------------------------------------------------


class TestIsDuplicate:
    @pytest.fixture(autouse=True)
    def _clear_dedup_state(self):
        """Clear the dedup cache before each test."""
        _seen_sids.clear()
        yield
        _seen_sids.clear()

    def test_second_sid_is_duplicate(self):
        assert _is_duplicate("SM_TWICE") is False
        assert _is_duplicate("SM_TWICE") is True
