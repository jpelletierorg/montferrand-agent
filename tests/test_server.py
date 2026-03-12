"""Tests for the FastAPI webhook server.

Uses FastAPI's TestClient — no real Twilio or LLM calls.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from montferrand_agent.conversation import ConversationError
from montferrand_agent.models import Dialog
from montferrand_agent.server import _is_duplicate, _seen_sids, app
from montferrand_agent.tenant import load_tenant_profile, save_tenant_profile

from .conftest import ADMIN_TOKEN, CUSTOMER_NUMBER, TEST_PROFILE, TWILIO_NUMBER

_PROCESS_MESSAGE = "montferrand_agent.server.process_message"
_SEND_SMS = "montferrand_agent.server._send_sms"

_SMS_FORM = {
    "To": TWILIO_NUMBER,
    "From": CUSTOMER_NUMBER,
    "Body": "J'ai une fuite",
    "MessageSid": "SM0001",
}


@pytest.fixture
def client():
    """Create a test client with Twilio validation disabled."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


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

    def test_valid_upsert(self, client: TestClient, isolated_tenant_dir: Path):
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

    def test_upsert_overwrites(self, client: TestClient, isolated_tenant_dir: Path):
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

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        _append_messages_to_disk("conv1", [msg], TWILIO_NUMBER)
        _append_messages_to_disk("conv2", [msg], TWILIO_NUMBER)

        response = client.delete(
            f"/admin/tenants/{TWILIO_NUMBER}/conversations",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["deleted"] == 2

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
    def _disable_twilio_validation(self, monkeypatch: pytest.MonkeyPatch):
        """Disable Twilio signature validation for all SMS tests."""
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)

    @pytest.fixture(autouse=True)
    def _clear_dedup_state(self):
        """Clear MessageSid dedup cache before each test."""
        _seen_sids.clear()
        yield
        _seen_sids.clear()

    def test_missing_to_or_from(self, client: TestClient):
        response = client.post("/sms", data={"Body": "hello"})
        assert response.status_code == 400

    def test_no_tenant_config(self, client: TestClient, isolated_tenant_dir: Path):
        response = client.post("/sms", data=_SMS_FORM)
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
            patch(_SEND_SMS, new_callable=AsyncMock) as mock_send,
        ):
            response = client.post("/sms", data=_SMS_FORM)

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

    def test_agent_error_sends_friendly_sms(self, client: TestClient, sms_tenant: Path):
        """When process_message raises, background task sends an error SMS."""
        with (
            patch(
                _PROCESS_MESSAGE,
                new_callable=AsyncMock,
                side_effect=ConversationError("API timeout"),
            ),
            patch(_SEND_SMS, new_callable=AsyncMock) as mock_send,
        ):
            response = client.post("/sms", data={**_SMS_FORM, "MessageSid": "SM0002"})

            assert response.status_code == 200
            # Response is empty TwiML
            assert "<Response />" in response.text or "<Response/>" in response.text

            # Error message sent via REST API
            mock_send.assert_called_once()
            sent_body: str = mock_send.call_args.kwargs["body"]
            assert "erreur" in sent_body.lower() or "réessayer" in sent_body

    def test_dedup_skips_duplicate_message_sid(
        self, client: TestClient, sms_tenant: Path
    ):
        """A retried MessageSid should be skipped — no background task launched."""
        with (
            patch(_PROCESS_MESSAGE, new_callable=AsyncMock) as mock_pm,
            patch(_SEND_SMS, new_callable=AsyncMock) as mock_send,
        ):
            # First request — should process
            response1 = client.post(
                "/sms", data={**_SMS_FORM, "MessageSid": "SM_DEDUP_TEST"}
            )
            assert response1.status_code == 200
            assert mock_pm.call_count == 1

            # Second request with same MessageSid — should skip
            response2 = client.post(
                "/sms", data={**_SMS_FORM, "MessageSid": "SM_DEDUP_TEST"}
            )
            assert response2.status_code == 200
            # process_message should NOT have been called again
            assert mock_pm.call_count == 1


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
