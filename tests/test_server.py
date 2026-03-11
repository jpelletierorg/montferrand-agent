"""Tests for the FastAPI webhook server.

Uses FastAPI's TestClient — no real Twilio or LLM calls.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from montferrand_agent.conversation import ConversationError
from montferrand_agent.models import Dialog
from montferrand_agent.server import app
from montferrand_agent.tenant import load_tenant_profile, save_tenant_profile

from .conftest import ADMIN_TOKEN, CUSTOMER_NUMBER, TEST_PROFILE, TWILIO_NUMBER

_PROCESS_MESSAGE = "montferrand_agent.server.process_message"

_SMS_FORM = {
    "To": TWILIO_NUMBER,
    "From": CUSTOMER_NUMBER,
    "Body": "J'ai une fuite",
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
# POST /sms
# ---------------------------------------------------------------------------


class TestSmsWebhook:
    @pytest.fixture(autouse=True)
    def _disable_twilio_validation(self, monkeypatch: pytest.MonkeyPatch):
        """Disable Twilio signature validation for all SMS tests."""
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)

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
        mock_dialog = Dialog(message="Bonjour! Comment puis-je vous aider?")

        with patch(
            _PROCESS_MESSAGE,
            new_callable=AsyncMock,
            return_value=mock_dialog,
        ) as mock_pm:
            response = client.post("/sms", data=_SMS_FORM)

            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            assert "Bonjour" in response.text

            # Verify process_message was called with correct args
            mock_pm.assert_called_once()
            call_args = mock_pm.call_args
            assert call_args.args[1] == "J'ai une fuite"
            assert call_args.kwargs["tenant_profile"] == TEST_PROFILE

    def test_agent_error_returns_friendly_message(
        self, client: TestClient, sms_tenant: Path
    ):
        with patch(
            _PROCESS_MESSAGE,
            new_callable=AsyncMock,
            side_effect=ConversationError("API timeout"),
        ):
            response = client.post("/sms", data=_SMS_FORM)

            assert response.status_code == 200
            assert "erreur" in response.text.lower() or "réessayer" in response.text

    def test_twiml_response_format(self, client: TestClient, sms_tenant: Path):
        """Verify the response is valid TwiML XML."""
        with patch(
            _PROCESS_MESSAGE,
            new_callable=AsyncMock,
            return_value=Dialog(message="Test reply"),
        ):
            response = client.post("/sms", data={**_SMS_FORM, "Body": "test"})

            assert response.text.startswith("<?xml")
            assert "<Response>" in response.text
            assert "<Message>" in response.text
            assert "Test reply" in response.text
