"""Shared fixtures and constants for the Montferrand test suite."""

from pathlib import Path

import pytest

from montferrand_agent.tenant import save_tenant_profile

# ---------------------------------------------------------------------------
# Test constants — phone numbers, tokens, profiles
# ---------------------------------------------------------------------------

TWILIO_NUMBER = "+15551234567"
CUSTOMER_NUMBER = "+14381112222"
TENANT_PHONE = "+14385551234"
ADMIN_TOKEN = "secret123"
TEST_PROFILE = "You are a plumber."


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def assert_hex_string(value: str, expected_length: int) -> None:
    """Assert *value* is a lowercase hex string of *expected_length* chars."""
    assert len(value) == expected_length
    assert all(c in "0123456789abcdef" for c in value)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_tenant_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point MONTFERRAND_TENANT_DIR to a temp dir and return it."""
    monkeypatch.setenv("MONTFERRAND_TENANT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point MONTFERRAND_DATA_DIR to a temp dir and return it."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def sms_tenant(
    isolated_tenant_dir: Path,
    isolated_data_dir: Path,
) -> Path:
    """Create an SMS tenant and return the tenant dir.

    Sets both MONTFERRAND_TENANT_DIR and MONTFERRAND_DATA_DIR, and
    creates a tenant config for TWILIO_NUMBER with TEST_PROFILE.
    """
    save_tenant_profile(TWILIO_NUMBER, TEST_PROFILE)
    return isolated_tenant_dir
