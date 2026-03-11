"""Tests for the tenant configuration module."""

from pathlib import Path

import pytest

from montferrand_agent.tenant import (
    TenantNotFoundError,
    list_tenants,
    load_tenant_profile,
    phone_to_filename,
    save_tenant_profile,
    tenant_dir,
)

from .conftest import TENANT_PHONE, TEST_PROFILE, assert_hex_string


# ---------------------------------------------------------------------------
# phone_to_filename
# ---------------------------------------------------------------------------


class TestPhoneToFilename:
    def test_returns_16_char_hex(self):
        assert_hex_string(phone_to_filename(TENANT_PHONE), 16)

    def test_deterministic(self):
        assert phone_to_filename(TENANT_PHONE) == phone_to_filename(TENANT_PHONE)

    def test_different_numbers_differ(self):
        assert phone_to_filename(TENANT_PHONE) != phone_to_filename("+15145559999")


# ---------------------------------------------------------------------------
# tenant_dir
# ---------------------------------------------------------------------------


class TestTenantDir:
    def test_default_is_config_tenants(self):
        directory = tenant_dir()
        assert directory.name == "tenants"
        assert directory.parent.name == "config"

    def test_env_override(self, isolated_tenant_dir: Path):
        assert tenant_dir() == isolated_tenant_dir

    def test_empty_env_uses_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_TENANT_DIR", "")
        directory = tenant_dir()
        assert directory.name == "tenants"


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_creates_file(self, isolated_tenant_dir: Path):
        path = save_tenant_profile(TENANT_PHONE, TEST_PROFILE)
        assert path.exists()
        assert path.suffix == ".txt"

    def test_save_creates_parent_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        nested = tmp_path / "a" / "b"
        monkeypatch.setenv("MONTFERRAND_TENANT_DIR", str(nested))
        path = save_tenant_profile(TENANT_PHONE, "profile")
        assert path.exists()

    def test_load_returns_profile(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)
        assert load_tenant_profile(TENANT_PHONE) == TEST_PROFILE

    def test_load_missing_raises(self, isolated_tenant_dir: Path):
        with pytest.raises(TenantNotFoundError):
            load_tenant_profile("+10000000000")

    def test_save_overwrites_existing(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, "version 1")
        save_tenant_profile(TENANT_PHONE, "version 2")
        assert load_tenant_profile(TENANT_PHONE) == "version 2"

    def test_multiline_profile(self, isolated_tenant_dir: Path):
        profile = "Line one.\nLine two.\nLine three."
        save_tenant_profile(TENANT_PHONE, profile)
        assert load_tenant_profile(TENANT_PHONE) == profile


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------


class TestFileFormat:
    def test_file_starts_with_comment(self, isolated_tenant_dir: Path):
        path = save_tenant_profile(TENANT_PHONE, "profile")
        content = path.read_text()
        assert content.startswith(f"# {TENANT_PHONE}\n")

    def test_load_strips_comment_header(self, isolated_tenant_dir: Path):
        """Manually write a file with a comment and verify load strips it."""
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text(f"# {TENANT_PHONE}\nThe actual profile.")
        assert load_tenant_profile(TENANT_PHONE) == "The actual profile."

    def test_load_file_without_comment_returns_full_text(
        self, isolated_tenant_dir: Path
    ):
        """A file missing the # header returns its full content."""
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text("No comment header here.")
        assert load_tenant_profile(TENANT_PHONE) == "No comment header here."


# ---------------------------------------------------------------------------
# list_tenants
# ---------------------------------------------------------------------------


class TestListTenants:
    def test_empty_dir(self, isolated_tenant_dir: Path):
        assert list_tenants() == []

    def test_nonexistent_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("MONTFERRAND_TENANT_DIR", str(tmp_path / "nope"))
        assert list_tenants() == []

    def test_lists_saved_tenants(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, "profile A")
        save_tenant_profile("+15145559999", "profile B")
        result = list_tenants()
        phones = [phone for phone, _ in result]
        assert TENANT_PHONE in phones
        assert "+15145559999" in phones
        assert len(result) == 2

    def test_ignores_files_without_comment(self, isolated_tenant_dir: Path):
        (isolated_tenant_dir / "junk.txt").write_text("no comment header")
        assert list_tenants() == []
