"""Tests for the tenant configuration module."""

from pathlib import Path

import pytest

from montferrand_agent.tenant import (
    TenantConfig,
    TenantNotFoundError,
    is_boss,
    list_tenants,
    load_tenant_config,
    load_tenant_profile,
    phone_to_filename,
    save_tenant_config,
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

    def test_different_numbers_differ(self):
        assert phone_to_filename(TENANT_PHONE) != phone_to_filename("+15145559999")


# ---------------------------------------------------------------------------
# tenant_dir
# ---------------------------------------------------------------------------


class TestTenantDir:
    def test_missing_env_var_crashes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MONTFERRAND_DATA_DIR", raising=False)
        with pytest.raises(RuntimeError, match="MONTFERRAND_DATA_DIR"):
            tenant_dir()

    def test_returns_tenants_subdir(self, isolated_data_dir: Path):
        directory = tenant_dir()
        assert directory == isolated_data_dir / "tenants"


# ---------------------------------------------------------------------------
# save / load roundtrip (TOML format)
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_creates_toml_file(self, isolated_tenant_dir: Path):
        config = TenantConfig(phone=TENANT_PHONE, profile=TEST_PROFILE)
        path = save_tenant_config(config)
        assert path.exists()
        assert path.suffix == ".toml"

    def test_save_creates_parent_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        nested = tmp_path / "a" / "b"
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(nested))
        config = TenantConfig(phone=TENANT_PHONE, profile="profile")
        path = save_tenant_config(config)
        assert path.exists()

    def test_load_returns_config(self, isolated_tenant_dir: Path):
        config = TenantConfig(
            phone=TENANT_PHONE,
            profile=TEST_PROFILE,
            boss_numbers=["+14381112222"],
        )
        save_tenant_config(config)

        loaded = load_tenant_config(TENANT_PHONE)
        assert loaded.phone == TENANT_PHONE
        assert loaded.profile == TEST_PROFILE
        assert loaded.boss_numbers == ["+14381112222"]

    def test_load_profile_returns_text(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)
        assert load_tenant_profile(TENANT_PHONE) == TEST_PROFILE

    def test_load_missing_raises(self, isolated_tenant_dir: Path):
        with pytest.raises(TenantNotFoundError):
            load_tenant_config("+10000000000")

    def test_save_overwrites_existing(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, "version 1")
        save_tenant_profile(TENANT_PHONE, "version 2")
        assert load_tenant_profile(TENANT_PHONE) == "version 2"

    def test_multiline_profile(self, isolated_tenant_dir: Path):
        profile = "Line one.\nLine two.\nLine three."
        save_tenant_profile(TENANT_PHONE, profile)
        assert load_tenant_profile(TENANT_PHONE) == profile

    def test_save_profile_preserves_boss_numbers(self, isolated_tenant_dir: Path):
        """save_tenant_profile() preserves existing boss_numbers."""
        config = TenantConfig(
            phone=TENANT_PHONE,
            profile="v1",
            boss_numbers=["+14381112222"],
        )
        save_tenant_config(config)

        # Update profile via the convenience wrapper
        save_tenant_profile(TENANT_PHONE, "v2")

        loaded = load_tenant_config(TENANT_PHONE)
        assert loaded.profile == "v2"
        assert loaded.boss_numbers == ["+14381112222"]

    def test_default_boss_numbers_empty(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)
        loaded = load_tenant_config(TENANT_PHONE)
        assert loaded.boss_numbers == []


# ---------------------------------------------------------------------------
# TOML file format
# ---------------------------------------------------------------------------


class TestFileFormat:
    def test_file_is_valid_toml(self, isolated_tenant_dir: Path):
        import tomllib

        config = TenantConfig(phone=TENANT_PHONE, profile="profile text")
        path = save_tenant_config(config)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert data["phone"] == TENANT_PHONE
        assert data["profile"]["text"] == "profile text"

    def test_boss_numbers_in_toml(self, isolated_tenant_dir: Path):
        import tomllib

        config = TenantConfig(
            phone=TENANT_PHONE,
            profile="profile",
            boss_numbers=["+14381112222", "+14389876543"],
        )
        path = save_tenant_config(config)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert data["boss_numbers"] == ["+14381112222", "+14389876543"]


# ---------------------------------------------------------------------------
# Legacy .txt fallback
# ---------------------------------------------------------------------------


class TestLegacyTxtFallback:
    def test_loads_legacy_txt(self, isolated_tenant_dir: Path):
        """A legacy .txt file should load with empty boss_numbers."""
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text(f"# {TENANT_PHONE}\nThe actual profile.")

        config = load_tenant_config(TENANT_PHONE)
        assert config.profile == "The actual profile."
        assert config.boss_numbers == []

    def test_legacy_without_comment_returns_full_text(self, isolated_tenant_dir: Path):
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text("No comment header here.")

        config = load_tenant_config(TENANT_PHONE)
        assert config.profile == "No comment header here."

    def test_toml_takes_priority_over_txt(self, isolated_tenant_dir: Path):
        """When both .toml and .txt exist, TOML wins."""
        txt_path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        txt_path.write_text(f"# {TENANT_PHONE}\nfrom txt")

        config = TenantConfig(phone=TENANT_PHONE, profile="from toml")
        save_tenant_config(config)

        loaded = load_tenant_config(TENANT_PHONE)
        assert loaded.profile == "from toml"


# ---------------------------------------------------------------------------
# Empty / missing profile detection
# ---------------------------------------------------------------------------


class TestEmptyProfile:
    def test_empty_toml_profile_raises(self, isolated_tenant_dir: Path):
        """A TOML file with empty profile text must crash."""
        import tomli_w

        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.toml"
        path.write_bytes(
            tomli_w.dumps(
                {"phone": TENANT_PHONE, "boss_numbers": [], "profile": {"text": ""}}
            ).encode()
        )
        with pytest.raises(TenantNotFoundError, match="empty"):
            load_tenant_config(TENANT_PHONE)

    def test_legacy_comment_only_raises(self, isolated_tenant_dir: Path):
        """A legacy .txt file with only a comment header."""
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text(f"# {TENANT_PHONE}\n")
        with pytest.raises(TenantNotFoundError):
            load_tenant_config(TENANT_PHONE)

    def test_legacy_whitespace_only_raises(self, isolated_tenant_dir: Path):
        """A legacy .txt file with only whitespace."""
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text("   \n  \n")
        with pytest.raises(TenantNotFoundError):
            load_tenant_config(TENANT_PHONE)


# ---------------------------------------------------------------------------
# is_boss
# ---------------------------------------------------------------------------


class TestIsBoss:
    def test_boss_number_returns_true(self, isolated_tenant_dir: Path):
        config = TenantConfig(
            phone=TENANT_PHONE,
            profile="profile",
            boss_numbers=["+14381112222"],
        )
        save_tenant_config(config)
        assert is_boss(TENANT_PHONE, "+14381112222") is True

    def test_non_boss_returns_false(self, isolated_tenant_dir: Path):
        config = TenantConfig(
            phone=TENANT_PHONE,
            profile="profile",
            boss_numbers=["+14381112222"],
        )
        save_tenant_config(config)
        assert is_boss(TENANT_PHONE, "+14389999999") is False

    def test_no_boss_numbers_returns_false(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, "profile")
        assert is_boss(TENANT_PHONE, "+14381112222") is False


# ---------------------------------------------------------------------------
# list_tenants
# ---------------------------------------------------------------------------


class TestListTenants:
    def test_empty_dir(self, isolated_tenant_dir: Path):
        assert list_tenants() == []

    def test_nonexistent_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path / "nope"))
        assert list_tenants() == []

    def test_lists_saved_tenants(self, isolated_tenant_dir: Path):
        save_tenant_profile(TENANT_PHONE, "profile A")
        save_tenant_profile("+15145559999", "profile B")
        result = list_tenants()
        phones = [phone for phone, _ in result]
        assert TENANT_PHONE in phones
        assert "+15145559999" in phones
        assert len(result) == 2

    def test_lists_legacy_txt_tenants(self, isolated_tenant_dir: Path):
        """Legacy .txt files should appear in tenant listing."""
        path = isolated_tenant_dir / f"{phone_to_filename(TENANT_PHONE)}.txt"
        path.write_text(f"# {TENANT_PHONE}\nprofile text")
        result = list_tenants()
        phones = [phone for phone, _ in result]
        assert TENANT_PHONE in phones
