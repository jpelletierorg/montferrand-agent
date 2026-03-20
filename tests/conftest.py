"""Shared fixtures and constants for the Montferrand test suite."""

import sqlite3
from pathlib import Path

import pytest

import montferrand_agent.crm as crm_module
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
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point MONTFERRAND_DATA_DIR to a temp dir and return it.

    All subdirectories (tenants/, conversations/, calendars/) live under
    this single root, matching the production layout.
    """
    monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def isolated_tenant_dir(isolated_data_dir: Path) -> Path:
    """Return the tenants/ subdirectory under the isolated data dir.

    Depends on ``isolated_data_dir`` so MONTFERRAND_DATA_DIR is already
    set.  Creates the directory eagerly so tests that write files
    directly into it don't fail.
    """
    d = isolated_data_dir / "tenants"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sms_tenant(
    isolated_tenant_dir: Path,
    isolated_data_dir: Path,
) -> Path:
    """Create an SMS tenant and return the tenant dir.

    Sets MONTFERRAND_DATA_DIR and creates a tenant config for
    TWILIO_NUMBER with TEST_PROFILE.
    """
    save_tenant_profile(TWILIO_NUMBER, TEST_PROFILE)
    return isolated_tenant_dir


def _migration_version(path: Path) -> str:
    return path.name.split("_", maxsplit=1)[0]


def _read_up_sql(path: Path) -> str:
    lines: list[str] = []
    capture = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-- migrate:up"):
            capture = True
            continue
        if line.startswith("-- migrate:down"):
            capture = False
            continue
        if capture:
            lines.append(line)
    return "\n".join(lines).strip()


@pytest.fixture
def fake_dbmate(monkeypatch: pytest.MonkeyPatch):
    """Patch CRM dbmate calls with a lightweight in-process SQLite runner."""

    migration_files = sorted(crm_module.crm_migrations_dir().glob("*.sql"))

    def fake_run_dbmate(db_path: Path, command: str) -> None:
        if command == "up":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        elif not db_path.exists():
            raise RuntimeError(f"missing db for fake dbmate: {db_path}")

        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY)"
        )
        try:
            if command in {"up", "migrate"}:
                applied = {
                    row[0]
                    for row in conn.execute("SELECT version FROM schema_migrations")
                }
                for migration_path in migration_files:
                    version = _migration_version(migration_path)
                    if version in applied:
                        continue
                    script = _read_up_sql(migration_path)
                    if script:
                        conn.executescript(script)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (?)",
                        (version,),
                    )
                conn.commit()
                return

            if command == "status":
                applied = {
                    row[0]
                    for row in conn.execute("SELECT version FROM schema_migrations")
                }
                expected = {_migration_version(path) for path in migration_files}
                if applied != expected:
                    missing = ", ".join(sorted(expected - applied)) or "unknown"
                    raise RuntimeError(f"pending migrations: {missing}")
                return

            raise RuntimeError(f"unsupported fake dbmate command: {command}")
        finally:
            conn.close()

    monkeypatch.setattr(crm_module, "_run_dbmate", fake_run_dbmate)
    return fake_run_dbmate
