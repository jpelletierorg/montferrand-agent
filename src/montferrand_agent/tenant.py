"""Tenant configuration management.

Each tenant is identified by a Twilio phone number in E.164 format (e.g.,
``+14385551234``).  The tenant's configuration is stored as a TOML file
on disk.  The file name is a SHA-256 hash prefix of the phone number.

TOML file format::

    phone = "+14385551234"
    boss_numbers = ["+14381112222"]

    [profile]
    text = \"\"\"
    - Business name: Acme Plumbing
    ...
    \"\"\"

Legacy ``.txt`` files (plain text with a ``# +14385551234`` comment
header) are supported as a read-only fallback.  The first call to
``save_tenant_config()`` converts them to TOML.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

from montferrand_agent.config import tenants_dir


class TenantNotFoundError(RuntimeError):
    """Raised when no tenant config exists for a given phone number."""


# ---------------------------------------------------------------------------
# TenantConfig
# ---------------------------------------------------------------------------


@dataclass
class TenantConfig:
    """Full configuration for a single tenant.

    ``phone``         — the tenant's Twilio phone number (E.164).
    ``profile``       — freeform text injected into the agent prompt.
    ``boss_numbers``  — E.164 numbers that get the boss/control-plane
                        prompt instead of the customer prompt.
    """

    phone: str
    profile: str
    boss_numbers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------


def tenant_dir() -> Path:
    """Return the directory where tenant config files are stored."""
    return tenants_dir()


# ---------------------------------------------------------------------------
# Phone number → filename
# ---------------------------------------------------------------------------


def phone_to_filename(number: str) -> str:
    """Deterministic filename for a phone number.

    >>> phone_to_filename("+14385551234")
    'a1b2...'  # 16-char hex prefix of sha256
    """
    return hashlib.sha256(number.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# TOML serialization
# ---------------------------------------------------------------------------


def _tenant_path(twilio_number: str) -> Path:
    """Return the full path for a tenant's TOML config file."""
    return tenant_dir() / f"{phone_to_filename(twilio_number)}.toml"


def _legacy_txt_path(twilio_number: str) -> Path:
    """Return the full path for a tenant's legacy .txt config file."""
    return tenant_dir() / f"{phone_to_filename(twilio_number)}.txt"


def _serialize_config(config: TenantConfig) -> bytes:
    """Serialize a TenantConfig to TOML bytes."""
    data: dict = {
        "phone": config.phone,
        "boss_numbers": config.boss_numbers,
        "profile": {"text": config.profile},
    }
    return tomli_w.dumps(data, multiline_strings=True).encode("utf-8")


def _deserialize_config(raw: bytes) -> TenantConfig:
    """Deserialize TOML bytes into a TenantConfig."""
    data = tomllib.loads(raw.decode("utf-8"))
    return TenantConfig(
        phone=data["phone"],
        profile=data.get("profile", {}).get("text", ""),
        boss_numbers=data.get("boss_numbers", []),
    )


def _load_legacy_txt(twilio_number: str) -> TenantConfig | None:
    """Try to load a legacy .txt tenant file, returning None if not found."""
    path = _legacy_txt_path(twilio_number)
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")

    # Strip the header comment (first line starting with #)
    lines = text.split("\n", maxsplit=1)
    if lines and lines[0].startswith("#"):
        profile = lines[1].strip() if len(lines) > 1 else ""
    else:
        profile = text.strip()

    if not profile:
        return None

    return TenantConfig(
        phone=twilio_number,
        profile=profile,
        boss_numbers=[],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_tenant_config(config: TenantConfig) -> Path:
    """Save (or overwrite) a tenant config as TOML.

    Creates the tenant directory if it doesn't exist.
    Returns the path of the written file.
    """
    path = _tenant_path(config.phone)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialize_config(config))
    return path


def load_tenant_config(twilio_number: str) -> TenantConfig:
    """Load the full tenant config for a phone number.

    Tries the TOML file first, then falls back to legacy ``.txt``.

    Raises:
        TenantNotFoundError: If no config file exists for the number.
    """
    # Try TOML first
    path = _tenant_path(twilio_number)
    if path.exists():
        config = _deserialize_config(path.read_bytes())
        if not config.profile:
            raise TenantNotFoundError(
                f"Tenant config for {twilio_number} is empty (file exists but "
                f"contains no profile text): {path}"
            )
        return config

    # Fall back to legacy .txt
    config = _load_legacy_txt(twilio_number)
    if config is not None:
        return config

    raise TenantNotFoundError(
        f"No tenant config for {twilio_number} "
        f"(expected {path} or {_legacy_txt_path(twilio_number)})"
    )


def load_tenant_profile(twilio_number: str) -> str:
    """Load the tenant profile text for a phone number.

    Convenience wrapper around ``load_tenant_config()`` for callers that
    only need the profile string.

    Raises:
        TenantNotFoundError: If no config file exists for the number.
    """
    return load_tenant_config(twilio_number).profile


def save_tenant_profile(twilio_number: str, profile: str) -> Path:
    """Save (or overwrite) the tenant profile for a phone number.

    Convenience wrapper for backward compatibility.  Creates a
    TenantConfig with no boss numbers.  If a config already exists,
    preserves the existing boss_numbers.
    """
    # Try to preserve existing boss_numbers
    try:
        existing = load_tenant_config(twilio_number)
        boss_numbers = existing.boss_numbers
    except TenantNotFoundError:
        boss_numbers = []

    config = TenantConfig(
        phone=twilio_number,
        profile=profile,
        boss_numbers=boss_numbers,
    )
    return save_tenant_config(config)


def is_boss(twilio_number: str, from_number: str) -> bool:
    """Check if *from_number* is a boss number for the given tenant."""
    config = load_tenant_config(twilio_number)
    return from_number in config.boss_numbers


def list_tenants() -> list[tuple[str, Path]]:
    """List all configured tenants.

    Returns a list of ``(phone_number, file_path)`` tuples.
    Reads TOML files first, then legacy .txt files.
    """
    directory = tenant_dir()
    if not directory.exists():
        return []

    results: list[tuple[str, Path]] = []
    seen_phones: set[str] = set()

    # TOML files
    for path in sorted(directory.glob("*.toml")):
        try:
            config = _deserialize_config(path.read_bytes())
            if config.phone not in seen_phones:
                results.append((config.phone, path))
                seen_phones.add(config.phone)
        except Exception:
            pass

    # Legacy .txt files (only if not already seen via TOML)
    for path in sorted(directory.glob("*.txt")):
        first_line = path.read_text(encoding="utf-8").split("\n", maxsplit=1)[0]
        if first_line.startswith("# "):
            phone = first_line[2:].strip()
            if phone not in seen_phones:
                results.append((phone, path))
                seen_phones.add(phone)

    return results
