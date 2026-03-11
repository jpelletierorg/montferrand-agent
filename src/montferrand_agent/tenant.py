"""Tenant configuration management.

Each tenant is identified by a Twilio phone number in E.164 format (e.g.,
``+14385551234``).  The tenant's profile (company-specific information) is
stored as a plain text file on disk.  The file name is a SHA-256 hash
prefix of the phone number.

File format::

    # +14385551234
    <tenant profile text>

The first line is a comment with the phone number for human readability
and reverse lookup.  Everything after is the profile — freeform text
describing the company (name, pricing, service area, hours, etc.) that
gets injected into the master prompt template at runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from montferrand_agent.agent import PROJECT_ROOT, dir_from_env


class TenantNotFoundError(RuntimeError):
    """Raised when no tenant config exists for a given phone number."""


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

_DEFAULT_TENANT_DIR = PROJECT_ROOT / "config" / "tenants"


def tenant_dir() -> Path:
    """Return the directory where tenant config files are stored.

    Reads ``MONTFERRAND_TENANT_DIR`` from the environment, falling back
    to ``config/tenants/`` relative to the project root.
    """
    return dir_from_env("MONTFERRAND_TENANT_DIR", _DEFAULT_TENANT_DIR)


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
# Load / save / list
# ---------------------------------------------------------------------------


def _tenant_path(twilio_number: str) -> Path:
    """Return the full path for a tenant's config file."""
    return tenant_dir() / f"{phone_to_filename(twilio_number)}.txt"


def load_tenant_profile(twilio_number: str) -> str:
    """Load the tenant profile for a phone number.

    Raises:
        TenantNotFoundError: If no config file exists for the number.
    """
    path = _tenant_path(twilio_number)
    if not path.exists():
        raise TenantNotFoundError(
            f"No tenant config for {twilio_number} (expected {path})"
        )

    text = path.read_text(encoding="utf-8")

    # Strip the header comment (first line starting with #)
    lines = text.split("\n", maxsplit=1)
    if lines and lines[0].startswith("#"):
        return lines[1] if len(lines) > 1 else ""

    return text


def save_tenant_profile(twilio_number: str, profile: str) -> Path:
    """Save (or overwrite) the tenant profile for a phone number.

    Creates the tenant directory if it doesn't exist.
    Returns the path of the written file.
    """
    path = _tenant_path(twilio_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# {twilio_number}\n{profile}"
    path.write_text(content, encoding="utf-8")
    return path


def list_tenants() -> list[tuple[str, Path]]:
    """List all configured tenants.

    Returns a list of ``(phone_number, file_path)`` tuples.  The phone
    number is read from the comment header in each file.
    """
    directory = tenant_dir()
    if not directory.exists():
        return []

    results: list[tuple[str, Path]] = []
    for path in sorted(directory.glob("*.txt")):
        first_line = path.read_text(encoding="utf-8").split("\n", maxsplit=1)[0]
        if first_line.startswith("# "):
            phone = first_line[2:].strip()
            results.append((phone, path))
    return results
