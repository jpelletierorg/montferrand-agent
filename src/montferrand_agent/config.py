"""Centralized data-directory configuration.

Every module that needs a persistent directory imports from here.
There is a single environment variable: ``MONTFERRAND_DATA_DIR``.
The program crashes immediately if it is not set — an unset variable
means the deployment is misconfigured.

Directory layout::

    $MONTFERRAND_DATA_DIR/
        tenants/          # tenant config .txt files
        conversations/    # NDJSON conversation histories
        calendars/        # vdir .ics event files
"""

from __future__ import annotations

import os
from pathlib import Path


def _require_data_dir() -> Path:
    """Return the root data directory or crash."""
    value = os.getenv("MONTFERRAND_DATA_DIR", "").strip()
    if not value:
        raise RuntimeError(
            "MONTFERRAND_DATA_DIR is not set.  "
            "Set it in .env or the environment (e.g. /opt/montferrand)."
        )
    return Path(value)


def data_dir() -> Path:
    """Return the root data directory."""
    return _require_data_dir()


def tenants_dir() -> Path:
    """Return the directory for tenant config files."""
    return _require_data_dir() / "tenants"


def conversations_dir() -> Path:
    """Return the directory for conversation NDJSON files."""
    return _require_data_dir() / "conversations"


def calendars_dir() -> Path:
    """Return the directory for calendar vdir files."""
    return _require_data_dir() / "calendars"
