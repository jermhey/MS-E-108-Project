"""
config.py — Secure Configuration Loader
=========================================
Reads secrets and runtime parameters from a ``.env`` file (via python-dotenv)
so that credentials never appear in source code.

Usage
-----
>>> from src.config import settings
>>> settings.kalshi_access_key
'abc123...'
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from dotenv import load_dotenv


# ── locate .env relative to project root ────────────────────────────────────

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_PATH)


# ── typed settings object ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """Immutable container for all environment-sourced config."""

    kalshi_access_key: str
    kalshi_api_secret: str

    # Kalshi API base URLs
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_demo_url: str = "https://demo-api.kalshi.co/trade-api/v2"

    # Chartmetric API
    chartmetric_refresh_token: str = ""
    chartmetric_base_url: str = "https://api.chartmetric.com"
    chartmetric_artist_id: str = ""     # Chartmetric numeric artist ID

    # Artist (configurable via .env or CLI override)
    artist_name: str = "Bad Bunny"

    # Paths
    project_root: pathlib.Path = _PROJECT_ROOT


def load_settings(*, require_secrets: bool = True) -> Settings:
    """
    Build a ``Settings`` instance from the environment.

    Parameters
    ----------
    require_secrets : bool
        If True (default), raise if Kalshi keys are missing.
        Set to False for offline / data-only workflows.
    """
    access_key = os.getenv("KALSHI_ACCESS_KEY", "")
    api_secret = os.getenv("KALSHI_API_SECRET", "")

    if require_secrets and (not access_key or not api_secret):
        raise EnvironmentError(
            "Missing KALSHI_ACCESS_KEY and/or KALSHI_API_SECRET. "
            "Copy .env.example → .env and fill in your credentials."
        )

    return Settings(
        kalshi_access_key=access_key,
        kalshi_api_secret=api_secret,
        chartmetric_refresh_token=os.getenv("CHARTMETRIC_REFRESH_TOKEN", ""),
        chartmetric_artist_id=os.getenv("CHARTMETRIC_ARTIST_ID", ""),
        artist_name=os.getenv("ARTIST_NAME", "Bad Bunny"),
    )


# Convenience: pre-loaded instance (secrets optional at import time)
settings = load_settings(require_secrets=False)
