from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    # discord
    discord_bot_token: str

    # storage
    database_url: str
    retention_hours: int

    # behavior defaults
    default_summary_window_min: int
    max_summary_window_min: int
    audio_chunk_seconds: int

    # LLM / STT
    openai_api_key: Optional[str]
    openai_chat_model: str
    openai_stt_model: str

    # paths
    outputs_dir: Path


_CACHED: Optional[Settings] = None


def _require_env(name: str) -> str:
    # helper to force required env vars
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}."
            "Check your .env file (or environment) and try again."
        )
    return value

def _get_int(name: str, default: int, *, min_value: Optional[int] = None) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid value for {name}: expected an integer, got '{raw}'.") from e
    
    if min_value is not None and value < min_value:
        raise RuntimeError(f"Invalid value for {name}: must be >= {min_value}, got {value}.")
    
    return value

def _get_settings() -> Settings:
    """
    Loads env vars once and caches them.

    Call this from anywhere to get the same global settings.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    
    # load .env into envrionment if present
    load_dotenv()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    discord_bot_token = _require_env("DISCORD_BOT_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None

    database_url = os.getenv("DATABASE_URL", "sqlite:///outputs/irc.sqlite").strip()
    retention_hours = _get_int("RETENTION_HOURS", 24, min_value=1)

    default_summary_window_min = _get_int("DEFAULT_SUMMARY_WINDOW_MIN", 20, min_value=1)
    max_summary_window_min = _get_int("MAX_SUMMARY_WINDOW_MIN", 120, min_value=1)
    audio_chunk_seconds = _get_int("AUDIO_CHUNK_SECONDS", 10, min_value=1)

    openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    openai_stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe").strip()

    # quick consistency check
    if default_summary_window_min > max_summary_window_min:
        raise RuntimeError(
            "DEFAULT_SUMMARY_WINDOW_MIN cannot be greater than MAX_SUMMARY_WINDOW_MIN."
        )

    _CACHED = Settings(
        discord_bot_token=discord_bot_token,
        database_url=database_url,
        retention_hours=retention_hours,
        default_summary_window_min=default_summary_window_min,
        max_summary_window_min=max_summary_window_min,
        audio_chunk_seconds=audio_chunk_seconds,
        openai_api_key=openai_api_key,
        openai_chat_model=openai_chat_model,
        openai_stt_model=openai_stt_model,
        outputs_dir=outputs_dir,
    )

    return _CACHED