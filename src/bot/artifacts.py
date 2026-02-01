from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionArtifact:
    # schema
    schema_version: int

    # identity
    session_id: str
    guild_id: int
    voice_channel_id: int
    started_by_user_id: int

    # timing
    started_at_unix: float
    ended_at_unix: Optional[float]
    duration_s: Optional[float]

    # participants + audio
    participants: List[Dict[str, Any]]
    audio_files: List[Dict[str, Any]]

    # other session artifacts produced by the pipeline
    artifacts: Dict[str, str]

    # notes for debugging
    notes: List[str]


def write_session_json(
    base_dir: Path,
    artifact: SessionArtifact,
    *,
    filename: str = "session.json",
) -> Path:
    # ensure output dir exists
    base_dir.mkdir(parents=True, exist_ok=True)

    path = base_dir / filename
    payload = asdict(artifact)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
