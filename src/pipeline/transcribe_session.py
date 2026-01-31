from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.stt.whisper_api import TranscriptionResult, WhisperTranscriber


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    session_json: Path
    transcripts_json: Path


def find_latest_session_dir(sessions_root: Path = Path("outputs") / "audio" / "sessions") -> Optional[Path]:
    # finds the most recently modified session folder
    if not sessions_root.exists():
        return None
    
    dirs = [p for p in sessions_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]

def resolve_session_paths(session_dir: Path) -> SessionPaths:
    session_dir = Path(session_dir)
    return SessionPaths(
        session_dir=session_dir,
        session_json=session_dir / "session.json",
        transcripts_json=session_dir / "transcripts.json",
    )

def load_session_json(session_json_path: Path) -> Dict[str, Any]:
    # loads the session.json artifact written by the bot
    data = json.loads(Path(session_json_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("session.json did not contain an object")
    return data

def list_audio_files(session_dir: Path) -> List[Path]:
    # returns wav files in a stable order
    session_dir = Path(session_dir)
    wavs = sorted(session_dir.glob("*.wav"))

    # ignore any future mixed file naming
    wavs = [p for p in wavs if p.name.lower() not in {"mixed.wav"}]
    return wavs

def transcribe_session_dir(
        session_dir: Path,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-transcribe",
        language: Optional[str] = "en",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        want_timestamps: bool = True,
        overwrite: bool = False,
        logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Transcribes all wav files in a session directory and writes transcripts.json.

    Returns the path to the transcripts.json file.
    """
    log = logger or logging.getLogger("irc.pipeline.transcribe")
    paths = resolve_session_paths(session_dir)

    if not paths.session_json.exists():
        raise FileNotFoundError(f"Missing session.json: {paths.session_json}")
    
    if paths.transcripts_json.exists() and not overwrite:
        log.info("transcripts.json already exists: %s", paths.transcripts_json.as_posix())
        return paths.transcripts_json
    
    session = load_session_json(paths.session_json)
    audio_files = list_audio_files(paths.session_dir)

    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {paths.session_dir.as_posix()}")
    
    transcriber = WhisperTranscriber(api_key=api_key, model=model, logger=log)

    outputs: List[Dict[str, Any]] = []
    started_at = time.time()

    for wav_path in audio_files:
        log.info("transcribing %s", wav_path.name)
        result = transcriber.transcribe_wav(
            wav_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            want_timestamps=want_timestamps,
        )
        outputs.append(_result_to_record(wav_path.name, result))

    finished_at = time.time()

    payload: Dict[str, Any] = {
        "session_id": session.get("session_id"),
        "guild_id": session.get("guild_id"),
        "voice_channel_id": session.get("voice_channel_id"),
        "started_at_unix": session.get("started_at_unix"),
        "ended_at_unix": session.get("ended_at_unix"),
        "created_at_unix": finished_at,
        "elapsed_s": round(finished_at - started_at, 3),
        "model": model,
        "language": language,
        "items": outputs,
        "notes": [
            "per-file timestamps are relative to each file, not aligned to a shared session timeline yet",
        ],
    }

    paths.transcripts_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote %s", paths.transcripts_json.as_posix())
    return paths.transcripts_json

def _result_to_record(filename: str, result: TranscriptionResult) -> Dict[str, Any]:
    # normalizes our transcriber output to a JSON-friendly record
    return {
        "audio_filename": filename,
        "text": result.text,
        "model": result.model,
        "response_format": result.response_format,
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in result.segments
        ],
    }

def transcribe_latest_session(
        *,
        sessions_root: Path = Path("outputs") / "audio" / "sessions",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-transcribe",
        language: Optional[str] = "en",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        want_timestamps: bool = True,
        overwrite: bool = False,
        logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Path]:
    """
    Wrapper for local testing. Finds the latest session dir and transcribes it.
    """
    session_dir = find_latest_session_dir(sessions_root=sessions_root)
    if session_dir is None:
        raise FileNotFoundError(f"No session directories found under {sessions_root.as_posix()}")

    out = transcribe_session_dir(
        session_dir,
        api_key=api_key,
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        want_timestamps=want_timestamps,
        overwrite=overwrite,
        logger=logger,
    )
    return session_dir, out