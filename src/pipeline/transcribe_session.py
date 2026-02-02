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
    chunks_json: Path


def find_latest_session_dir(
    sessions_root: Path = Path("outputs") / "audio" / "sessions",
) -> Optional[Path]:
    # returns the most recently modified session directory
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
        chunks_json=session_dir / "chunks.json",
    )


def load_session_json(session_json_path: Path) -> Dict[str, Any]:
    data = json.loads(Path(session_json_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("session.json did not contain an object")
    return data


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
    Transcribes a session directory and writes transcripts.json.

    Chunked sessions:
      - reads chunks.json
      - transcribes wavs under chunks/####/
      - writes per-file items plus a flattened event list

    Legacy sessions:
      - transcribes wavs in the session root
      - writes only per-file items
    """
    log = logger or logging.getLogger("irc.pipeline.transcribe")
    paths = resolve_session_paths(session_dir)

    if not paths.session_json.exists():
        raise FileNotFoundError(f"Missing session.json: {paths.session_json}")

    if paths.transcripts_json.exists() and not overwrite:
        log.info("transcripts.json already exists: %s", paths.transcripts_json.as_posix())
        return paths.transcripts_json

    session = load_session_json(paths.session_json)
    transcriber = WhisperTranscriber(api_key=api_key, model=model, logger=log)

    started_at = time.time()

    chunk_mode = paths.chunks_json.exists()
    items: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    if chunk_mode:
        chunks = _load_chunks(paths.chunks_json)
        if not chunks:
            raise FileNotFoundError(f"chunks.json exists but is empty: {paths.chunks_json.as_posix()}")

        for ch in chunks:
            chunk_id, chunk_dir_rel, chunk_started_at_unix, chunk_ended_at_unix = _parse_chunk_record(ch)
            if chunk_id is None:
                continue

            chunk_dir = paths.session_dir / chunk_dir_rel
            if not chunk_dir.exists():
                continue

            # audio_index.json gives us user_id/speaker_label per wav
            file_meta = _build_chunk_file_meta(chunk_dir)

            for wav_path in _list_wavs(chunk_dir):
                meta = file_meta.get(wav_path.name, {})
                speaker_label = meta.get("label") if isinstance(meta.get("label"), str) else None
                user_id = meta.get("user_id") if isinstance(meta.get("user_id"), int) else None

                log.info("Transcribing chunk %04d: %s", chunk_id, f"{chunk_dir_rel}/{wav_path.name}")

                result = transcriber.transcribe_wav(
                    wav_path,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    want_timestamps=want_timestamps,
                )

                record = _result_to_record(filename=f"{chunk_dir_rel}/{wav_path.name}", result=result)
                record.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_dir": chunk_dir_rel,
                        "chunk_started_at_unix": float(chunk_started_at_unix),
                        "chunk_ended_at_unix": float(chunk_ended_at_unix),
                        "speaker_label": speaker_label,
                        "user_id": user_id,
                    }
                )
                items.append(record)
                events.extend(_flatten_events(record, result, chunk_started_at_unix))

        items.sort(key=lambda r: (r.get("chunk_id", 0), str(r.get("audio_filename", ""))))
        events.sort(
            key=lambda e: (
                e.get("chunk_id", 0),
                1 if e.get("abs_start_unix") is None else 0,
                float(e.get("abs_start_unix") or 0.0),
                str(e.get("audio_filename", "")),
            )
        )

    else:
        wavs = _list_wavs(paths.session_dir)
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {paths.session_dir.as_posix()}")

        for wav_path in wavs:
            log.info("Transcribing %s", wav_path.name)

            result = transcriber.transcribe_wav(
                wav_path,
                language=language,
                prompt=prompt,
                temperature=temperature,
                want_timestamps=want_timestamps,
            )

            items.append(_result_to_record(wav_path.name, result))

    finished_at = time.time()

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": finished_at,
        "elapsed_s": round(finished_at - started_at, 3),
        "model": model,
        "language": language,
        "chunk_mode": bool(chunk_mode),
        "session": {
            "session_id": session.get("session_id"),
            "guild_id": session.get("guild_id"),
            "voice_channel_id": session.get("voice_channel_id"),
            "started_at_unix": session.get("started_at_unix"),
            "ended_at_unix": session.get("ended_at_unix"),
            "duration_s": session.get("duration_s"),
        },
        "items": items,
        "events": events if chunk_mode else [],
        "notes": (
            [
                "chunk_mode=true: audio_filename includes chunk_dir prefix (ex: chunks/0003/user.wav)",
                "events are ordered by chunk_id then abs_start_unix when available",
                "abs_*_unix is approximated as chunk_started_at_unix + segment.start; per-user trimming can shift this",
            ]
            if chunk_mode
            else [
                "chunk_mode=false: timestamps are per-file only (no shared session timeline)",
            ]
        ),
    }

    paths.transcripts_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote %s", paths.transcripts_json.as_posix())
    return paths.transcripts_json


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


def _list_wavs(dir_path: Path) -> List[Path]:
    # ignore mixed.wav if you ever decide to add a combined track later
    wavs = sorted(dir_path.glob("*.wav"))
    return [p for p in wavs if p.name.lower() not in {"mixed.wav"}]


def _load_chunks(chunks_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(chunks_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("chunks.json did not contain a list")
    return data


def _parse_chunk_record(ch: Dict[str, Any]) -> Tuple[Optional[int], str, float, float]:
    chunk_id = ch.get("chunk_id")
    chunk_dir_rel = ch.get("chunk_dir")
    chunk_started_at_unix = ch.get("started_at_unix")
    chunk_ended_at_unix = ch.get("ended_at_unix")

    if not isinstance(chunk_id, int) or not isinstance(chunk_dir_rel, str):
        return None, "", 0.0, 0.0

    if not isinstance(chunk_started_at_unix, (int, float)) or not isinstance(chunk_ended_at_unix, (int, float)):
        return None, "", 0.0, 0.0

    return chunk_id, chunk_dir_rel, float(chunk_started_at_unix), float(chunk_ended_at_unix)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _build_chunk_file_meta(chunk_dir: Path) -> Dict[str, Dict[str, Any]]:
    # maps filename -> metadata from audio_index.json
    idx = _load_json_if_exists(chunk_dir / "audio_index.json") or {}
    out: Dict[str, Dict[str, Any]] = {}

    items = idx.get("items")
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        filename = it.get("filename")
        if isinstance(filename, str):
            out[filename] = it

    return out


def _result_to_record(filename: str, result: TranscriptionResult) -> Dict[str, Any]:
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


def _flatten_events(
    record: Dict[str, Any],
    result: TranscriptionResult,
    chunk_started_at_unix: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if result.segments:
        for seg in result.segments:
            start_rel = seg.start
            end_rel = seg.end
            abs_start = (chunk_started_at_unix + float(start_rel)) if start_rel is not None else None
            abs_end = (chunk_started_at_unix + float(end_rel)) if end_rel is not None else None

            out.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "audio_filename": record.get("audio_filename"),
                    "speaker_label": record.get("speaker_label"),
                    "user_id": record.get("user_id"),
                    "start_rel_s": start_rel,
                    "end_rel_s": end_rel,
                    "abs_start_unix": abs_start,
                    "abs_end_unix": abs_end,
                    "text": seg.text,
                }
            )
        return out

    # fallback: no segments available (json mode), so treat whole file as one event
    out.append(
        {
            "chunk_id": record.get("chunk_id"),
            "audio_filename": record.get("audio_filename"),
            "speaker_label": record.get("speaker_label"),
            "user_id": record.get("user_id"),
            "start_rel_s": None,
            "end_rel_s": None,
            "abs_start_unix": None,
            "abs_end_unix": None,
            "text": result.text,
        }
    )
    return out
