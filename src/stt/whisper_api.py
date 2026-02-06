from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    text: str = ""


class TranscriptionResult(BaseModel):
    # normalized result that the rest of the app can rely on
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None

    # debugging/info fields
    model: str
    response_format: str
    timestamps_supported: bool = False


class WhisperTranscriber:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-transcribe",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.log = logger or logging.getLogger("irc.stt")

        # cache whether this model supports verbose_json timestamps
        # None = unknown, True/False = known
        self._verbose_json_supported: Optional[bool] = None
        self._warned_no_timestamps: bool = False

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = "en",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        want_timestamps: bool = True,
    ) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga", ".oga"}:
            self.log.warning("Uncommon audio extension for stt: %s", audio_path.suffix)

        # attempt timestamps only if caller wants them and we haven't proven they are unsupported
        if want_timestamps and self._verbose_json_supported is not False:
            try:
                res = self._transcribe(
                    audio_path,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                )
                self._verbose_json_supported = True
                # mark in the output for downstream code
                res.timestamps_supported = True
                return res
            except Exception as e:
                if _is_verbose_json_unsupported(e):
                    # permanently disable verbose_json timestamps for this run
                    self._verbose_json_supported = False
                    if not self._warned_no_timestamps:
                        self.log.warning(
                            "Model '%s' does not support response_format='verbose_json'; "
                            "Disabling timestamp attempts and using response_format='json' instead",
                            self.model,
                        )
                        self._warned_no_timestamps = True
                else:
                    # transient/unknown failure: keep the cache unknown so a future file could still succeed
                    self.log.warning("Failed to get timestamps from stt (falling back to json): %s", str(e))

        # fallback to plain json (no timestamps)
        res = self._transcribe(
            audio_path,
            response_format="json",
            timestamp_granularities=None,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )
        res.timestamps_supported = False
        return res

    def _transcribe(
        self,
        audio_path: Path,
        *,
        response_format: str,
        timestamp_granularities: Optional[list[str]],
        language: Optional[str],
        prompt: Optional[str],
        temperature: float,
    ) -> TranscriptionResult:
        with open(audio_path, "rb") as f:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "file": f,
                "response_format": response_format,
                "temperature": temperature,
            }

            if language:
                kwargs["language"] = language

            if prompt:
                kwargs["prompt"] = prompt

            if timestamp_granularities and response_format == "verbose_json":
                kwargs["timestamp_granularities"] = timestamp_granularities

            resp = self.client.audio.transcriptions.create(**kwargs)

        # normalize to dict
        if hasattr(resp, "model_dump"):
            data = resp.model_dump()
        elif isinstance(resp, dict):
            data = resp
        else:
            data = {
                "text": getattr(resp, "text", ""),
                "segments": getattr(resp, "segments", None),
                "language": getattr(resp, "language", None),
                "duration": getattr(resp, "duration", None),
            }

        text = str(data.get("text", "") or "")
        language_out = data.get("language")
        duration_out = data.get("duration")

        segments_out: list[TranscriptSegment] = []
        raw_segments = data.get("segments")
        if isinstance(raw_segments, list):
            for s in raw_segments:
                if not isinstance(s, dict):
                    continue
                segments_out.append(
                    TranscriptSegment(
                        start=_as_float(s.get("start")),
                        end=_as_float(s.get("end")),
                        text=str(s.get("text", "") or ""),
                    )
                )

        return TranscriptionResult(
            text=text,
            segments=segments_out,
            language=str(language_out) if isinstance(language_out, str) else None,
            duration=_as_float(duration_out),
            model=self.model,
            response_format=response_format,
            timestamps_supported=(response_format == "verbose_json" and len(segments_out) > 0),
        )


def _is_verbose_json_unsupported(exc: Exception) -> bool:
    # detect "no verbose_json support" once and stop attempting it
    msg = str(exc)
    return (
        "response_format 'verbose_json' is not compatible with model" in msg
        or "param': 'response_format'" in msg
        or "code': 'unsupported_value'" in msg
    )


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None