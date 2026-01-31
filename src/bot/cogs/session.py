from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
from discord.ext import commands

from src.bot.config import get_settings
from src.bot.permissions import gate_guild, deny
from src.bot.artifacts import SessionArtifact, write_session_json


@dataclass
class ActiveSession:
    session_id: str
    guild_id: int
    voice_channel_id: int
    started_by_user_id: int
    started_at: float
    voice_client: discord.VoiceClient
    sink: discord.sinks.Sink
    finished_event: asyncio.Event


class SessionCog(commands.Cog):
    def __init__(self, bot: discord.Bot) -> None:
        self.bot = bot
        self.settings = get_settings()
        self.log = logging.getLogger("irc.session")

        # keep one active session per guild for simplicity
        self.active_by_guild: Dict[int, ActiveSession] = {}

    @discord.slash_command(name="start", description="Start recording in your current voice channel.")
    async def start(self, ctx: discord.ApplicationContext) -> None:
        # gate first so random servers can't fill your disk
        gate = gate_guild(ctx)
        if not gate.allowed:
            await deny(ctx, gate.reason or "Not allowed.")
            return

        guild = ctx.guild
        author = ctx.author
        if guild is None:
            await ctx.respond("This command can only be used in a server.", ephemeral=True)
            return

        if guild.id in self.active_by_guild:
            await ctx.respond(
                "A recording session is already active in this server. Use /stop first.",
                ephemeral=True,
            )
            return

        # figure out which voice channel the user is currently in
        voice_state = getattr(author, "voice", None)
        voice_channel = getattr(voice_state, "channel", None)

        if voice_channel is None or not isinstance(voice_channel, discord.VoiceChannel):
            await ctx.respond("Join a voice channel first, then run /start.", ephemeral=True)
            return

        # create outputs folder for this session
        session_id = f"{int(time.time())}_{guild.id}_{voice_channel.id}"
        base_dir = Path("outputs") / "audio" / "sessions" / session_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # join (or move to) the voice channel
        try:
            voice_client = await voice_channel.connect()
        except discord.ClientException:
            # already connected somewhere else, try moving
            vc = guild.voice_client
            if vc is None:
                await ctx.respond("Failed to connect to the voice channel.", ephemeral=True)
                return
            voice_client = vc
            await voice_client.move_to(voice_channel)
        except Exception:
            self.log.exception("Failed to connect to voice channel.")
            await ctx.respond("Failed to connect to the voice channel.", ephemeral=True)
            return

        # small delay helps the voice connection settle before recording starts
        await asyncio.sleep(0.5)

        sink = discord.sinks.WaveSink()
        finished_event = asyncio.Event()

        async def _on_recording_finished(sink_: discord.sinks.Sink, *args: Any) -> None:
            # this runs after stop_recording() finishes collecting audio
            try:
                self._save_sink_audio(sink_, base_dir)
                self.log.info("Saved session audio to %s", base_dir.as_posix())
            except Exception:
                self.log.exception("Failed while saving recorded audio.")
            finally:
                finished_event.set()

        try:
            voice_client.start_recording(sink, _on_recording_finished)
        except Exception:
            self.log.exception("Failed to start recording.")
            await ctx.respond("Failed to start recording.", ephemeral=True)
            try:
                await voice_client.disconnect()
            except Exception:
                pass
            return

        self.active_by_guild[guild.id] = ActiveSession(
            session_id=session_id,
            guild_id=guild.id,
            voice_channel_id=voice_channel.id,
            started_by_user_id=author.id if isinstance(author, discord.Member) else 0,
            started_at=time.time(),
            voice_client=voice_client,
            sink=sink,
            finished_event=finished_event,
        )

        await ctx.respond(
            f"Recording started in **{voice_channel.name}**. Use **/stop** to end the session.",
            ephemeral=False,
        )

    @discord.slash_command(name="stop", description="Stop the active recording session for this server.")
    async def stop(self, ctx: discord.ApplicationContext) -> None:
        gate = gate_guild(ctx)
        if not gate.allowed:
            await deny(ctx, gate.reason or "Not allowed.")
            return

        guild = ctx.guild
        if guild is None:
            await ctx.respond("This command can only be used in a server.", ephemeral=True)
            return

        session = self.active_by_guild.get(guild.id)
        if session is None:
            await ctx.respond("No active recording session found. Use /start first.", ephemeral=True)
            return

        ended_at = time.time()

        # stop recording triggers the callback that saves audio
        try:
            session.voice_client.stop_recording()
        except Exception:
            self.log.exception("Failed to stop recording.")
            await ctx.respond("Failed to stop recording.", ephemeral=True)
            return

        # wait for the recording-finished callback to complete writing wav files
        try:
            await asyncio.wait_for(session.finished_event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            self.log.warning("Timed out waiting for audio save callback; proceeding anyway.")

        # disconnect after stopping
        try:
            await session.voice_client.disconnect()
        except Exception:
            self.log.exception("Failed to disconnect after recording.")

        # write session.json
        session_dir = Path("outputs") / "audio" / "sessions" / session.session_id
        participants: List[Dict[str, Any]] = []
        audio_files: List[Dict[str, Any]] = []

        try:
            if session_dir.exists():
                for p in sorted(session_dir.glob("*.wav")):
                    audio_files.append({"filename": p.name, "bytes": p.stat().st_size})

            for f in audio_files:
                participants.append({"speaker_label": f["filename"].replace(".wav", "")})

            artifact = SessionArtifact(
                session_id=session.session_id,
                guild_id=session.guild_id,
                voice_channel_id=session.voice_channel_id,
                started_by_user_id=session.started_by_user_id,
                started_at_unix=session.started_at,
                ended_at_unix=ended_at,
                participants=participants,
                audio_files=audio_files,
                notes=[
                    "per-speaker wav files may be voice-activity trimmed; timestamps are not aligned yet",
                    "audio_index.json contains per-file wav durations (header repaired if needed)",
                ],
            )

            write_session_json(session_dir, artifact)
            self.log.info("Wrote session.json for session %s", session.session_id)
        except Exception:
            self.log.exception("Failed to write session.json")

        duration_s = int(ended_at - session.started_at)
        self.active_by_guild.pop(guild.id, None)

        await ctx.respond(f"Recording stopped. Duration: **{duration_s}s**.", ephemeral=False)

    def _save_sink_audio(self, sink: discord.sinks.Sink, base_dir: Path) -> None:
        audio_data = getattr(sink, "audio_data", None)
        if not audio_data:
            (base_dir / "EMPTY_SESSION.txt").write_text(
                "No audio data was captured. This can happen if nobody spoke, or if voice receive isn't working.\n",
                encoding="utf-8",
            )
            return

        keys = list(audio_data.keys())
        pretty = []
        for k in keys:
            pretty.append(
                f"{type(k).__name__}:id={getattr(k, 'id', None)} name={getattr(k, 'name', None)} repr={k!r}"
            )
        self.log.info("Recorded sink keys: %s", " | ".join(pretty))

        index_items: List[Dict[str, Any]] = []

        for idx, (speaker_key, data) in enumerate(audio_data.items(), start=1):
            if isinstance(speaker_key, int):
                user_id = speaker_key
                username = None
            else:
                user_id = getattr(speaker_key, "id", None)
                username = getattr(speaker_key, "name", None)

            safe_name = ""
            if isinstance(username, str) and username.strip():
                cleaned = "".join(ch for ch in username if ch.isalnum() or ch in ("-", "_"))[:24]
                if cleaned:
                    safe_name = f"_{cleaned}"

            if isinstance(user_id, int):
                filename = f"{user_id}{safe_name}.wav"
                label = str(user_id)
            else:
                filename = f"unknown_{idx}.wav"
                label = f"unknown_{idx}"

            out_path = base_dir / filename

            # calling cleanup() after finished can raise; we just skip
            try:
                is_finished = bool(getattr(data, "finished", False))
                if not is_finished and hasattr(data, "cleanup") and callable(getattr(data, "cleanup")):
                    data.cleanup()
            except Exception as e:
                self.log.debug("cleanup() skipped/failed for %s (%s)", label, type(e).__name__)

            file_obj = getattr(data, "file", None)
            if file_obj is None:
                self.log.warning("No file object for %s; skipping.", label)
                continue

            try:
                file_obj.seek(0)
                blob = file_obj.read()
            except Exception:
                self.log.exception("Failed reading audio bytes for %s; skipping.", label)
                continue

            if not blob:
                self.log.warning("Empty audio buffer for %s; skipping.", label)
                continue

            out_path.write_bytes(blob)

            # repair wav header sizes if py-cord left them as 0
            repaired = _repair_wav_header_if_needed(out_path)
            if repaired:
                self.log.info("Repaired wav header for %s -> %s", label, out_path.name)

            self.log.info("Wrote %s (%d bytes) -> %s", label, len(blob), out_path.name)

            wav_info = _read_wav_info(out_path)
            sink_dbg = _extract_sink_debug(data)

            index_items.append(
                {
                    "label": label,
                    "user_id": user_id if isinstance(user_id, int) else None,
                    "username": username if isinstance(username, str) else None,
                    "filename": out_path.name,
                    "bytes": len(blob),
                    "wav": wav_info,
                    "sink_timing": sink_dbg,
                }
            )

        audio_index_path = base_dir / "audio_index.json"
        payload = {
            "created_at_unix": time.time(),
            "items": index_items,
            "notes": [
                "wav durations come from wave module after optional header repair",
                "sink_timing is a best-effort dump of attrs; timing offsets are not exposed in current sink objects",
            ],
        }
        audio_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.log.info("Wrote audio_index.json -> %s", audio_index_path.as_posix())


def _read_wav_info(path: Path) -> Dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration_s = (frames / float(rate)) if rate else None
            return {
                "channels": channels,
                "sample_rate_hz": rate,
                "sample_width_bytes": sampwidth,
                "frames": frames,
                "duration_s": float(duration_s) if duration_s is not None else None,
            }
    except Exception:
        return {
            "channels": None,
            "sample_rate_hz": None,
            "sample_width_bytes": None,
            "frames": None,
            "duration_s": None,
            "error": "failed_to_read_wav_header",
        }


def _extract_sink_debug(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    try:
        attrs = [a for a in dir(obj) if not a.startswith("_")]
        out["available_attrs_sample"] = attrs[:60]
    except Exception:
        out["available_attrs_sample"] = []

    if hasattr(obj, "finished"):
        try:
            out["finished"] = bool(getattr(obj, "finished"))
        except Exception:
            out["finished"] = None

    return out


def _repair_wav_header_if_needed(path: Path) -> bool:
    # returns true if we modified the file
    data = path.read_bytes()
    if len(data) < 44:
        return False

    # must be riff/wave
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False

    # walk chunks to find data chunk offset + declared size
    # riff header is 12 bytes, then repeated: 4-byte id + 4-byte size + payload (padded to even)
    pos = 12
    data_chunk_size_offset: Optional[int] = None
    data_payload_offset: Optional[int] = None
    declared_data_size: Optional[int] = None

    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        payload_start = pos + 8
        payload_end = payload_start + chunk_size

        if chunk_id == b"data":
            data_chunk_size_offset = pos + 4
            data_payload_offset = payload_start
            declared_data_size = chunk_size
            break

        # chunks are padded to even sizes
        pos = payload_end + (chunk_size % 2)

    if data_chunk_size_offset is None or data_payload_offset is None:
        return False

    actual_data_size = max(0, len(data) - data_payload_offset)

    # if header already matches, do nothing
    if declared_data_size == actual_data_size:
        return False

    # only auto-repair when declared size is clearly wrong (most commonly 0)
    if declared_data_size not in (0, None):
        return False

    # patch riff size (file size - 8) and data chunk size (actual payload bytes)
    patched = bytearray(data)
    struct.pack_into("<I", patched, 4, len(data) - 8)
    struct.pack_into("<I", patched, data_chunk_size_offset, actual_data_size)
    path.write_bytes(patched)
    return True


def setup(bot: discord.Bot) -> None:
    bot.add_cog(SessionCog(bot))
