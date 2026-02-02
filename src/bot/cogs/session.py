from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import discord
from discord.ext import commands

from src.bot.artifacts import SessionArtifact, write_session_json
from src.bot.chunking import ChunkRecord, append_chunk, chunk_dir as mk_chunk_dir, load_chunks
from src.bot.config import get_settings
from src.bot.permissions import deny, gate_guild

# this file handles voice recording and writes artifacts to outputs/audio/sessions/<session_id>/
# - chunks/####/user.wav
# - chunks/####/audio_index.json
# - chunks.json
# - session.json


@dataclass
class ActiveSession:
    # identity + discord routing
    session_id: str
    guild_id: int
    voice_channel_id: int
    started_by_user_id: int

    # timing (wall clock)
    started_at: float
    chunk_started_at: float

    # voice + filesystem
    voice_client: discord.VoiceClient
    session_dir: Path

    # chunk rotation control
    chunk_seconds: int
    chunk_id: int
    chunk_finished_event: asyncio.Event
    rotate_task: Optional[asyncio.Task]
    stopping: bool


class SessionCog(commands.Cog):
    """
    /start joins your current voice channel and begins recording into fixed-length chunks.
    /stop finalizes the current chunk, writes session.json, and disconnects.

    Design notes:
    - we store separate wavs per user per chunk
    - timeline alignment is approximated by chunk boundaries
    - later, transcripts/events can be ordered by chunk_id (and segment timestamps if available)
    """

    def __init__(self, bot: discord.Bot) -> None:
        self.bot = bot
        self.settings = get_settings()
        self.log = logging.getLogger("irc.session")

        # one active session per guild for now (keeps state simple)
        self.active_by_guild: Dict[int, ActiveSession] = {}

    @discord.slash_command(name="start", description="Start recording in your current voice channel.")
    async def start(self, ctx: discord.ApplicationContext) -> None:
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

        voice_state = getattr(author, "voice", None)
        voice_channel = getattr(voice_state, "channel", None)
        if voice_channel is None or not isinstance(voice_channel, discord.VoiceChannel):
            await ctx.respond("Join a voice channel first, then run /start.", ephemeral=True)
            return

        # create a stable folder per session; chunked audio goes under session_dir/chunks/####/
        session_id = f"{int(time.time())}_{guild.id}_{voice_channel.id}"
        session_dir = Path("outputs") / "audio" / "sessions" / session_id
        (session_dir / "chunks").mkdir(parents=True, exist_ok=True)

        voice_client = await self._connect_to_voice_channel(guild, voice_channel, ctx)
        if voice_client is None:
            return

        # small delay helps the voice connection settle before starting the sink
        await asyncio.sleep(0.5)

        # pull from settings
        chunk_seconds = int(self.settings.audio_chunk_seconds)

        active = ActiveSession(
            session_id=session_id,
            guild_id=guild.id,
            voice_channel_id=voice_channel.id,
            started_by_user_id=author.id if isinstance(author, discord.Member) else 0,
            started_at=time.time(),
            chunk_started_at=time.time(),
            voice_client=voice_client,
            session_dir=session_dir,
            chunk_seconds=chunk_seconds,
            chunk_id=1,
            chunk_finished_event=asyncio.Event(),
            rotate_task=None,
            stopping=False,
        )

        self.active_by_guild[guild.id] = active

        try:
            await self._start_chunk_recording(active)
        except Exception:
            self.log.exception("Failed to start initial chunk recording")
            self.active_by_guild.pop(guild.id, None)
            try:
                await voice_client.disconnect()
            except Exception:
                pass
            await ctx.respond("Failed to start recording.", ephemeral=True)
            return

        active.rotate_task = asyncio.create_task(self._chunk_rotator(active))

        await ctx.respond(
            f"Recording started in **{voice_channel.name}**. Use **/stop** to end.",
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
        session.stopping = True

        # stop the background rotator first so it doesn't race with our final stop
        await self._cancel_rotator(session)

        # stop_recording triggers the sink callback (which saves chunk audio)
        try:
            session.voice_client.stop_recording()
        except Exception:
            self.log.exception("Failed to stop recording")
            await ctx.respond("Failed to stop recording.", ephemeral=True)
            return

        # wait for the final chunk to be written
        try:
            await asyncio.wait_for(session.chunk_finished_event.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            # we still try to salvage whatever got written
            self.log.warning("Timed out waiting for final chunk save; proceeding anyway")

        try:
            await session.voice_client.disconnect()
        except Exception:
            self.log.exception("Failed to disconnect after recording")

        participants, audio_files = _aggregate_manifest_from_chunks(guild=guild, session_dir=session.session_dir)

        # write session.json as the "root" artifact for use later in the pipeline
        try:
            artifact = SessionArtifact(
                schema_version=1,
                session_id=session.session_id,
                guild_id=session.guild_id,
                voice_channel_id=session.voice_channel_id,
                started_by_user_id=session.started_by_user_id,
                started_at_unix=session.started_at,
                ended_at_unix=ended_at,
                duration_s=(ended_at - session.started_at),
                participants=participants,
                audio_files=audio_files,
                artifacts={
                    "session": "session.json",
                    "chunks": "chunks.json",
                    "transcripts": "transcripts.json",
                    "audio_index": "audio_index.json",
                },
                notes=[
                    "audio is stored as fixed wall-clock chunks under chunks/####/",
                    "per-speaker wav files may be packet-trimmed; ordering is approximated by chunk boundaries",
                    "chunk transcriptions can be flattened into events ordered by (chunk_id, segment.start)",
                ],
            )
            write_session_json(session.session_dir, artifact)
            self.log.info("Wrote session.json for session %s", session.session_id)
        except Exception:
            self.log.exception("Failed to write session.json")

        duration_s_int = int(ended_at - session.started_at)
        self.active_by_guild.pop(guild.id, None)

        await ctx.respond(f"Recording stopped. Duration: **{duration_s_int}s**.", ephemeral=False)

    async def _connect_to_voice_channel(
        self,
        guild: discord.Guild,
        voice_channel: discord.VoiceChannel,
        ctx: discord.ApplicationContext,
    ) -> Optional[discord.VoiceClient]:
        # connect, or move if already connected elsewhere
        try:
            return await voice_channel.connect()
        except discord.ClientException:
            vc = guild.voice_client
            if vc is None:
                await ctx.respond("Failed to connect to the voice channel.", ephemeral=True)
                return None
            try:
                await vc.move_to(voice_channel)
                return vc
            except Exception:
                self.log.exception("Failed to move voice client to channel")
                await ctx.respond("Failed to connect to the voice channel.", ephemeral=True)
                return None
        except Exception:
            self.log.exception("Failed to connect to voice channel")
            await ctx.respond("Failed to connect to the voice channel.", ephemeral=True)
            return None

    async def _cancel_rotator(self, session: ActiveSession) -> None:
        if session.rotate_task is None:
            return

        session.rotate_task.cancel()
        try:
            await session.rotate_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.exception("Chunk rotator task errored on cancel")

    async def _chunk_rotator(self, session: ActiveSession) -> None:
        try:
            while not session.stopping:
                await asyncio.sleep(session.chunk_seconds)
                if session.stopping:
                    break

                # clear the event so we can wait for the next callback
                session.chunk_finished_event.clear()

                try:
                    session.voice_client.stop_recording()
                except Exception:
                    self.log.exception("Failed to stop recording for chunk rotation")
                    break

                try:
                    await asyncio.wait_for(session.chunk_finished_event.wait(), timeout=20.0)
                except asyncio.TimeoutError:
                    self.log.warning("Timed out waiting for chunk save during rotation; stopping rotation")
                    break
        except asyncio.CancelledError:
            return

    async def _start_chunk_recording(self, session: ActiveSession) -> None:
        """
        Starts a new wave sink that writes into chunks/<chunk_id>/.
        """
        chunk_id = session.chunk_id
        chunk_started_at = time.time()
        session.chunk_started_at = chunk_started_at
        session.chunk_finished_event.clear()

        cdir = mk_chunk_dir(session.session_dir, chunk_id)
        cdir.mkdir(parents=True, exist_ok=True)

        sink = discord.sinks.WaveSink()

        async def _on_chunk_finished(sink_: discord.sinks.Sink, *args: Any) -> None:
            try:
                self._save_sink_audio(sink_, cdir)

                chunk_ended_at = time.time()
                rel_chunk_dir = f"chunks/{chunk_id:04d}"
                rel_audio_index = f"{rel_chunk_dir}/audio_index.json"

                append_chunk(
                    session.session_dir,
                    ChunkRecord(
                        chunk_id=chunk_id,
                        started_at_unix=chunk_started_at,
                        ended_at_unix=chunk_ended_at,
                        duration_s=(chunk_ended_at - chunk_started_at),
                        chunk_dir=rel_chunk_dir,
                        audio_index=rel_audio_index,
                    ),
                )

                self.log.info("Saved chunk %04d -> %s", chunk_id, cdir.as_posix())
            except Exception:
                self.log.exception("Failed while saving chunk audio")
            finally:
                session.chunk_finished_event.set()

                # auto-start next chunk unless we're stopping
                if not session.stopping:
                    session.chunk_id += 1
                    try:
                        await self._start_chunk_recording(session)
                    except Exception:
                        self.log.exception("Failed to start next chunk recording")
                        session.stopping = True

        session.voice_client.start_recording(sink, _on_chunk_finished)

    def _save_sink_audio(self, sink: discord.sinks.Sink, base_dir: Path) -> None:
        """
        Writes per-speaker wav files for a single chunk and emits audio_index.json.
        """
        audio_data = getattr(sink, "audio_data", None)
        if not audio_data:
            (base_dir / "EMPTY_CHUNK.txt").write_text(
                "No audio data was captured for this chunk.\n",
                encoding="utf-8",
            )
            _write_audio_index(base_dir, [])
            return

        index_items: List[Dict[str, Any]] = []

        for idx, (speaker_key, data) in enumerate(audio_data.items(), start=1):
            user_id, username = _extract_speaker_identity(speaker_key)
            filename, label = _format_speaker_filename(user_id, username, idx)

            out_path = base_dir / filename

            # allow sinks to clean up resources if needed
            try:
                is_finished = bool(getattr(data, "finished", False))
                if not is_finished and callable(getattr(data, "cleanup", None)):
                    data.cleanup()
            except Exception:
                pass

            file_obj = getattr(data, "file", None)
            if file_obj is None:
                continue

            try:
                file_obj.seek(0)
                blob = file_obj.read()
            except Exception:
                continue

            if not blob:
                continue

            out_path.write_bytes(blob)

            repaired = _repair_wav_header_if_needed(out_path)
            wav_info = _read_wav_info(out_path)

            index_items.append(
                {
                    "label": label,
                    "user_id": user_id,
                    "username": username,
                    "filename": out_path.name,
                    "bytes": len(blob),
                    "wav": wav_info,
                    "header_repaired": bool(repaired),
                }
            )

        _write_audio_index(base_dir, index_items)


def _extract_speaker_identity(speaker_key: Any) -> Tuple[Optional[int], Optional[str]]:
    # py-cord uses a few possible key types; normalize them into (user_id, username)
    if isinstance(speaker_key, int):
        return speaker_key, None

    user_id = getattr(speaker_key, "id", None)
    if not isinstance(user_id, int):
        user_id = None

    username = getattr(speaker_key, "name", None)
    if not isinstance(username, str) or not username.strip():
        username = None

    return user_id, username


def _format_speaker_filename(user_id: Optional[int], username: Optional[str], idx: int) -> Tuple[str, str]:
    # keep filenames deterministic and safe on windows/mac/linux
    safe_suffix = ""
    if username:
        cleaned = "".join(ch for ch in username if ch.isalnum() or ch in ("-", "_"))[:24]
        if cleaned:
            safe_suffix = f"_{cleaned}"

    if user_id is not None:
        return f"{user_id}{safe_suffix}.wav", str(user_id)

    return f"unknown_{idx}.wav", f"unknown_{idx}"


def _write_audio_index(base_dir: Path, items: List[Dict[str, Any]]) -> None:
    audio_index_path = base_dir / "audio_index.json"
    payload = {
        "created_at_unix": time.time(),
        "items": items,
        "notes": [
            "this file describes only the wavs present in this chunk directory",
            "durations come from wave module after optional header repair",
        ],
    }
    audio_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _repair_wav_header_if_needed(path: Path) -> bool:
    """
    Returns true if we modified the file.

    Some sinks produce wav headers with data chunk size = 0.
    wave.open() will then report 0 frames even though bytes exist.
    We patch riff size and data size when the declared data size is clearly wrong.
    """
    data = path.read_bytes()
    if len(data) < 44:
        return False

    # must be riff/wave
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False

    # walk chunks: 12-byte header then repeating (id, size, payload)
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

    # only auto-repair when declared size is clearly wrong (common case: 0)
    if declared_data_size not in (0, None):
        return False

    patched = bytearray(data)
    struct.pack_into("<I", patched, 4, len(data) - 8)  # riff chunk size
    struct.pack_into("<I", patched, data_chunk_size_offset, actual_data_size)
    path.write_bytes(patched)
    return True


def _aggregate_manifest_from_chunks(
    *,
    guild: discord.Guild,
    session_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds:
    - participants: unique users observed across all chunks
    - audio_files: all audio files across all chunks (with chunk-relative paths)
    """
    chunks = load_chunks(session_dir)

    participants_by_id: Dict[Optional[int], Dict[str, Any]] = {}
    audio_files: List[Dict[str, Any]] = []

    for ch in chunks:
        ch_dir_rel = ch.get("chunk_dir")
        if not isinstance(ch_dir_rel, str):
            continue

        ch_dir = session_dir / ch_dir_rel
        idx_path = ch_dir / "audio_index.json"
        if not idx_path.exists():
            continue

        try:
            audio_index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for it in audio_index.get("items", []):
            user_id = it.get("user_id")
            label = it.get("label")
            filename = it.get("filename")
            size_bytes = it.get("bytes")
            wav = it.get("wav") or {}
            duration_s = wav.get("duration_s")

            audio_files.append(
                {
                    "filename": f"{ch_dir_rel}/{filename}",
                    "bytes": size_bytes,
                    "user_id": user_id,
                    "speaker_label": label,
                    "duration_s": duration_s,
                }
            )

            if user_id not in participants_by_id:
                display_name = None
                if isinstance(user_id, int):
                    member = guild.get_member(user_id)
                    display_name = member.display_name if member else None

                participants_by_id[user_id] = {
                    "user_id": user_id,
                    "speaker_label": label,
                    "display_name": display_name,
                }

    participants = list(participants_by_id.values())

    participants.sort(key=lambda x: (x["user_id"] is None, x["user_id"] or 0))
    audio_files.sort(key=lambda x: x["filename"])

    return participants, audio_files


def setup(bot: discord.Bot) -> None:
    bot.add_cog(SessionCog(bot))
