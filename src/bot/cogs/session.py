from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import discord
from discord.ext import commands

from src.bot.config import get_settings
from src.bot.permissions import gate_guild, deny


@dataclass
class ActiveSession:
    session_id: str
    guild_id: int
    voice_channel_id: int
    started_by_user_id: int
    started_at: float
    voice_client: discord.VoiceClient
    sink: discord.sinks.Sink


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

        # start recording into a sink
        # WaveSink keeps per-user audio buffers we can write in the callback
        sink = discord.sinks.WaveSink()

        async def _on_recording_finished(sink_: discord.sinks.Sink, *args: Any) -> None:
            # this runs after stop_recording() finishes collecting audio
            try:
                self._save_sink_audio(sink_, base_dir)
                self.log.info("Saved session audio to %s", base_dir.as_posix())
            except Exception:
                self.log.exception("Failed while saving recorded audio.")

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

        # stop recording triggers the callback that saves audio
        try:
            session.voice_client.stop_recording()
        except Exception:
            self.log.exception("Failed to stop recording.")
            await ctx.respond("Failed to stop recording.", ephemeral=True)
            return

        # disconnect after stopping
        try:
            await session.voice_client.disconnect()
        except Exception:
            self.log.exception("Failed to disconnect after recording.")

        duration_s = int(time.time() - session.started_at)
        self.active_by_guild.pop(guild.id, None)

        await ctx.respond(f"Recording stopped. Duration: **{duration_s}s**.", ephemeral=False)

    def _save_sink_audio(self, sink: discord.sinks.Sink, base_dir: Path) -> None:
        # writes per-speaker wav buffers from the sink
        audio_data = getattr(sink, "audio_data", None)
        if not audio_data:
            (base_dir / "EMPTY_SESSION.txt").write_text(
                "No audio data was captured. This can happen if nobody spoke, or if voice receive isn't working.\n",
                encoding="utf-8",
            )
            return

        # print raw key info so we can see what pycord is actually giving us
        keys = list(audio_data.keys())
        pretty = []
        for k in keys:
            pretty.append(
                f"{type(k).__name__}:id={getattr(k, 'id', None)} name={getattr(k, 'name', None)} repr={k!r}"
            )
        self.log.info("Recorded sink keys: %s", " | ".join(pretty))

        # always write unique files so nothing can overwrite
        for idx, (speaker_key, data) in enumerate(audio_data.items(), start=1):
            # try to derive a discord id if available
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

            # if we have an id, use it; otherwise fall back to an index
            if isinstance(user_id, int):
                filename = f"{user_id}{safe_name}.wav"
                label = str(user_id)
            else:
                filename = f"unknown_{idx}.wav"
                label = f"unknown_{idx}"

            out_path = base_dir / filename

            file_obj = getattr(data, "file", None)
            if file_obj is None:
                self.log.warning("No file object for %s; skipping.", label)
                continue

            file_obj.seek(0)
            blob = file_obj.read()

            if not blob:
                self.log.warning("Empty audio buffer for %s; skipping.", label)
                continue

            out_path.write_bytes(blob)
            self.log.info("Wrote %s (%d bytes) -> %s", label, len(blob), out_path.name)


def setup(bot: discord.Bot) -> None:
    bot.add_cog(SessionCog(bot))
