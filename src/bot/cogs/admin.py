from __future__ import annotations

import time
import discord
from discord.ext import commands

from src.bot.config import get_settings


class AdminCog(commands.Cog):
    def __init__(self, bot: discord.Bot) -> None:
        self.bot = bot
        self.settings = get_settings()
        self.started_at = time.time()

    @discord.slash_command(name="ping", description="Check if the bot is responsive.")
    async def ping(self, ctx: discord.ApplicationContext) -> None:
        await ctx.respond("Pong!", ephemeral=True)

    @discord.slash_command(name="health", description="Show basic runtime info (for debugging).")
    async def health(self, ctx: discord.ApplicationContext) -> None:
        uptime_s = int(time.time() - self.started_at)
        openai_status = "Set" if self.settings.openai_api_key else "Not set"

        lines = [
            f"Uptime: {uptime_s}s",
            f"Database: {self.settings.database_url}",
            f"Retention: {self.settings.retention_hours}h",
            f"Audio chunk: {self.settings.audio_chunk_seconds}s",
            f"OpenAI key: {openai_status}",
        ]

        await ctx.respond("```\n" + "\n".join(lines) + "\n```", ephemeral=True)


def setup(bot: discord.Bot) -> None:
    bot.add_cog(AdminCog(bot))