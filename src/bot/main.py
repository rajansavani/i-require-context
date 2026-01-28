from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import discord
from discord.ext import commands

from src.bot.config import get_settings


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # help keep it readable by reducing some noisy loggers
    logging.getLogger("discord").setLevel(logging.INFO)
    logging.getLogger("discord.http").setLevel(logging.WARNING) 


def _build_intents() -> discord.Intents:
    # mainly just need voice status updates
    intents = discord.Intents.default()
    intents.guilds = True
    intents.voice_states = True
    return intents


class IRCBot(commands.Bot):
    def __init__(self) -> None:
        self.settings = get_settings()

        super().__init__(
            command_prefix="!", # not used for slash commands, but required by commands.Bot
            intents=_build_intents(),
        )

        self.log = logging.getLogger("irc.bot")

    async def setup_hook(self) -> None:
        """
        Runs before the bot connects to Discord.

        Load cogs and sync slash commands here.
        """
        await self._load_cogs()

        # can take awhile without specific guild_id for dev server
        self.log.info("Syncing application commands (global)...")
        try:
            await self.tree.sync()
            self.log.info("Slash commands synced.")
        except Exception:
            self.log.exception("Failed to sync slash commands.")

    async def on_ready(self) -> None:
        # fires when connected and ready
        user = self.user
        if user is None:
            self.log.info("Bot is ready.")
            return
        
        self.log.info("Logged in as %s (id=%s)", user, user.id)

    async def _load_cogs(self) -> None:
        """
        Loads all cogs under src/bot/cogs.

        This keeps adding commands simple (just drop in a new cog file).
        """
        cogs_path = Path(__file__).parent / "cogs"
        if not cogs_path.exists():
            self.log.warning("No cogs folder found at %s", cogs_path)
            return
        
        loaded = 0
        for file in sorted(cogs_path.glob("*.py")):
            if file.name.startswith("_"):
                continue

            module = f"bot.cogs.{file.stem}"
            try:
                await self.load_extension(module)
                loaded += 1
                self.log.info("Loaded cog: %s", module)
            except Exception:
                self.log.exception("Failed to load cog: %s", module)

        if loaded == 0:
            self.log.warning("No cogs were loaded. Add a cog under src/bot/cogs/ to register commands.")


async def _main() -> None:
    _setup_logging()

    settings = get_settings()
    log = logging.getLogger("irc.main")

    log.info("Starting I Require Context...")
    log.info("Database: %s", settings.database_url)
    log.info("Retention: %d hours", settings.retention_hours)

    bot = IRCBot()

    # start the bot
    await bot.start(settings.discord_bot_token)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Shutting down.")