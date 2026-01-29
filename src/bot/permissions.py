from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Set

import discord


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: Optional[str] = None


def _parse_int_set(csv: str) -> Set[int]:
    # parses "1,2,3" into {1, 2, 3}
    out: Set[int] = set()
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            # ignore invalid entries instead of crashing
            continue
    return out

def allowed_guild_ids() -> Set[int]:
    """
    Allowlist for guilds (servers) that can use the bot (or expensive features).

    Set ALLOWED_GUILD_IDS="123,456" in .env.
    If empty/unset, we treat it as "allow all" (for local dev and testing).
    """
    raw = os.getenv("ALLOWED_GUILD_IDS", "").strip()
    if not raw:
        return set()
    return _parse_int_set(raw)

def is_guild_allowed(guild_id: int) -> bool:
    ids = allowed_guild_ids()
    if not ids:
        # no allowlist conigured = allow all
        return True
    return guild_id in ids

def gate_guild(ctx: discord.ApplicationContext) -> GateResult:
    # guards commands that should only run in approved servers
    if ctx.guild is None:
        return GateResult(False, "This command can only be used in a server.")
    if not is_guild_allowed(ctx.guild.id):
        return GateResult(False, "This server is not authorized to use this bot.")
    return GateResult(True)

def gate_admin(ctx: discord.ApplicationContext) -> GateResult:
    # require server admin permissions
    if ctx.guild is None:
        return GateResult(False, "This command can only be used in a server.")
    
    member = ctx.author
    if not isinstance(member, discord.Member):
        return GateResult(False, "Unable to verify permissions for this user.")
    
    if member.guild_permissions.administrator:
        return GateResult(True)
    
    return GateResult(False, "You need to be a server admin to use this command.")

def gate_openai(ctx: discord.ApplicationContext) -> GateResult:
    """
    Guard anything that would spend money (STT/LLM calls).

    For now, we just require:
    - being in guild allowlist
    - OPENAI_API_KEY is set
    """
    guild_gate = gate_guild(ctx)
    if not guild_gate.allowed:
        return guild_gate
    
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return GateResult(False, "OPENAI_API_KEY is not configured on the bot host.")

    return GateResult(True)


async def deny(ctx: discord.ApplicationContext, reason: str, *, ephemeral: bool = True) -> None:
    # standard way to reply when a gate fails
    await ctx.respond(reason, ephemeral=ephemeral)