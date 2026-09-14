"""
bot.py — Discord bot for static Lua/Luau deobfuscation.

Flow:
  1. User uploads a .txt/.lua/.luau file and sends `.deobfuscate`
     (same message, a reply, or the message just before).
  2. The file is downloaded to a temp dir, decoded, and run through the
     static analysis engine (never executed — pure static analysis).
  3. The bot replies with TWO SEPARATE outputs:
       a) the analysis report (message or file), and
       b) the clean deobfuscated script as a .lua file — pure code only.
  4. Temp files are always removed (privacy) in finally blocks.

Hard limits: 25 MB per file, one file per user per 30 s, 60 s processing
timeout, engine budgets well below that so it degrades gracefully.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from typing import Dict, Optional

import aiofiles
import discord
import audioop 
from discord.ext import commands
from dotenv import load_dotenv

from config import (
    ALLOWED_EXTENSIONS,
    COMMAND,
    COMMAND_PREFIX,
    DISCORD_MSG_LIMIT,
    DISCORD_TOKEN,
    ENGINE,
    MAX_FILE_BYTES,
    MAX_PROCESS_SECONDS,
    RATE_LIMIT_NOTICE,
    RATE_LIMIT_SECONDS,
    TMP_DIR,
    TMP_TTL_SECONDS,
)
from deobfuscator import DeobfuscationEngine

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("deobfuscate-bot")

# ── temp dir ─────────────────────────────────────────────────────────
os.makedirs(TMP_DIR, exist_ok=True)

# ── engine worker ────────────────────────────────────────────────────
# A FRESH engine instance is created per file (RULE 14: no cross-sample
# contamination — a previous file's pools/aliases must never leak into
# the current analysis).  Runs in a worker thread via asyncio.to_thread.


def run_engine(source: str) -> Dict[str, str]:
    engine = DeobfuscationEngine()
    return engine.deobfuscate(source)


# ── bot setup ────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True          # required to read `.deobfuscate`

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents,
                   help_command=None)

# per-user rate limiting: {user_id: monotonic last-processed time}
_user_last: Dict[int, float] = {}


# ── helpers ──────────────────────────────────────────────────────────
def _ext_ok(filename: str) -> bool:
    return filename.lower().endswith(ALLOWED_EXTENSIONS)


def _base_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0] or "script"
    return stem


async def _find_attachment(message: discord.Message) -> Optional[discord.Attachment]:
    """Attachment from this message, the replied-to message, or the
    previous message in the channel (most recent first)."""
    if message.attachments:
        return message.attachments[0]

    # reply-reference: the message the user replied to
    ref = message.reference
    if ref is not None and ref.message_id is not None:
        try:
            ref_msg = await message.channel.fetch_message(ref.message_id)
            if ref_msg.attachments:
                return ref_msg.attachments[0]
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass

    # previous message in channel history
    async for prev in message.channel.history(limit=1, before=message):
        if prev.attachments:
            return prev.attachments[0]
        break
    return None


async def _send_analysis(message: discord.Message, report: str) -> None:
    """Deliver the analysis report — message when short, file when long.
    (Report may contain [UNKNOWN] markers; the clean script never does.)"""
    if len(report) <= DISCORD_MSG_LIMIT:
        await message.reply(report[:DISCORD_MSG_LIMIT])
        return
    buf = io.BytesIO(report.encode("utf-8", "replace"))
    await message.reply(
        "📋 Analysis report (full version attached):",
        file=discord.File(buf, filename="analysis_report.txt"),
    )


async def _send_clean_script(message: discord.Message, script: str,
                             filename: str) -> None:
    """Deliver the deobfuscated script as a clean file — pure code only."""
    out_name = f"deobfuscated_{_base_name(filename)}.lua"
    data = script.encode("utf-8", "replace")
    if not data:
        data = b"\n"          # avoid a zero-byte file Discord would reject
    buf = io.BytesIO(data)
    await message.reply(
        "📄 Deobfuscated script (clean code only):",
        file=discord.File(buf, filename=out_name),
    )


def _temp_path(message_id: int, name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return os.path.join(TMP_DIR, f"{message_id}_{safe}")


async def _cleanup_old_temp() -> None:
    """Remove temp files older than TMP_TTL_SECONDS (best effort)."""
    now = time.time()
    try:
        for entry in os.listdir(TMP_DIR):
            path = os.path.join(TMP_DIR, entry)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < now - TMP_TTL_SECONDS:
                    os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


# ── events ───────────────────────────────────────────────────────────
@bot.event
async def on_ready() -> None:
    log.info("%s connected — %d guild(s)", bot.user, len(bot.guilds))
    log.info("command: %s%s", COMMAND_PREFIX, COMMAND)
    asyncio.create_task(_cleanup_loop())


async def _cleanup_loop() -> None:
    """Background sweep: purge expired temp files every 60 s."""
    while True:
        await asyncio.sleep(60)
        await _cleanup_old_temp()


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if message.author == bot.user:
        return

    # Ignore our own prefix for other commands (fallthrough to commands ext)
    if message.content.strip().lower() != f"{COMMAND_PREFIX}{COMMAND}":
        await bot.process_commands(message)
        return

    # ── rate limit: 1 file per user per 30 s ──
    now = time.monotonic()
    last = _user_last.get(message.author.id, 0.0)
    if now - last < RATE_LIMIT_SECONDS:
        wait = int(RATE_LIMIT_SECONDS - (now - last))
        await message.reply(
            f"⏳ You're rate-limited — try again in {wait}s "
            f"(1 file per user per {RATE_LIMIT_SECONDS}s)."
        )
        return

    # ── find the file ──
    attachment = await _find_attachment(message)
    if attachment is None:
        await message.reply(
            "❌ No file found. Upload a `.txt`, `.lua` or `.luau` file "
            "(in this message, as a reply, or the message just above) "
            "and then send `.deobfuscate`."
        )
        return

    # ── validation ──
    if not _ext_ok(attachment.filename):
        await message.reply(
            f"❌ Invalid file type `{attachment.filename}` — only "
            f"{', '.join(ALLOWED_EXTENSIONS)} files are accepted."
        )
        return
    if attachment.size > MAX_FILE_BYTES:
        await message.reply(
            f"❌ File too large ({attachment.size:,} bytes; max 25 MB)."
        )
        return

    _user_last[message.author.id] = time.monotonic()

    status = await message.reply(
        "⏳ Processing your file — this may take up to a minute…"
    )

    tmp_path = _temp_path(message.id, attachment.filename)
    try:
        # ── download to temp (aiofiles, async) ──
        data = await attachment.read()
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(data)

        # ── decode ──
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            source = data.decode("utf-8", "replace")
            await message.reply(
                "⚠️ File is not valid UTF-8 — processing with replacement "
                "characters; obfuscated binary may be incompletely recovered."
            )

        # ── run the engine in a worker thread with a hard timeout ──
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(run_engine, source),
                timeout=MAX_PROCESS_SECONDS,
            )
        except asyncio.TimeoutError:
            await message.reply(
                f"❌ Processing exceeded {MAX_PROCESS_SECONDS}s and was "
                "stopped. The obfuscation is too complex for the safe "
                "analysis budget — try a smaller sample."
            )
            return

        analysis = result["analysis"]
        clean = result["clean_script"]

        # ── two SEPARATE outputs ──
        await _send_analysis(message, analysis)
        await _send_clean_script(message, clean, attachment.filename)

        try:
            await status.delete()
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass

    except Exception as exc:  # graceful error handling
        log.exception("processing failed")
        try:
            await message.reply(f"❌ Error while processing: `{type(exc).__name__}`")
        except (discord.Forbidden, discord.HTTPException):
            pass
    finally:
        # ── privacy: temp file always removed ──
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ── health endpoint (Render health check) ────────────────────────────
async def _health_server(host: str = "0.0.0.0", port: int = 0):
    """Start the tiny /healthz server on the first free port."""
    loop = asyncio.get_running_loop()
    server = await loop.create_server(
        lambda: _HealthProtocol(), host, port)
    sock = server.sockets[0]
    port = sock.getsockname()[1]
    log.info("health endpoint on port %d", port)
    async with server:
        await server.serve_forever()


class _HealthProtocol(asyncio.Protocol):
    """Line protocol: answers any request with 200 + JSON status."""

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport

    def data_received(self, data: bytes):
        body = b'{"status":"ok"}\n'
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
        self.transport.write(head + body)
        self.transport.close()


# ── entry point ──────────────────────────────────────────────────────
async def main() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit(
            "DISCORD_TOKEN is not set — put it in .env (see .env.example) "
            "or the Render dashboard."
        )
    # health server alongside the bot (Render healthCheckPath: /healthz)
    health_task = asyncio.create_task(_health_server(
        port=int(os.getenv("PORT", "10000"))))
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        health_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
