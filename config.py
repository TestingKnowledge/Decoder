"""
config.py — configuration and hard limits for the Discord deobfuscation bot.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Discord / runtime ────────────────────────────────────────────────────────
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")

COMMAND_PREFIX = "."
COMMAND = "deobfuscate"           # full command: ".deobfuscate"

MAX_FILE_BYTES = 25 * 1024 * 1024            # 25 MB
ALLOWED_EXTENSIONS = (".txt", ".lua", ".luau")

MAX_PROCESS_SECONDS = 60                      # per-file deobfuscation timeout
RATE_LIMIT_SECONDS = 30                       # 1 file per user per 30s
RATE_LIMIT_NOTICE = (
    f"⏳ Rate limit: one file every {RATE_LIMIT_SECONDS} seconds. Try again later."
)

# How long to keep temp files before forced cleanup (belt & braces).
TMP_DIR = os.getenv("TEMP_DIR", "temp")
TMP_TTL_SECONDS = 120

# ── Engine limits (protects the worker thread) ───────────────────────────────
ENGINE = {
    # Lexing / parsing
    "max_source_bytes": MAX_FILE_BYTES,
    "max_tokens": 4_000_000,                 # hard cap on token count
    "max_ast_nodes": 3_000_000,              # hard cap on AST node count
    "parse_timeout_seconds": 30,

    # Constant folding / emulation
    "max_fold_ops": 2_000_000,               # total folding operations
    "max_string_build": 8 * 1024 * 1024,     # 8 MB max string built during folding
    "max_loops_emulated": 50_000,            # while/for iterations emulated
    "max_total_emulated_steps": 2_000_000,   # total interpreter steps
    "emulation_timeout_seconds": 25,

    # String pool handling
    "max_pool_entries": 200_000,             # entries per pool
    "max_pools": 256,

    # Output
    "max_report_chars": 1_800,               # inline report limit before file
    "max_output_chars": 1_200_000,           # cap on clean-script length
    "analysis_head_rows": 12,                # rows shown per table in report
}

# Discord message limits
DISCORD_MSG_LIMIT = 1900                     # safe limit under 2000
