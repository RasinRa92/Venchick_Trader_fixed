"""
config.py
=========
Centralized environment-variable loader for Venchick Trader.

Reads MT5 connection settings from environment variables and exposes them
via get_mt5_config(), which returns a dict in the exact shape that
mt5_engine.MT5_CONFIG and mt5_engine.init_mt5() already expect. This
keeps credentials out of source code and makes the configuration state
explicit and auditable at startup.

Why load_dotenv() is called here at module level
-------------------------------------------------
main.py imports mt5_engine (line 41) before its own load_dotenv() call
(line 69). mt5_engine imports this module, so env vars from the .env file
would not be available when get_mt5_config() runs unless we load them
here first. This mirrors the pattern already used in news_engine.py.
Subsequent load_dotenv() calls anywhere are no-ops (python-dotenv is
idempotent), so there is no conflict with main.py or news_engine.py.

Terminal-authenticated mode (no credentials in .env)
-----------------------------------------------------
If MT5_LOGIN / MT5_PASSWORD / MT5_SERVER are all absent, get_mt5_config()
returns login=None, password=None, server=None. mt5_engine._do_mt5_init()
already skips mt5.login() in that case and relies on the terminal being
pre-authenticated — behaviour identical to the previous hardcoded-None
MT5_CONFIG. A clear INFO message is emitted so the operator always knows
which mode is active.

Partial credentials (only 1 or 2 of the trio are set) are almost certainly
a misconfiguration; get_mt5_config() logs a WARNING and falls back to
terminal-authenticated mode rather than attempting a doomed login.

Environment variables (all optional)
-------------------------------------
MT5_LOGIN      integer  Broker account number        (required for cred login)
MT5_PASSWORD   string   Account password             (required for cred login)
MT5_SERVER     string   Broker server name           (required for cred login)
               e.g. "RoboForex-Pro"
MT5_PATH       string   Path to terminal64.exe       (optional, auto-detected)
MT5_TIMEOUT    integer  Connection timeout in ms     (optional, default: 60000)

Security notes
--------------
- The password is NEVER written to logs or repr'd in any log message.
- Credentials are loaded from .env or the OS environment; they are never
  present in any Python source file.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("forex_bot.config")

# ---------------------------------------------------------------------------
# Load .env before reading any env vars. Uses a try/except import identical
# to the pattern in main.py so missing python-dotenv degrades gracefully
# (env vars can be set at the OS level without the package installed).
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _ = load_dotenv()  # return value (bool) intentionally discarded
except ImportError:
    pass  # python-dotenv not installed; rely on OS-level environment variables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mt5_config() -> dict[str, int | str | None]:
    """
    Read MT5 connection settings from environment variables and return a
    config dict in the exact shape mt5_engine expects:

        {
            "login":    int | None,   # MT5 account number
            "password": str | None,   # account password  (never logged)
            "server":   str | None,   # broker server name
            "path":     str | None,   # optional path to terminal64.exe
            "timeout":  int,          # connection timeout in ms
        }

    When all three credentials (login / password / server) are absent the
    returned dict has login=None, password=None, server=None — identical to
    the previous hardcoded MT5_CONFIG defaults — and mt5_engine will use
    terminal-authenticated mode.

    Never raises. Invalid values are replaced with safe defaults and a
    warning is logged.
    """
    # ------------------------------------------------------------------
    # 1. Read the credential trio
    # ------------------------------------------------------------------
    raw_login = os.getenv("MT5_LOGIN", "").strip()
    password  = os.getenv("MT5_PASSWORD", "").strip() or None
    server    = os.getenv("MT5_SERVER",   "").strip() or None

    login: int | None = None
    if raw_login:
        try:
            login = int(raw_login)
        except ValueError:
            logger.error(
                "config: MT5_LOGIN value is not a valid integer — "
                "credential login will be skipped and terminal-authenticated "
                "mode will be used instead. Check your .env file.",
            )
            # login stays None; the credential fallback below will clear the
            # other two as well (partial credential guard).

    # ------------------------------------------------------------------
    # 2. Read optional / supplementary settings
    # ------------------------------------------------------------------
    path_val    = os.getenv("MT5_PATH", "").strip() or None

    timeout_raw = os.getenv("MT5_TIMEOUT", "").strip()
    timeout: int = 60_000
    if timeout_raw:
        try:
            timeout = int(timeout_raw)
        except ValueError:
            logger.warning(
                "config: MT5_TIMEOUT value %r is not a valid integer — "
                "using default 60000 ms.",
                timeout_raw,
            )

    # ------------------------------------------------------------------
    # 3. Partial-credential guard
    #    All three must be present together. One or two is a misconfiguration
    #    that would cause mt5.login() to fail with a confusing error; better
    #    to catch it here and fall back to terminal-authenticated mode.
    # ------------------------------------------------------------------
    cred_present = [login is not None, password is not None, server is not None]
    if any(cred_present) and not all(cred_present):
        missing_names = [
            name
            for name, present in zip(
                ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"), cred_present
            )
            if not present
        ]
        logger.warning(
            "config: partial MT5 credentials detected — %s is/are missing. "
            "All three (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER) must be set "
            "together for credential login. Falling back to "
            "terminal-authenticated mode.",
            ", ".join(missing_names),
        )
        login = password = server = None

    # ------------------------------------------------------------------
    # 4. Log the active mode — password is NEVER included in any log line
    # ------------------------------------------------------------------
    if login is not None:
        logger.info(
            "config: MT5 mode = CREDENTIAL LOGIN  "
            "account=%d  server=%s  path=%s  timeout=%d ms",
            login,
            server,
            path_val if path_val else "(auto-detect)",
            timeout,
        )
    else:
        logger.info(
            "config: MT5 mode = TERMINAL-AUTHENTICATED  "
            "(no credentials set — relying on a pre-authenticated MT5 terminal). "
            "To enable credential login set MT5_LOGIN, MT5_PASSWORD, and "
            "MT5_SERVER in your .env file.",
        )

    return {
        "login":    login,
        "password": password,
        "server":   server,
        "path":     path_val,
        "timeout":  timeout,
    }


def get_gold_symbol_candidate() -> str:
    """
    Return the configured broker Gold symbol candidate from the
    MT5_GOLD_SYMBOL environment variable.

    This is the CANDIDATE name only — the actual broker symbol is resolved
    and verified by mt5_engine.resolve_gold_symbol() after MT5 connects.
    If the env var is absent or empty, 'XAUUSD' is used as the first
    candidate to try (the correct name on most brokers including RoboForex).

    Common values by broker:
        RoboForex Pro / Standard : XAUUSD
        ICMarkets / Exness       : XAUUSDm
        FP Markets               : XAUUSD
        XM Micro / Standard      : XAUUSD

    Set MT5_GOLD_SYMBOL in .env only when your broker uses a non-standard
    name AND auto-detection fails (check the startup log for the resolved
    symbol name).
    """
    candidate = os.getenv("MT5_GOLD_SYMBOL", "XAUUSD").strip()
    if not candidate:
        logger.warning(
            "config: MT5_GOLD_SYMBOL is set but empty — "
            "defaulting to 'XAUUSD' as the first resolution candidate.",
        )
        return "XAUUSD"
    if candidate != "XAUUSD":
        logger.info("config: MT5_GOLD_SYMBOL candidate = %s", candidate)
    return candidate
