"""
ai_engine.py
============
"Venchick AI" — a real-time forex analyst chat layer powered by Groq
(groq SDK, Llama 3.3 70B), wired directly into the bot's own data
pipeline so it never has to guess a price, indicator value, or news
event.

Design principle: the LLM is NOT given raw tool-calling access to MT5 —
instead, THIS module calls the bot's existing analysis functions itself
(mt5_engine.analyze_market_structure, mt5_engine.get_live_spread,
news_engine.is_news_freeze_active / check_emergency_killswitch /
check_economic_calendar), formats the results into a compact
"LIVE DATA CONTEXT" block, and injects that block into the prompt before
the model ever sees the user's question. The model is instructed to
answer using ONLY those numbers. This keeps every technical/news claim
mathematically aligned with the rest of the bot instead of an LLM
hallucinating a price.

PHASE 6 — Institutional Context Injection
------------------------------------------
`_gather_market_context()` now additionally queries the four Phase 4/5
institutional engines — structure_engine, smc_engine, session_engine,
and bias_engine — plus news_engine's institutional calendar summary,
concurrently via asyncio.gather, and folds their output into the same
LIVE DATA CONTEXT block alongside the legacy MT5 signal-engine numbers.
Every one of those calls is individually try/excepted with a safe
"data unavailable" fallback line, so a single engine being down (e.g.
MT5 briefly disconnected) degrades that one section of the context
instead of breaking the chat response entirely. The public API,
message-history loop, and chart-tag mechanism in generate_ai_response()
are unchanged.

PHASE 9 — Self-Learning AI Context Memory Injection
----------------------------------------------------
Adds a "MEMORY & HISTORICAL CONTEXT" section (built by
`_gather_memory_context()`) alongside the LIVE DATA CONTEXT block:
recent conversation history with the requesting user, and recent
alert/scenario outcomes for the mentioned symbol(s), both read from
`memory_engine` (which persists to SQLite via database.py — see
Phase 8). After a successful reply, `generate_ai_response()` fires a
background task that logs the prompt/response pair to disk via
`memory_engine.log_interaction()`, so it's available for recall on the
very next message and survives a bot restart.

Like every other optional engine here, memory_engine is imported
defensively and every call goes through a `_safe_*` wrapper: if the DB
layer is down, the AI simply answers without that turn's recall/logging
instead of failing to respond. The in-process `_CHANNEL_HISTORY` rolling
buffer (used for the model's own multi-turn context within a live
conversation) is unchanged and still authoritative for *this session*;
persistent memory is the separate, restart-surviving layer on top.

PHASE 10 — /makeaccount Modal Profile Personalization
-------------------------------------------------------
Adds a "USER TRADING PROFILE" section (`_gather_user_profile_context()`)
built from `database.get_user_preferences(user_id)` — the broker/server,
preferred pairs, and country a trader entered into the `/makeaccount`
Modal in main.py. Country is mapped to an approximate fixed-offset local
time zone (see `_COUNTRY_TIMEZONES`) for local session-timing callouts;
broker/server name is mapped to a disclosed *heuristic* spread-profile
note (Raw/ECN-style vs. standard), never a fabricated real spread value —
the system prompt explicitly tells the model to defer to the live spread
figure already in LIVE DATA CONTEXT whenever the two would conflict.
Same defensive-import + `_safe_*` pattern as everything else: no
database module, no saved profile, or an unmapped country all degrade to
a plain "not available" line instead of blocking a response or guessing.

Dependencies:
    pip install groq

Environment variables:
    GROQ_API_KEY   required

Public API:
    configure(extra_pairs: list[str] | None = None) -> None
    async generate_ai_response(user_message: str, channel_context: dict) -> str
    extract_chart_tag(text: str) -> tuple[str, str | None]
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from groq import AsyncGroq

import mt5_engine
import news_engine

# ---------------------------------------------------------------------------
# Institutional engines (Phase 5) — imported defensively so a missing or
# broken module degrades _gather_market_context() to a "data unavailable"
# note for that section instead of preventing ai_engine.py from importing
# or responding at all.
# ---------------------------------------------------------------------------

try:
    import structure_engine
except Exception:  # pragma: no cover - defensive import
    structure_engine = None  # type: ignore[assignment]

try:
    import smc_engine
except Exception:  # pragma: no cover - defensive import
    smc_engine = None  # type: ignore[assignment]

try:
    import session_engine
except Exception:  # pragma: no cover - defensive import
    session_engine = None  # type: ignore[assignment]

try:
    import bias_engine
except Exception:  # pragma: no cover - defensive import
    bias_engine = None  # type: ignore[assignment]

# Phase 9 — memory_engine (persistent conversation + alert-outcome recall).
# Imported defensively like the other engines: if memory_engine or its DB
# layer is unavailable for any reason, the AI simply loses recall/logging
# for that turn instead of failing to respond at all.
try:
    import memory_engine
except Exception:  # pragma: no cover - defensive import
    memory_engine = None  # type: ignore[assignment]

# Phase 10 — database (get_user_preferences), for /makeaccount modal
# profile personalization. Imported defensively for the same reason as
# every other engine here: a missing/broken DB layer should degrade this
# one section of the prompt, not prevent the AI from responding.
try:
    import database
except Exception:  # pragma: no cover - defensive import
    database = None  # type: ignore[assignment]

logger = logging.getLogger("forex_bot.ai_engine")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"

# How many user/model turns to keep per channel for conversational memory.
MAX_HISTORY_TURNS = 6

# Venchick Trader is a Gold-only system. The AI recognises only Gold symbol
# names. main.py adds the resolved broker symbol (e.g. 'XAUUSDm') via
# configure() at startup so any broker-specific name is also detected.
_DEFAULT_PAIRS = ["XAUUSD"]
_extra_pairs: list[str] = []

_CHART_KEYWORDS = (
    "chart", "graph", "picture", "image", "plot", "visual",
    "show me", "screenshot", "candles", "candlestick",
)
_NEWS_KEYWORDS = (
    "news", "calendar", "nfp", "cpi", "fomc", "rate decision",
    "event", "release", "economic data",
)

_CHART_TAG_RE = re.compile(r"\[GENERATE_CHART:([A-Z]{3,10})\]")

# channel_id -> deque[("user"|"model", text)], simple rolling chat memory.
_CHANNEL_HISTORY: dict[int, deque] = {}

_client: Optional[AsyncGroq] = None
_client_init_attempted = False

# Number of items kept per SMC list (FVGs / order blocks / sweeps / pools)
# when formatting the context block — keeps prompt size bounded even when
# an instrument has a lot of active structure.
_SMC_ITEM_LIMIT = 3

# Phase 9 — how many rows to pull from persistent memory per prompt.
_CONVERSATION_MEMORY_LIMIT = 5
_ALERT_HISTORY_LIMIT = 5
# Longest a single stored prompt/response is allowed to render as inside a
# context block — keeps one old wall-of-text reply from blowing up token
# usage on every subsequent message.
_MEMORY_TEXT_TRUNCATE = 240

# --------------------------------------------------------------------------- #
# Phase 10 — /makeaccount profile personalization
# --------------------------------------------------------------------------- #

# Country (free text, as typed into the Modal, lowercased) -> (tz label,
# fixed UTC offset in hours). This is a STANDARD-TIME reference table, not
# a full IANA tzdata implementation — it intentionally does not attempt
# DST transitions (which vary by country and by year), so the local time
# it renders can be off by an hour during a country's DST window. That's
# an accepted, disclosed simplification: the system prompt instructs the
# model to treat this as an approximate reference, not an exact clock.
# Extend this table as your user base grows; unmapped countries simply
# render without a local-time line instead of guessing.
_COUNTRY_TIMEZONES: Dict[str, tuple[str, float]] = {
    "malaysia": ("MYT", 8.0),
    "singapore": ("SGT", 8.0),
    "indonesia": ("WIB", 7.0),
    "philippines": ("PHT", 8.0),
    "thailand": ("ICT", 7.0),
    "vietnam": ("ICT", 7.0),
    "hong kong": ("HKT", 8.0),
    "china": ("CST", 8.0),
    "japan": ("JST", 9.0),
    "south korea": ("KST", 9.0),
    "india": ("IST", 5.5),
    "pakistan": ("PKT", 5.0),
    "bangladesh": ("BST", 6.0),
    "uae": ("GST", 4.0),
    "united arab emirates": ("GST", 4.0),
    "saudi arabia": ("AST", 3.0),
    "qatar": ("AST", 3.0),
    "nigeria": ("WAT", 1.0),
    "south africa": ("SAST", 2.0),
    "kenya": ("EAT", 3.0),
    "egypt": ("EET", 2.0),
    "uk": ("GMT", 0.0),
    "united kingdom": ("GMT", 0.0),
    "ireland": ("GMT", 0.0),
    "germany": ("CET", 1.0),
    "france": ("CET", 1.0),
    "spain": ("CET", 1.0),
    "italy": ("CET", 1.0),
    "netherlands": ("CET", 1.0),
    "switzerland": ("CET", 1.0),
    "poland": ("CET", 1.0),
    "turkey": ("TRT", 3.0),
    "russia": ("MSK", 3.0),
    "usa": ("ET", -5.0),
    "united states": ("ET", -5.0),
    "canada": ("ET", -5.0),
    "mexico": ("CST", -6.0),
    "brazil": ("BRT", -3.0),
    "argentina": ("ART", -3.0),
    "australia": ("AEST", 10.0),
    "new zealand": ("NZST", 12.0),
}


def configure(extra_pairs: Optional[list[str]] = None) -> None:
    """Call once at bot startup (e.g. from main.py's on_ready) to make the
    AI aware of every pair the rest of the bot trades."""
    global _extra_pairs
    if extra_pairs:
        _extra_pairs = [p.upper() for p in extra_pairs]


def _get_client() -> Optional[AsyncGroq]:
    global _client, _client_init_attempted
    if _client is not None:
        return _client
    if _client_init_attempted:
        return None
    _client_init_attempted = True

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set — Venchick AI will not respond.")
        return None

    try:
        _client = AsyncGroq(api_key=GROQ_API_KEY)
        return _client
    except Exception:
        logger.exception("Failed to initialize Groq client.")
        return None


# --------------------------------------------------------------------------- #
# Intent detection (deterministic — never left to the LLM to decide alone)
# --------------------------------------------------------------------------- #

def _known_pairs() -> list[str]:
    return list(dict.fromkeys(_DEFAULT_PAIRS + _extra_pairs))  # de-duped, order-preserved


def _detect_symbols(text: str) -> list[str]:
    """Detect Gold/XAUUSD mentions in the user's message.

    Venchick Trader is a Gold-only system. Returns a list containing the
    resolved broker Gold symbol when Gold is mentioned (by configured symbol
    name, 'GOLD', or 'XAU'), or an empty list when no Gold-related content
    is detected.

    Never returns non-Gold instrument symbols. If a user asks about EURUSD
    or any other instrument, this returns [] and the AI answers without
    market data context — the system prompt instructs it to decline non-Gold
    analysis rather than inventing data.
    """
    upper = text.upper()
    known = _known_pairs()  # ["XAUUSD"] + broker alias after configure()

    # Resolve the broker symbol for MT5-accurate context fetching.
    # Falls back to known[0] ("XAUUSD") if MT5 not yet connected.
    try:
        gold = mt5_engine.get_gold_symbol()
    except RuntimeError:
        gold = known[0] if known else "XAUUSD"

    # Check for any known Gold symbol name in the message.
    if any(p in upper for p in known):
        return [gold]

    # Also recognise plain-language Gold references.
    if any(kw in upper for kw in ("GOLD", "XAU")):
        return [gold]

    # No Gold-related content detected — return empty so the AI answers
    # without pulling live market context (correct for off-topic questions).
    return []


def _wants_chart(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _CHART_KEYWORDS)


def _mentions_news(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _NEWS_KEYWORDS)


# --------------------------------------------------------------------------- #
# Institutional engine calls — each individually safe, never raises
# --------------------------------------------------------------------------- #

async def _safe_legacy_analysis(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        return await mt5_engine.analyze_market_structure(symbol)
    except Exception:
        logger.exception("ai_engine: analyze_market_structure failed for %s", symbol)
        return None


async def _safe_legacy_spread(symbol: str) -> Optional[float]:
    try:
        return await mt5_engine.get_live_spread(symbol)
    except Exception:
        logger.exception("ai_engine: get_live_spread failed for %s", symbol)
        return None


async def _safe_structure(symbol: str) -> Optional[Dict[str, Any]]:
    if structure_engine is None:
        return None
    try:
        return await structure_engine.analyze_xauusd_structure(symbol=symbol)
    except Exception:
        logger.exception("ai_engine: structure_engine.analyze_xauusd_structure failed for %s", symbol)
        return None


async def _safe_smc(symbol: str) -> Optional[Dict[str, Any]]:
    if smc_engine is None:
        return None
    try:
        return await smc_engine.analyze_xauusd_smc(symbol=symbol)
    except Exception:
        logger.exception("ai_engine: smc_engine.analyze_xauusd_smc failed for %s", symbol)
        return None


async def _safe_session(symbol: str) -> Optional[Dict[str, Any]]:
    if session_engine is None:
        return None
    try:
        return await session_engine.get_xauusd_session_context(symbol=symbol)
    except Exception:
        logger.exception("ai_engine: session_engine.get_xauusd_session_context failed for %s", symbol)
        return None


async def _safe_institutional_news() -> Optional[Dict[str, Any]]:
    try:
        return await news_engine.get_institutional_news_context()
    except Exception:
        logger.exception("ai_engine: news_engine.get_institutional_news_context failed")
        return None


async def _safe_bias(symbol: str) -> Optional[Dict[str, Any]]:
    if bias_engine is None:
        return None
    try:
        return await bias_engine.evaluate_xauusd_bias_and_scenarios(symbol=symbol)
    except Exception:
        logger.exception("ai_engine: bias_engine.evaluate_xauusd_bias_and_scenarios failed for %s", symbol)
        return None


async def _safe_conversation_memory(user_id: Optional[str]) -> list[Dict[str, Any]]:
    """Recent logged interactions for this specific user (Phase 9 recall)."""
    if memory_engine is None or not user_id:
        return []
    try:
        return await memory_engine.get_recent_conversation_memory(
            str(user_id), limit=_CONVERSATION_MEMORY_LIMIT
        )
    except Exception:
        logger.exception("ai_engine: get_recent_conversation_memory failed for user_id=%s", user_id)
        return []


async def _safe_alert_history(symbol: str) -> list[Dict[str, Any]]:
    """Recent logged alerts/scenarios for this symbol (Phase 9 recall)."""
    if memory_engine is None:
        return []
    try:
        return await memory_engine.get_recent_alert_history(symbol, limit=_ALERT_HISTORY_LIMIT)
    except Exception:
        logger.exception("ai_engine: get_recent_alert_history failed for symbol=%s", symbol)
        return []


async def _safe_user_preferences(user_id: Optional[str]) -> Optional[Any]:
    """The user's saved /makeaccount profile (database.UserPreferences), if any."""
    if database is None or not user_id:
        return None
    try:
        return await database.get_user_preferences(int(user_id))
    except (ValueError, TypeError):
        # user_id wasn't int-coercible (e.g. a test/channel placeholder) —
        # not a database problem, just no profile to look up.
        return None
    except Exception:
        logger.exception("ai_engine: get_user_preferences failed for user_id=%s", user_id)
        return None


# --------------------------------------------------------------------------- #
# Formatting helpers — turn each engine's dict into a short, readable block.
# Every helper accepts an Optional dict and never raises: missing/partial
# data always renders as a plain "not available" line instead of a KeyError.
# --------------------------------------------------------------------------- #

def _format_legacy_signal(symbol: str, analysis: Optional[Dict[str, Any]], spread: Optional[float]) -> str:
    if analysis is None:
        return (
            "Legacy MT5 signal engine: no live data available right now "
            "(MT5 disconnected or insufficient candle history)."
        )

    lines = [
        f"Model bias: {analysis.get('action')}  (confidence {analysis.get('confidence', 0):.0f}%, "
        f"strategy: {analysis.get('recommended_strategy')})",
        f"RSI(H1): {analysis.get('rsi_h1')}   ATR(H1): {analysis.get('atr_h1')}",
    ]
    if spread is not None:
        lines.append(f"Live spread: {spread} pips")
    if analysis.get("swing_high_h4") is not None:
        lines.append(f"H4 swing high/low: {analysis['swing_high_h4']} / {analysis['swing_low_h4']}")
    if analysis.get("action") in ("BUY", "SELL"):
        lines.append(
            f"Suggested levels — Entry: {analysis.get('entry')}  SL: {analysis.get('sl')}  "
            f"TP1: {analysis.get('tp1')}  TP2: {analysis.get('tp2')}  TP3: {analysis.get('tp3')}"
        )
    if analysis.get("notes"):
        lines.append("Notes: " + "; ".join(analysis["notes"]))

    return "\n".join(lines)


def _format_session_and_levels(session_ctx: Optional[Dict[str, Any]]) -> str:
    if not session_ctx:
        return "Session/levels engine: not available right now."

    lines = [
        f"Active session(s): {', '.join(session_ctx.get('active_sessions', []) or ['UNKNOWN'])}  "
        f"(primary: {session_ctx.get('primary_session', 'UNKNOWN')})",
    ]
    time_remaining = session_ctx.get("time_remaining_in_session")
    if time_remaining:
        lines.append(f"Time remaining in primary session: {time_remaining}")

    levels = session_ctx.get("levels") or {}
    if levels:
        level_bits = [f"{k}: {v}" for k, v in levels.items() if v is not None]
        if level_bits:
            lines.append("Institutional reference levels — " + " | ".join(level_bits))

    swept = session_ctx.get("asian_range_swept") or {}
    if swept.get("ash_swept") or swept.get("asl_swept"):
        lines.append(
            f"Asian range sweep — ASH swept: {swept.get('ash_swept', False)}, "
            f"ASL swept: {swept.get('asl_swept', False)}"
        )

    if session_ctx.get("error"):
        lines.append(f"(session engine partial data — {session_ctx['error']})")

    return "\n".join(lines)


def _format_structure(structure_result: Optional[Dict[str, Any]]) -> str:
    if not structure_result:
        return "Structure engine: not available right now."

    by_tf = structure_result.get("timeframes", {}) or {}
    if not by_tf:
        return "Structure engine: no timeframe data returned."

    tf_order = ["D1", "H4", "H1", "M15"]
    lines = []
    for tf in tf_order:
        tf_data = by_tf.get(tf)
        if not tf_data:
            continue
        bos = tf_data.get("last_bos")
        choch = tf_data.get("last_choch")
        bos_str = f"{bos['direction']} @ {bos['price']}" if bos else "none recent"
        choch_str = f"{choch['direction']} @ {choch['price']}" if choch else "none recent"
        lines.append(
            f"{tf}: {tf_data.get('trend_bias', 'NEUTRAL')} | last BOS: {bos_str} | "
            f"last CHOCH: {choch_str} | Zone: {tf_data.get('zone', 'UNKNOWN')}"
        )

    return "\n".join(lines) if lines else "Structure engine: no timeframe data returned."


def _format_smc(smc_result: Optional[Dict[str, Any]]) -> str:
    if not smc_result:
        return "SMC/liquidity engine: not available right now."

    lines = []

    fvgs = (smc_result.get("unfilled_fvgs") or [])[:_SMC_ITEM_LIMIT]
    if fvgs:
        lines.append("Unfilled FVGs: " + " | ".join(
            f"{f['direction']} [{f['gap_bottom']}-{f['gap_top']}]" for f in fvgs
        ))

    obs = (smc_result.get("order_blocks") or [])[:_SMC_ITEM_LIMIT]
    if obs:
        lines.append("Active order blocks: " + " | ".join(
            f"{o['type']} [{o['ob_low']}-{o['ob_high']}] (src: {o.get('source')})" for o in obs
        ))

    bsl = (smc_result.get("active_bsl") or [])[:_SMC_ITEM_LIMIT]
    ssl = (smc_result.get("active_ssl") or [])[:_SMC_ITEM_LIMIT]
    if bsl:
        lines.append("Active BSL (buy-side liquidity): " + ", ".join(str(b["price"]) for b in bsl))
    if ssl:
        lines.append("Active SSL (sell-side liquidity): " + ", ".join(str(s["price"]) for s in ssl))

    sweeps = (smc_result.get("recent_sweeps") or [])[:_SMC_ITEM_LIMIT]
    if sweeps:
        lines.append("Recent liquidity sweeps: " + " | ".join(
            f"{s['swept_level_type']} @ {s['level_price']} (swept to {s['sweep_price']})" for s in sweeps
        ))

    if not lines:
        return "SMC/liquidity engine: no active FVGs, order blocks, or sweeps detected right now."

    return "\n".join(lines)


def _format_news_risk(news_ctx: Optional[Dict[str, Any]]) -> str:
    if not news_ctx:
        return "News risk engine: not available right now."

    lines = []
    if news_ctx.get("active_killswitch"):
        lines.append("🛑 EMERGENCY KILLSWITCH ACTIVE")
    if news_ctx.get("active_news_freeze"):
        lines.append("⏸ News freeze window ACTIVE right now")

    next_event = news_ctx.get("next_major_event")
    if next_event:
        lines.append(
            f"Next major USD event: {next_event.get('event')} at {next_event.get('time')} "
            f"(in ~{next_event.get('minutes_remaining')} min, impact: {next_event.get('impact')})"
        )

    upcoming = (news_ctx.get("upcoming_usd_events_today") or [])[:3]
    if upcoming:
        lines.append("Upcoming USD events today: " + " | ".join(
            f"{e.get('event')} ({e.get('impact')}) @ {e.get('time')}" for e in upcoming
        ))

    recent = (news_ctx.get("recent_releases") or [])[:2]
    if recent:
        lines.append("Recent releases: " + " | ".join(
            f"{e.get('event')} actual={e.get('actual')} (forecast {e.get('forecast')})" for e in recent
        ))

    if not lines:
        return "News risk engine: no active freeze, no killswitch, no high-impact USD events in range."

    return "\n".join(lines)


def _format_bias_and_scenarios(bias_result: Optional[Dict[str, Any]]) -> str:
    if not bias_result:
        return "Bias/scenario engine: not available right now."

    tf_biases = bias_result.get("timeframe_biases", {}) or {}
    lines = [
        f"Overall multi-timeframe bias: {bias_result.get('overall_bias', 'NEUTRAL')}  "
        f"(confluence score: {bias_result.get('confluence_score', 1)}/5)",
    ]
    if tf_biases:
        lines.append(
            "Per-TF bias — Daily: {daily_bias}, 4H: {h4_bias}, 1H: {h1_bias}, 15M: {m15_bias}".format(
                daily_bias=tf_biases.get("daily_bias", "NEUTRAL"),
                h4_bias=tf_biases.get("h4_bias", "NEUTRAL"),
                h1_bias=tf_biases.get("h1_bias", "NEUTRAL"),
                m15_bias=tf_biases.get("m15_bias", "NEUTRAL"),
            )
        )

    scenarios = bias_result.get("active_scenarios") or []
    if scenarios:
        lines.append("Active conditional scenarios:")
        for s in scenarios:
            lines.append(
                f"  • [{s.get('scenario_type')}] IF: {s.get('trigger_condition')} "
                f"| Invalidation: {s.get('invalidation_level')} "
                f"| Targets: {s.get('target_levels')} | Risk: {s.get('risk_rating')}"
            )
    else:
        lines.append("No fully-confirmed conditional scenarios active right now — conditions not yet met.")

    return "\n".join(lines)


def _truncate(text: Optional[str], limit: int = _MEMORY_TEXT_TRUNCATE) -> str:
    if not text:
        return "(no text logged)"
    text = " ".join(text.split())  # collapse newlines/whitespace
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _relative_time(iso_timestamp: Optional[str]) -> str:
    """Renders a stored UTC ISO timestamp as '23 min ago' / '3h ago' /
    '2d ago' for compact, model-friendly recency framing. Never raises —
    falls back to the raw string (or 'unknown time ago') on a bad value."""
    if not iso_timestamp:
        return "unknown time ago"
    try:
        ts = datetime.fromisoformat(iso_timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        seconds = max(delta.total_seconds(), 0)
        if seconds < 3600:
            return f"{int(seconds // 60)} min ago"
        if seconds < 86400:
            return f"{seconds / 3600:.1f}h ago"
        return f"{int(seconds // 86400)}d ago"
    except (ValueError, TypeError):
        return f"{iso_timestamp} (unparsed timestamp)"


def _format_conversation_memory(user_id: Optional[str], entries: list[Dict[str, Any]]) -> str:
    if not user_id:
        return "Conversation memory: no user_id supplied for this turn — recall skipped."
    if not entries:
        return "Conversation memory: no prior logged interactions with this user."

    lines = [f"Last {len(entries)} logged interaction(s) with this user (most recent first):"]
    for entry in entries:
        when = _relative_time(entry.get("timestamp"))
        cmd = entry.get("command_name") or "chat"
        prompt = _truncate(entry.get("prompt_text"))
        response = _truncate(entry.get("response_text"))
        lines.append(f"  • [{when}, {cmd}] User asked: \"{prompt}\" → You replied: \"{response}\"")
    return "\n".join(lines)


def _format_alert_outcomes(symbol: str, alerts: list[Dict[str, Any]]) -> str:
    if not alerts:
        return f"Alert/scenario history: no prior {symbol} alerts logged."

    lines = [f"Last {len(alerts)} logged {symbol} alert(s)/scenario(s) (most recent first):"]
    for alert in alerts:
        when = _relative_time(alert.get("timestamp"))
        event_type = alert.get("event_type", "EVENT")
        price_at_alert = alert.get("price_at_alert")
        payload = alert.get("payload") or {}
        label = payload.get("scenario") or payload.get("scenario_type") or event_type

        base = f"  • [{when}] {label}"
        if price_at_alert is not None:
            base += f" generated at {price_at_alert}"

        outcome_bits = []
        if alert.get("outcome_30m") is not None:
            outcome_bits.append(f"30m move: {alert['outcome_30m']:+.2f}")
        if alert.get("outcome_2h") is not None:
            outcome_bits.append(f"2h move: {alert['outcome_2h']:+.2f}")

        if outcome_bits:
            base += " — " + " | ".join(outcome_bits)
        else:
            base += " — outcome not yet due/evaluated"

        lines.append(base)
    return "\n".join(lines)


def _country_to_timezone(country: Optional[str]) -> Optional[tuple[str, float]]:
    """Looks up a free-text country string in _COUNTRY_TIMEZONES. Case/
    whitespace-insensitive; returns None (never guesses) for anything not
    in the table."""
    if not country:
        return None
    return _COUNTRY_TIMEZONES.get(country.strip().lower())


def _classify_broker_spread_profile(broker_server: Optional[str]) -> str:
    """
    Heuristic-only classification of a broker/server string into a
    *typical* spread profile, based on naming conventions common across
    brokers (Raw/Zero/ECN/Pro accounts vs. Standard ones). This is
    explicitly NOT a real per-broker spread lookup — we have no live feed
    of actual broker terms — so the wording always frames it as a
    heuristic and defers to the live spread figure already present in the
    LIVE DATA CONTEXT block, consistent with the system prompt's STRICT
    ACCURACY RULE (never invent a number we don't actually have).
    """
    if not broker_server:
        return "Not stated — no spread-profile assumption made."

    lowered = broker_server.lower()
    raw_markers = ("raw", "zero", "ecn", "pro", "prime")
    if any(marker in lowered for marker in raw_markers):
        return (
            f"'{broker_server}' naming suggests a Raw/ECN-style account — typically tight "
            f"raw spreads plus a separate per-lot commission. Treat as a heuristic; confirm "
            f"against the live spread reading in the data block above."
        )
    return (
        f"'{broker_server}' reads as a standard/market-maker-style account — typically wider "
        f"all-in spreads with no separate commission. Treat as a heuristic; confirm against "
        f"the live spread reading in the data block above."
    )


def _format_user_profile(user_id: Optional[str], prefs: Optional[Any]) -> str:
    if not user_id:
        return "Trading profile: no user_id supplied for this turn — personalization skipped."
    if prefs is None:
        return "Trading profile: no /makeaccount profile saved for this user yet — treat generically."

    lines = [f"Broker/server: {prefs.broker_server}"]
    lines.append("Spread profile (heuristic — see rule below): " + _classify_broker_spread_profile(prefs.broker_server))

    pairs = prefs.preferred_pairs or []
    if pairs:
        gold_flag = " (includes Gold/XAUUSD — weight SMC/liquidity analysis toward it when relevant)" if "XAUUSD" in pairs else ""
        lines.append(f"Preferred pairs: {', '.join(pairs)}{gold_flag}")
    else:
        lines.append("Preferred pairs: none saved.")

    tz = _country_to_timezone(prefs.country)
    if tz:
        tz_label, offset = tz
        try:
            local_now = datetime.now(timezone(timedelta(hours=offset))).strftime("%H:%M")
            lines.append(
                f"Country: {prefs.country} → approx local time zone {tz_label} (UTC{offset:+g}), "
                f"current local time ~{local_now} (standard-time approximation, not DST-adjusted)."
            )
        except Exception:
            lines.append(f"Country: {prefs.country} (time zone lookup succeeded but local-time render failed).")
    else:
        lines.append(f"Country: {prefs.country} (not in local time zone lookup table — make no local-session-time claims for this user).")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Live data gathering — calls the bot's OWN functions, never guesses
# --------------------------------------------------------------------------- #

async def _gather_market_context(symbol: str) -> str:
    """
    Builds the full "INSTITUTIONAL MARKET CONTEXT" block for `symbol` by
    concurrently querying:
        - mt5_engine (legacy signal engine: action/confidence/RSI/ATR/spread)
        - structure_engine  (1D/4H/1H/15M trend, BOS, CHOCH, Premium/Discount)
        - smc_engine        (unfilled FVGs, order blocks, BSL/SSL, sweeps)
        - session_engine    (active session, time remaining, PDH/PDL/PWH/PWL/
                              PMH/PML, Asian High/Low)
        - news_engine       (institutional USD news risk summary)
        - bias_engine       (overall multi-TF bias, confluence score, and
                              conditional trade scenarios)

    Every call is individually wrapped (see the `_safe_*` helpers above),
    so a single engine failing degrades only its own section of the
    returned text instead of raising or blanking the whole context. This
    function never raises.
    """
    (
        analysis,
        spread,
        structure_result,
        smc_result,
        session_ctx,
        news_ctx,
        bias_result,
    ) = await asyncio.gather(
        _safe_legacy_analysis(symbol),
        _safe_legacy_spread(symbol),
        _safe_structure(symbol),
        _safe_smc(symbol),
        _safe_session(symbol),
        _safe_institutional_news(),
        _safe_bias(symbol),
    )

    sections = [
        f"### {symbol} — INSTITUTIONAL MARKET CONTEXT",
        "-- Live Price / Legacy Signal Engine --\n" + _format_legacy_signal(symbol, analysis, spread),
        "-- Session & Institutional Reference Levels --\n" + _format_session_and_levels(session_ctx),
        "-- Multi-Timeframe Structure (BOS / CHOCH / Premium-Discount) --\n" + _format_structure(structure_result),
        "-- Active Liquidity & SMC Levels --\n" + _format_smc(smc_result),
        "-- USD News Risk Status --\n" + _format_news_risk(news_ctx),
        "-- Overall Bias, Confluence & Trade Scenarios --\n" + _format_bias_and_scenarios(bias_result),
    ]

    return "\n\n".join(sections)


async def _gather_news_context() -> str:
    lines = []

    try:
        killswitch = await news_engine.check_emergency_killswitch()
        if killswitch.triggered:
            lines.append(f"🛑 EMERGENCY KILLSWITCH ACTIVE: {killswitch.reason} (headline: {killswitch.matched_headline})")
    except Exception:
        logger.exception("ai_engine: killswitch check failed")

    try:
        frozen, event = await news_engine.is_news_freeze_active()
        if frozen and event:
            lines.append(f"⏸ News freeze ACTIVE — {event.event_name} ({event.country}) at {event.event_time.isoformat()}")
    except Exception:
        logger.exception("ai_engine: news freeze check failed")

    try:
        events = await news_engine.check_economic_calendar()
        now = datetime.now(timezone.utc)
        upcoming_high = [
            e for e in events
            if e.is_high_impact() and e.event_time >= now
        ][:3]
        if upcoming_high:
            lines.append("Upcoming high-impact events:")
            for e in upcoming_high:
                lines.append(f"  • {e.event_time.isoformat()} — {e.country} {e.event_name}")
    except Exception:
        logger.exception("ai_engine: calendar fetch failed")

    if not lines:
        return "### News/Calendar\nNo active freeze, no killswitch, no high-impact events in the near-term window."
    return "### News/Calendar\n" + "\n".join(lines)


async def _gather_memory_context(user_id: Optional[str], symbols: list[str]) -> str:
    """
    Builds the "MEMORY & HISTORICAL CONTEXT" block (Phase 9): this user's
    recent conversation history plus recent alert/scenario outcomes for
    every symbol mentioned in the current message. Every underlying call
    goes through a _safe_* wrapper, so a memory_engine/DB outage degrades
    this section to a "not available" note instead of breaking the
    response. Never raises.
    """
    convo_entries, *alert_lists = await asyncio.gather(
        _safe_conversation_memory(user_id),
        *[_safe_alert_history(sym) for sym in symbols],
    )

    sections = [
        "### MEMORY & HISTORICAL CONTEXT",
        "-- Recent Conversation With This User --\n" + _format_conversation_memory(user_id, convo_entries),
    ]
    for sym, alerts in zip(symbols, alert_lists):
        sections.append(f"-- Recent {sym} Alert/Scenario Outcomes --\n" + _format_alert_outcomes(sym, alerts))

    return "\n\n".join(sections)


async def _gather_user_profile_context(user_id: Optional[str]) -> str:
    """
    Builds the "USER TRADING PROFILE" block (Phase 10): the requesting
    user's saved /makeaccount profile — broker/server, preferred pairs,
    and country (mapped to an approximate local time zone) — so the model
    can personalize session-timing callouts and pair prioritization.
    Goes through _safe_user_preferences(), so a missing database module
    or DB outage degrades this to a "not available" note rather than
    breaking the response. Never raises.
    """
    prefs = await _safe_user_preferences(user_id)
    return "### USER TRADING PROFILE\n" + _format_user_profile(user_id, prefs)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _system_prompt() -> str:
    return (
        "You are Venchick AI, the Lead Institutional Gold Analyst embedded in a trading Discord "
        "server. You speak with the clarity and terminology of an institutional Smart Money "
        "Concepts (SMC) / ICT-style trading desk — market structure, liquidity, order blocks, "
        "fair value gaps, premium/discount, session timing — not retail chatroom hype.\n\n"
        "STRICT ACCURACY RULE: You will be given an 'INSTITUTIONAL MARKET CONTEXT' / 'LIVE DATA "
        "CONTEXT' block before each user message. That block is the ONLY source of truth for "
        "prices, indicator values (RSI, ATR), spreads, session state, PDH/PDL/PWH/PWL/PMH/PML/"
        "Asian High-Low levels, structure state (trend bias, BOS, CHOCH, Premium/Discount zone), "
        "liquidity/SMC levels (FVGs, order blocks, BSL/SSL, sweeps), news/economic-calendar facts, "
        "and multi-timeframe bias/confluence/scenario data. NEVER invent, estimate, or recall a "
        "price, level, or indicator value from your own training data — if the context block "
        "doesn't contain something, say plainly that you don't have live data for it right now "
        "rather than guessing.\n\n"
        "NO FINANCIAL ADVICE / NO DIRECTIVE CALLS: You NEVER give financial advice, NEVER promise "
        "a guaranteed outcome or profit, and NEVER issue a bare directive like 'Buy now' or 'Sell "
        "now'. You are an analyst describing what the data shows and what would need to happen "
        "next — the trader in front of you makes their own decisions and manages their own risk.\n\n"
        "CONDITIONAL REASONING: When you discuss a directional idea, ALWAYS frame it as an "
        "explicit 'IF... THEN...' scenario grounded in the context block — e.g. liquidity sweeps "
        "of a specific level, a Market Structure Shift / CHOCH on a stated timeframe, price "
        "reacting to a stated order block or FVG, and the session/timing behind it. If the "
        "'Active conditional scenarios' section is empty, say plainly that no scenario has fully "
        "confirmed yet rather than manufacturing one.\n\n"
        "RISK & NEWS AWARENESS: Always respect the news risk status in the context block. If a "
        "news freeze or emergency killswitch is active, or a high-impact USD event is imminent, "
        "lead with that and caution that new positioning carries elevated risk right now — do not "
        "downplay it. When you do discuss a scenario's risk_rating, pass it along honestly rather "
        "than softening a HIGH rating.\n\n"
        "PERSISTENT MEMORY (Phase 9): You have persistent memory of past market calls, user "
        "queries, and scenario outcomes, delivered to you each turn in a 'MEMORY & HISTORICAL "
        "CONTEXT' block — treat it with the same STRICT ACCURACY RULE as the live data block: "
        "it is real logged history, not something to embellish. When it's genuinely relevant, "
        "reference past predictions or market setups naturally (e.g. 'As noted in our previous "
        "analysis 2 hours ago...') and continuously refine your read based on whether prior "
        "liquidity pools, FVGs, or order blocks were respected or invalidated since. If that "
        "block shows no prior history, or history that isn't relevant to the current question, "
        "don't force a callback — plain analysis is fine. Never claim to remember something that "
        "isn't actually present in that block.\n\n"
        "PERSONALIZATION (Phase 10): You'll also receive a 'USER TRADING PROFILE' block built "
        "from the trader's /makeaccount registration — their broker/server, preferred pairs, and "
        "country. Use it to personalize, not to gate: prioritize SMC/liquidity analysis on the "
        "pairs they said they prefer (especially Gold/XAUUSD if it's one of them) when the "
        "question is open-ended about 'what looks good', but still answer fully and accurately "
        "if they ask about a different pair. When their country maps to a known local time zone, "
        "you can reference session timing in their local terms (e.g. 'London Open is at 3pm your "
        "time') — but if that block says the country isn't in the lookup table, don't invent a "
        "time zone or a local time for them. The spread-profile line is an explicit heuristic "
        "based on their broker's name, not verified data — mention it only in passing if at all, "
        "and always defer to the actual live spread figure in the LIVE DATA CONTEXT block, never "
        "to the heuristic, when the two would conflict. If no profile block is present or it says "
        "no profile is saved, just answer normally without personalization.\n\n"
        "TONE: Confident, concise, professional-but-personable — like a sharp institutional desk "
        "analyst, not a generic chatbot. You can also just chat casually if the user isn't asking "
        "about markets. Keep replies tight (a few sentences to a short paragraph, plus bullet "
        "levels/scenarios when relevant) — no walls of text.\n\n"
        "RISK DISCLAIMER: When you discuss a directional bias, a scenario, or specific price "
        "levels, keep it brief and note this is analysis, not financial advice — do not repeat "
        "this disclaimer in every single message, only when giving a concrete market read.\n\n"
        "CHART RULE: If the user is explicitly asking to SEE a chart, graph, picture, or visual of "
        "Gold, end your reply with a line containing EXACTLY `[GENERATE_CHART:SYMBOL]` (use the "
        "exact broker Gold symbol in uppercase, e.g. `[GENERATE_CHART:XAUUSD]`) and nothing after "
        "it. Only include this tag when a chart was actually requested — never for plain text "
        "questions. Do NOT generate chart tags for non-Gold instruments."
    )


def _build_user_turn(user_message: str, live_context: str, channel_context: dict) -> str:
    trader_line = ""
    if channel_context.get("balance") is not None:
        trader_line = (
            f"(This trader's account: {channel_context.get('account_type', 'Unknown')} type, "
            f"balance ${channel_context['balance']:.2f}.)\n"
        )

    return (
        f"[LIVE DATA CONTEXT]\n{live_context}\n\n"
        f"{trader_line}"
        f"User ({channel_context.get('username', 'trader')}): {user_message}"
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def generate_ai_response(user_message: str, channel_context: dict) -> str:
    """
    Gathers live market/news data via the bot's own MT5 + institutional
    engines, injects it into the prompt, and asks Groq (Llama 3.3 70B)
    for a grounded reply.

    channel_context (all keys optional, pass what you have):
        {
            "channel_id": int,
            "username": str,
            "is_private_room": bool,
            "balance": float | None,
            "account_type": str | None,
            "user_id": int | str | None,     # Phase 9 — enables per-user
                                              # conversation recall + logging.
                                              # Without it, memory recall/
                                              # caching is skipped for this
                                              # turn (everything else works
                                              # exactly as before).
            "message_id": int | str | None,  # Phase 9 — Discord message ID,
                                              # used as the message_logs
                                              # primary key. If omitted, a
                                              # synthetic key is generated
                                              # so the interaction still
                                              # gets logged.
        }

    Returns the raw text reply, which may end with a `[GENERATE_CHART:SYMBOL]`
    tag — pass the return value through extract_chart_tag() before sending
    it to Discord.
    """
    client = _get_client()
    if client is None:
        return "⚠️ Venchick AI isn't configured yet — ask an admin to set GROQ_API_KEY."

    symbols = _detect_symbols(user_message)
    wants_chart = _wants_chart(user_message)
    chart_symbol = symbols[0] if (wants_chart and symbols) else None

    context_blocks: list[str] = []
    for sym in symbols[:2]:  # cap prompt size — most questions reference 1 pair
        context_blocks.append(await _gather_market_context(sym))

    if symbols or _mentions_news(user_message):
        context_blocks.append(await _gather_news_context())

    # Phase 9 — persistent memory: this user's recent conversation plus
    # recent alert/scenario outcomes for whatever symbol(s) are in play.
    # Always gathered (even with no symbols) so conversational continuity
    # ("what did you tell me earlier?") works outside of market questions
    # too; a missing user_id or an empty history degrades to a plain
    # "no prior history" line rather than omitting the section.
    user_id = channel_context.get("user_id")
    context_blocks.append(await _gather_memory_context(user_id, symbols[:2]))
    context_blocks.append(await _gather_user_profile_context(user_id))

    live_context = "\n\n".join(context_blocks) if context_blocks else (
        "No specific pair mentioned in this message — general conversation, no market data needed."
    )

    channel_id = channel_context.get("channel_id")
    history = _CHANNEL_HISTORY.setdefault(channel_id, deque(maxlen=MAX_HISTORY_TURNS * 2))

    # Groq's chat.completions API uses OpenAI-style {"role", "content"}
    # messages, with "model" turns represented as "assistant".
    messages = [{"role": "system", "content": _system_prompt()}]
    messages += [
        {"role": ("assistant" if role == "model" else role), "content": text}
        for role, text in history
    ]
    messages.append({"role": "user", "content": _build_user_turn(user_message, live_context, channel_context)})

    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5,
            max_completion_tokens=700,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("ai_engine: Groq call failed")
        return "⚠️ Venchick AI hit an error reaching Groq — try again in a moment."

    if not text:
        text = "I don't have a clear read on that right now — could you rephrase?"

    # Deterministic safety net: if chart intent was detected in code but the
    # model forgot the tag, append it ourselves so the feature never silently
    # fails.
    if chart_symbol and f"[GENERATE_CHART:{chart_symbol}]" not in text:
        text = f"{text}\n[GENERATE_CHART:{chart_symbol}]"

    history.append(("user", user_message))
    history.append(("model", text))

    # Phase 9 — automatic message caching: log every prompt/response to
    # disk instantly so it survives a restart and feeds future recall via
    # _gather_memory_context(). Fired as a background task (not awaited)
    # so a slow/failed DB write never adds latency to the Discord reply;
    # _safe_log_interaction swallows its own exceptions so an unretrieved
    # task exception can't surface as a warning/crash later either.
    asyncio.create_task(
        _safe_log_interaction(
            message_id=channel_context.get("message_id"),
            channel_id=channel_id,
            user_id=user_id,
            command_name="ai_chat",
            prompt=user_message,
            response=text,
            market_data_dict={
                "symbols": symbols,
                "chart_symbol": chart_symbol,
                "wants_chart": wants_chart,
            },
        )
    )

    return text


async def _safe_log_interaction(
    message_id: Optional[Any],
    channel_id: Optional[Any],
    user_id: Optional[Any],
    command_name: str,
    prompt: str,
    response: str,
    market_data_dict: Dict[str, Any],
) -> None:
    """
    Fire-and-forget wrapper around memory_engine.log_interaction() for
    Phase 9 automatic message caching. Never raises — a logging failure
    (missing memory_engine, DB locked/unavailable, etc.) is recorded to
    the logger only and must never affect the chat flow, which has
    already returned its reply to the caller by the time this runs.

    message_id is required by message_logs' schema (PRIMARY KEY); if the
    caller didn't pass one via channel_context (main.py currently doesn't
    thread the Discord message ID through), we synthesize a unique key so
    the row still gets written instead of silently being skipped —
    conversation recall (get_recent_conversation_memory) reads by
    user_id/timestamp, not message_id, so this doesn't affect Phase 9
    recall quality, only the primary key value.
    """
    if memory_engine is None:
        return
    try:
        resolved_message_id = message_id or f"ai_chat-{datetime.now(timezone.utc).timestamp()}"
        ok = await memory_engine.log_interaction(
            message_id=resolved_message_id,
            channel_id=channel_id or "unknown_channel",
            user_id=user_id or "unknown_user",
            command_name=command_name,
            prompt=prompt,
            response=response,
            market_data_dict=market_data_dict,
        )
        if not ok:
            logger.warning("ai_engine: log_interaction reported failure for message_id=%s", resolved_message_id)
    except Exception:
        logger.exception("ai_engine: log_interaction raised unexpectedly")


def extract_chart_tag(text: str) -> tuple[str, Optional[str]]:
    """Strips a trailing [GENERATE_CHART:SYMBOL] tag and returns (clean_text, symbol_or_None)."""
    match = _CHART_TAG_RE.search(text)
    if not match:
        return text, None
    symbol = match.group(1)
    cleaned = _CHART_TAG_RE.sub("", text).strip()
    return cleaned, symbol


# --------------------------------------------------------------------------- #
# Standalone smoke test — python ai_engine.py
#
# Prints the newly formatted institutional prompt context (and the system
# prompt) to the terminal for manual verification. Does NOT call Groq, so
# it works even without GROQ_API_KEY set — it only exercises
# _gather_market_context(), which is the piece Phase 6 changed. Requires
# a running/loggable MT5 terminal, exactly like the other engines'
# standalone test blocks.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    def _run() -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            print(
                "MetaTrader5 package not found — this smoke test needs it to pull live data "
                "(Windows only). Install with: pip install MetaTrader5"
            )
            return

        if not mt5.initialize():
            print(f"mt5.initialize() failed: {mt5.last_error()}")
            return

        try:
            print("=== Phase 6 smoke test: institutional AI context for XAUUSD ===\n")
            context = asyncio.run(_gather_market_context("XAUUSD"))
            print(context)

            print("\n\n=== System prompt (Lead Institutional Gold Analyst persona) ===\n")
            print(_system_prompt())
        finally:
            mt5.shutdown()

    _run()
