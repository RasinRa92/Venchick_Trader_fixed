"""
memory_engine.py
================
Phase 8: Persistent State Caching & Event Memory Engine.

Makes the bot resilient to restarts by caching every command interaction
(prompt/response/market snapshot) and every triggered alert (news, SMC
liquidity sweeps, bias scenarios) into SQLite via `database.py`. On
startup, `restore_bot_state_on_startup()` reads that history back so the
bot knows what it already told people and which alerts are still waiting
on an outcome, instead of starting from a blank slate every restart.

This module is additive and self-contained:
    * It does NOT modify mt5_engine.py, ai_engine.py, news_engine.py,
      chart_engine.py, structure_engine.py, smc_engine.py,
      session_engine.py, or bias_engine.py.
    * It reuses `database.DB_PATH` / `database.init_db()` and talks to the
      `message_logs` / `alert_logs` tables added to database.py in Phase 8
      (both created via `CREATE TABLE IF NOT EXISTS`, so calling
      `database.init_db()` is always safe to repeat).
    * mt5_engine is imported defensively (try/except) purely so this
      module can be imported and unit-tested in environments where the
      native MetaTrader5 package isn't installed (e.g. CI, this sandbox).
      In production, main.py already depends on mt5_engine directly, so
      this guard changes nothing at runtime.

Wire-up in main.py:
    async def on_ready():
        await database.init_db(db_path=DB_PATH)
        summary = await memory_engine.restore_bot_state_on_startup(db_path=DB_PATH)
        logger.info("Startup recovery: %s", summary)
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import aiosqlite

import database
from database import DB_PATH, DatabaseError

logger = logging.getLogger("forex_bot.memory_engine")

# mt5_engine is only needed by evaluate_past_alerts_outcomes() for live
# price checks. Imported defensively so the rest of this module (logging,
# recovery, recall) still works in environments without a live MT5
# terminal (tests, CI, this sandbox).
try:
    import mt5_engine
    _MT5_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host environment
    mt5_engine = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False
    logger.warning("mt5_engine not importable — evaluate_past_alerts_outcomes() will skip price checks.")

# news_engine is only used, best-effort, to report whether a news freeze
# was active across the downtime window. Same defensive-import pattern.
try:
    import news_engine
    _NEWS_ENGINE_AVAILABLE = True
except Exception:  # pragma: no cover
    news_engine = None  # type: ignore[assignment]
    _NEWS_ENGINE_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

class AlertEventType:
    """Mirrors the CHECK constraint on `alert_logs.event_type` in database.py."""
    NEWS_WARNING = "NEWS_WARNING"
    NEWS_RELEASE = "NEWS_RELEASE"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    BIAS_SCENARIO = "BIAS_SCENARIO"
    # Second-opinion LLM veto/adjustment verdict on a signal about to be
    # broadcast — see second_opinion_engine.py. database.py's alert_logs
    # CHECK constraint has a matching migration for this value.
    SECOND_OPINION = "SECOND_OPINION"


VALID_ALERT_EVENT_TYPES = {
    AlertEventType.NEWS_WARNING,
    AlertEventType.NEWS_RELEASE,
    AlertEventType.LIQUIDITY_SWEEP,
    AlertEventType.BIAS_SCENARIO,
    AlertEventType.SECOND_OPINION,
}

# Outcome checkpoints. An alert becomes eligible for its 30m outcome once
# it's at least this old, and for its 2h outcome once it's at least that
# old — evaluate_past_alerts_outcomes() is meant to be called periodically
# (e.g. from a tasks.loop in main.py) and only fills in whichever
# checkpoints have come due since the last run.
OUTCOME_WINDOW_30M = timedelta(minutes=30)
OUTCOME_WINDOW_2H = timedelta(hours=2)

DEFAULT_PRICE_TIMEFRAME = "M1"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class StartupSummary:
    last_active_timestamp: Optional[str]
    total_cached_calls: int
    unresolved_alerts_count: int
    active_news_freeze: Optional[bool] = None
    active_news_freeze_event: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "last_active_timestamp": self.last_active_timestamp,
            "total_cached_calls": self.total_cached_calls,
            "unresolved_alerts_count": self.unresolved_alerts_count,
            "active_news_freeze": self.active_news_freeze,
            "active_news_freeze_event": self.active_news_freeze_event,
        }


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(payload: Optional[dict[str, Any]]) -> Optional[str]:
    if payload is None:
        return None
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        logger.exception("Failed to JSON-encode payload — storing as string repr instead.")
        return json.dumps({"_unserializable": str(payload)})


def _loads(raw: Optional[str]) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Failed to decode stored JSON payload — returning raw string.")
        return {"_raw": raw}


async def _get_current_price(symbol: str) -> Optional[float]:
    """
    Best-effort live price lookup via mt5_engine, used only by
    evaluate_past_alerts_outcomes(). Returns None (never raises) if MT5
    isn't available/connected or the symbol has no recent candle — the
    caller simply leaves that alert's outcome for the next evaluation pass.
    """
    if not _MT5_AVAILABLE or mt5_engine is None:
        return None
    try:
        df = await mt5_engine.get_market_data(symbol, DEFAULT_PRICE_TIMEFRAME, count=1)
        if df is None or df.empty:
            return None
        return float(df.iloc[-1]["close"])
    except Exception:
        logger.exception("Failed to fetch current price for %s.", symbol)
        return None


# --------------------------------------------------------------------------- #
# 1. Interaction logging
# --------------------------------------------------------------------------- #

async def log_interaction(
    message_id: str,
    channel_id: str,
    user_id: str,
    command_name: str,
    prompt: Optional[str],
    response: Optional[str],
    market_data_dict: Optional[dict[str, Any]] = None,
    db_path: str = DB_PATH,
) -> bool:
    """
    Persist a full command interaction so it survives a restart.

    Uses INSERT OR REPLACE keyed on message_id, so re-logging the same
    Discord message (e.g. an edited response) safely overwrites the prior
    row instead of erroring on the primary key.

    Args:
        message_id: Discord message ID (stringified) of the bot's reply —
            use whatever unique-per-interaction key you have if there's no
            message yet (e.g. the interaction ID) as long as it's stable.
        channel_id: Discord channel ID (stringified).
        user_id: Discord user ID (stringified) of the requester.
        command_name: Slash command or trigger name, e.g. "balance", "ai_chat".
        prompt: The user's input / question, if any.
        response: The bot's textual response, if any.
        market_data_dict: Optional snapshot of market context at the time
            (e.g. symbol, price, indicators) — JSON-encoded for storage.
        db_path: Path to the SQLite database file.

    Returns:
        True on success, False if the write failed (failure is logged, not
        raised — a memory-logging failure should never take down a live
        trading/signal flow).
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO message_logs
                    (message_id, channel_id, user_id, command_name,
                     prompt_text, response_text, market_snapshot_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(message_id), str(channel_id), str(user_id), command_name,
                    prompt, response, _dumps(market_data_dict), _utcnow_iso(),
                ),
            )
            await db.commit()
        return True

    except aiosqlite.Error:
        logger.exception("log_interaction failed for message_id=%s.", message_id)
        return False


# --------------------------------------------------------------------------- #
# 2. Alert logging
# --------------------------------------------------------------------------- #

async def log_alert_event(
    channel_id: str,
    message_id: Optional[str],
    event_type: str,
    symbol: str,
    payload_dict: Optional[dict[str, Any]],
    current_price: Optional[float],
    db_path: str = DB_PATH,
) -> Optional[int]:
    """
    Persist a triggered alert (news warning/release, liquidity sweep, bias
    scenario) so it can be recovered and later scored by
    evaluate_past_alerts_outcomes().

    Args:
        channel_id: Discord channel ID (stringified) the alert was posted to.
        message_id: Discord message ID (stringified) of the alert card, if
            one was posted (None for silently-logged internal events).
        event_type: One of AlertEventType.{NEWS_WARNING, NEWS_RELEASE,
            LIQUIDITY_SWEEP, BIAS_SCENARIO}.
        symbol: The instrument the alert concerns, e.g. "XAUUSD".
        payload_dict: Arbitrary structured detail about the alert (target
            levels, invalidation, headline text, confidence, etc.) —
            JSON-encoded for storage.
        current_price: Market price at the moment the alert fired, used as
            the baseline for outcome_30m / outcome_2h deltas later.
        db_path: Path to the SQLite database file.

    Returns:
        The new alert_id, or None if the write failed or event_type was
        invalid (both cases are logged).
    """
    if event_type not in VALID_ALERT_EVENT_TYPES:
        logger.error("log_alert_event: invalid event_type %r for symbol %s.", event_type, symbol)
        return None

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO alert_logs
                    (channel_id, message_id, event_type, symbol, payload_json,
                     price_at_alert, outcome_30m, outcome_2h, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (
                    str(channel_id) if channel_id is not None else None,
                    str(message_id) if message_id is not None else None,
                    event_type, symbol, _dumps(payload_dict), current_price,
                    _utcnow_iso(),
                ),
            )
            await db.commit()
            alert_id = cursor.lastrowid

        logger.info("Logged alert #%s: %s %s @ %s", alert_id, event_type, symbol, current_price)
        return alert_id

    except aiosqlite.Error:
        logger.exception("log_alert_event failed for symbol=%s event_type=%s.", symbol, event_type)
        return None


# --------------------------------------------------------------------------- #
# 3. Startup recovery
# --------------------------------------------------------------------------- #

async def restore_bot_state_on_startup(db_path: str = DB_PATH) -> dict[str, Any]:
    """
    Scan message_logs / alert_logs to rebuild a picture of what the bot
    was doing before it restarted, and report anything left unresolved.

    "Unresolved" alerts are ones old enough that a checkpoint outcome
    should have been computed by now but wasn't (i.e. the bot was down
    across that checkpoint) — an alert younger than 30 minutes with a NULL
    outcome_30m is normal and not counted; one older than 30 minutes with
    a NULL outcome_30m, or older than 2 hours with a NULL outcome_2h, is.

    Safe to call multiple times; does not mutate any data, only reads.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A dict with:
            last_active_timestamp:  ISO timestamp of the most recent row
                across message_logs and alert_logs, or None if both are empty.
            total_cached_calls:     Total row count in message_logs.
            unresolved_alerts_count: Count of alerts overdue for a
                30m/2h outcome evaluation (see above).
            active_news_freeze:     True/False if news_engine is available
                and reachable, else None.
            active_news_freeze_event: Name of the event causing the freeze,
                if any.
    """
    # Idempotent — safe even if the caller hasn't already run this.
    try:
        await database.init_db(db_path=db_path)
    except DatabaseError:
        logger.exception("restore_bot_state_on_startup: init_db failed, continuing with a read-only scan.")

    now = datetime.now(timezone.utc)
    cutoff_30m = (now - OUTCOME_WINDOW_30M).isoformat()
    cutoff_2h = (now - OUTCOME_WINDOW_2H).isoformat()

    total_cached_calls = 0
    last_active_timestamp: Optional[str] = None
    unresolved_alerts_count = 0

    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM message_logs") as cursor:
                row = await cursor.fetchone()
                total_cached_calls = int(row[0]) if row else 0

            async with db.execute("SELECT MAX(timestamp) FROM message_logs") as cursor:
                row = await cursor.fetchone()
                last_msg_ts = row[0] if row else None

            async with db.execute("SELECT MAX(timestamp) FROM alert_logs") as cursor:
                row = await cursor.fetchone()
                last_alert_ts = row[0] if row else None

            candidates = [ts for ts in (last_msg_ts, last_alert_ts) if ts]
            last_active_timestamp = max(candidates) if candidates else None

            async with db.execute(
                """
                SELECT COUNT(*) FROM alert_logs
                WHERE (outcome_30m IS NULL AND timestamp <= ?)
                   OR (outcome_2h IS NULL AND timestamp <= ?)
                """,
                (cutoff_30m, cutoff_2h),
            ) as cursor:
                row = await cursor.fetchone()
                unresolved_alerts_count = int(row[0]) if row else 0

    except aiosqlite.Error:
        logger.exception("restore_bot_state_on_startup: failed to scan memory tables.")

    active_news_freeze: Optional[bool] = None
    active_news_freeze_event: Optional[str] = None
    if _NEWS_ENGINE_AVAILABLE and news_engine is not None:
        try:
            frozen, event = await news_engine.is_news_freeze_active()
            active_news_freeze = frozen
            active_news_freeze_event = getattr(event, "event_name", None) if event else None
        except Exception:
            logger.exception("restore_bot_state_on_startup: news freeze check failed.")

    summary = StartupSummary(
        last_active_timestamp=last_active_timestamp,
        total_cached_calls=total_cached_calls,
        unresolved_alerts_count=unresolved_alerts_count,
        active_news_freeze=active_news_freeze,
        active_news_freeze_event=active_news_freeze_event,
    ).as_dict()

    logger.info(
        "Startup recovery: last_active=%s cached_calls=%s unresolved_alerts=%s news_freeze=%s",
        summary["last_active_timestamp"], summary["total_cached_calls"],
        summary["unresolved_alerts_count"], summary["active_news_freeze"],
    )
    return summary


# --------------------------------------------------------------------------- #
# 4. Outcome tracking / performance evaluation
# --------------------------------------------------------------------------- #

async def evaluate_past_alerts_outcomes(db_path: str = DB_PATH) -> dict[str, int]:
    """
    Score alerts that have crossed the 30-minute and/or 2-hour checkpoint
    since they fired, by comparing the current MT5 price against
    `price_at_alert`. The outcome is stored as a signed price delta
    (current_price - price_at_alert) in `outcome_30m` / `outcome_2h` —
    positive means price moved up since the alert, negative means it moved
    down. Callers (e.g. a future AI prompt in Phase 9) can compare that
    delta against whatever target/invalidation levels are in the alert's
    `payload_json` to judge whether the scenario played out.

    Intended to be called periodically (e.g. from a `tasks.loop` in
    main.py) — each run only fills in checkpoints that have newly come due
    and leaves everything else untouched. Alerts whose symbol can't be
    priced right now (MT5 down, symbol not found) are skipped and
    retried on the next run.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        {"evaluated_30m": n, "evaluated_2h": n, "skipped_no_price": n}
    """
    now = datetime.now(timezone.utc)
    cutoff_30m = (now - OUTCOME_WINDOW_30M).isoformat()
    cutoff_2h = (now - OUTCOME_WINDOW_2H).isoformat()

    result = {"evaluated_30m": 0, "evaluated_2h": 0, "skipped_no_price": 0}
    price_cache: dict[str, Optional[float]] = {}

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row

            # --- 30-minute checkpoint ---------------------------------- #
            async with db.execute(
                "SELECT alert_id, symbol, price_at_alert FROM alert_logs "
                "WHERE outcome_30m IS NULL AND timestamp <= ?",
                (cutoff_30m,),
            ) as cursor:
                due_30m = await cursor.fetchall()

            for row in due_30m:
                symbol = row["symbol"]
                if symbol not in price_cache:
                    price_cache[symbol] = await _get_current_price(symbol)
                price = price_cache[symbol]

                if price is None or row["price_at_alert"] is None:
                    result["skipped_no_price"] += 1
                    continue

                delta = price - float(row["price_at_alert"])
                await db.execute(
                    "UPDATE alert_logs SET outcome_30m = ? WHERE alert_id = ?",
                    (delta, row["alert_id"]),
                )
                result["evaluated_30m"] += 1

            # --- 2-hour checkpoint --------------------------------------- #
            async with db.execute(
                "SELECT alert_id, symbol, price_at_alert FROM alert_logs "
                "WHERE outcome_2h IS NULL AND timestamp <= ?",
                (cutoff_2h,),
            ) as cursor:
                due_2h = await cursor.fetchall()

            for row in due_2h:
                symbol = row["symbol"]
                if symbol not in price_cache:
                    price_cache[symbol] = await _get_current_price(symbol)
                price = price_cache[symbol]

                if price is None or row["price_at_alert"] is None:
                    result["skipped_no_price"] += 1
                    continue

                delta = price - float(row["price_at_alert"])
                await db.execute(
                    "UPDATE alert_logs SET outcome_2h = ? WHERE alert_id = ?",
                    (delta, row["alert_id"]),
                )
                result["evaluated_2h"] += 1

            await db.commit()

    except aiosqlite.Error:
        logger.exception("evaluate_past_alerts_outcomes: database error during evaluation.")

    if result["evaluated_30m"] or result["evaluated_2h"] or result["skipped_no_price"]:
        logger.info("evaluate_past_alerts_outcomes: %s", result)
    return result


# --------------------------------------------------------------------------- #
# 5. Recall helpers (for Phase 9 AI prompt context)
# --------------------------------------------------------------------------- #

async def get_recent_conversation_memory(
    user_id: str,
    limit: int = 5,
    db_path: str = DB_PATH,
) -> list[dict[str, Any]]:
    """
    Fetch a user's most recent logged interactions, newest first, for use
    as AI prompt context (Phase 9).

    Args:
        user_id: Discord user ID (stringified).
        limit: Maximum number of interactions to return.
        db_path: Path to the SQLite database file.

    Returns:
        A list of dicts with keys: message_id, channel_id, user_id,
        command_name, prompt_text, response_text, market_snapshot
        (decoded dict or None), timestamp. Empty list on no history or on
        failure (failures are logged, not raised).
    """
    if limit <= 0:
        return []

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM message_logs
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ) as cursor:
                rows = await cursor.fetchall()

        memories = []
        for row in rows:
            entry = dict(row)
            entry["market_snapshot"] = _loads(entry.pop("market_snapshot_json", None))
            memories.append(entry)
        return memories

    except aiosqlite.Error:
        logger.exception("get_recent_conversation_memory failed for user_id=%s.", user_id)
        return []


async def get_recent_alert_history(
    symbol: str = "XAUUSD",
    limit: int = 5,
    db_path: str = DB_PATH,
) -> list[dict[str, Any]]:
    """
    Fetch the most recent alerts logged for a symbol, newest first, for
    use as AI prompt context (Phase 9) — e.g. "here's how the last few
    bias scenarios on XAUUSD actually played out."

    Args:
        symbol: Instrument to filter by, e.g. "XAUUSD".
        limit: Maximum number of alerts to return.
        db_path: Path to the SQLite database file.

    Returns:
        A list of dicts with keys: alert_id, channel_id, message_id,
        event_type, symbol, payload (decoded dict or None), price_at_alert,
        outcome_30m, outcome_2h, timestamp. Empty list on no history or on
        failure (failures are logged, not raised).
    """
    if limit <= 0:
        return []

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM alert_logs
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        alerts = []
        for row in rows:
            entry = dict(row)
            entry["payload"] = _loads(entry.pop("payload_json", None))
            alerts.append(entry)
        return alerts

    except aiosqlite.Error:
        logger.exception("get_recent_alert_history failed for symbol=%s.", symbol)
        return []


# --------------------------------------------------------------------------- #
# Standalone test block
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def _run_demo() -> None:
        test_db = os.path.join(tempfile.gettempdir(), "memory_engine_test.db")
        if os.path.exists(test_db):
            os.remove(test_db)

        print(f"\n== Using scratch DB: {test_db} ==")
        await database.init_db(db_path=test_db)

        print("\n-- log_interaction --")
        ok = await log_interaction(
            message_id="1000000000000000001",
            channel_id="500000000000000001",
            user_id="900000000000000001",
            command_name="balance",
            prompt="/balance",
            response="Your balance is $1,250.00 (Standard).",
            market_data_dict={"symbol": "XAUUSD", "price": 2415.32, "trend": "bullish"},
            db_path=test_db,
        )
        print("log_interaction ok:", ok)

        print("\n-- log_alert_event (backdated for outcome testing) --")
        alert_id = await log_alert_event(
            channel_id="500000000000000002",
            message_id="1000000000000000002",
            event_type=AlertEventType.BIAS_SCENARIO,
            symbol="XAUUSD",
            payload_dict={"scenario": "bullish_continuation", "target": 2430.0, "invalidation": 2405.0},
            current_price=2415.32,
            db_path=test_db,
        )
        print("alert_id:", alert_id)

        # Manually backdate the alert past both checkpoints so this demo
        # exercises evaluate_past_alerts_outcomes() without waiting 2 hours.
        backdated = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        async with aiosqlite.connect(test_db) as db:
            await db.execute(
                "UPDATE alert_logs SET timestamp = ? WHERE alert_id = ?",
                (backdated, alert_id),
            )
            await db.commit()

        print("\n-- restore_bot_state_on_startup --")
        summary = await restore_bot_state_on_startup(db_path=test_db)
        print(summary)

        print("\n-- evaluate_past_alerts_outcomes (no live MT5 expected in this sandbox) --")
        outcome_result = await evaluate_past_alerts_outcomes(db_path=test_db)
        print(outcome_result)
        if not _MT5_AVAILABLE:
            print("(mt5_engine unavailable here, so alerts were skipped rather than scored — expected in this environment.)")

        print("\n-- get_recent_conversation_memory --")
        convo = await get_recent_conversation_memory("900000000000000001", limit=5, db_path=test_db)
        for entry in convo:
            print(entry)

        print("\n-- get_recent_alert_history --")
        alerts = await get_recent_alert_history("XAUUSD", limit=5, db_path=test_db)
        for entry in alerts:
            print(entry)

        os.remove(test_db)
        print("\n== Done, scratch DB removed. ==")

    asyncio.run(_run_demo())
