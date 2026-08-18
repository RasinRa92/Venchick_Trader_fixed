"""
database.py
-----------
Standalone async database module for the Forex Discord Bot.

Uses aiosqlite for fully asynchronous, non-blocking SQLite access.
Manages two core tables:
    - traders : user accounts, balances, and performance stats
    - signals : trade signals issued by the strategy engine

Design notes:
    * A single module-level connection pool is NOT used; instead each
      function opens a short-lived aiosqlite connection. This keeps the
      module simple and safe for use inside a Discord bot's event loop
      (aiosqlite connections are cheap and this avoids cross-coroutine
      connection sharing issues). If you need higher throughput, wrap
      these calls with a connection pool or a single long-lived
      connection guarded by an asyncio.Lock.
    * All functions are typed and defensively coded against bad input
      and SQLite-level errors (aiosqlite.Error).
    * WAL mode is enabled on init for better concurrent read/write
      behavior, which matters for a Discord bot handling many
      concurrent slash commands.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger("forex_bot.database")

DB_PATH = "forex_bot.db"

# --------------------------------------------------------------------------- #
# Constants / enums (kept as plain strings for SQLite simplicity, but
# centralized here so the rest of the codebase doesn't hardcode literals)
# --------------------------------------------------------------------------- #

class AccountType:
    MICRO = "Micro"
    STANDARD = "Standard"


class SignalAction:
    BUY = "BUY"
    SELL = "SELL"


class SignalStatus:
    ACTIVE = "ACTIVE"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    CLOSED = "CLOSED"
    SL_HIT = "SL_HIT"


class TransactionType:
    """Ledger entry kinds recorded in the `transactions` table (Phase 7)."""
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    ADJUSTMENT = "ADJUSTMENT"
    SUBSCRIPTION_FEE = "SUBSCRIPTION_FEE"


VALID_TRANSACTION_TYPES = {
    TransactionType.DEPOSIT,
    TransactionType.WITHDRAWAL,
    TransactionType.ADJUSTMENT,
    TransactionType.SUBSCRIPTION_FEE,
}


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class Trader:
    discord_id: int
    username: str
    balance: float
    account_type: str
    private_channel_id: Optional[int]
    total_profit: float
    win_rate: float
    created_at: str


@dataclass
class UserPreferences:
    """Phase 10 — a trader's saved /makeaccount profile (broker, preferred
    pairs, country, starting balance), used to personalize ai_engine.py's
    Groq system prompt."""
    user_id: int
    broker_server: str
    preferred_pairs: list[str]
    country: str
    starting_balance: float
    created_at: str


@dataclass
class Transaction:
    id: int
    discord_id: int
    type: str
    amount: float
    balance_after: float
    admin_id: Optional[int]
    note: Optional[str]
    timestamp: str


@dataclass
class Signal:
    signal_id: int
    pair: str
    action: str
    entry_price: float
    sl: float
    tp1: float
    tp2: Optional[float]
    tp3: Optional[float]
    status: str
    timestamp: str
    confidence: Optional[float] = None
    recommended_strategy: Optional[str] = None


# --------------------------------------------------------------------------- #
# Custom exceptions
# --------------------------------------------------------------------------- #

class DatabaseError(Exception):
    """Raised when a database operation fails after logging the root cause."""


class TraderAlreadyExistsError(DatabaseError):
    """Raised when attempting to add a trader whose discord_id already exists."""


class TraderNotFoundError(DatabaseError):
    """Raised when a trader lookup / update targets a non-existent discord_id."""


class SignalNotFoundError(DatabaseError):
    """Raised when a signal lookup / update targets a non-existent signal_id."""


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #

async def init_db(db_path: str = DB_PATH) -> None:
    """
    Create the traders and signals tables if they do not already exist,
    and enable WAL journaling for better concurrent access.

    Args:
        db_path: Path to the SQLite database file.

    Raises:
        DatabaseError: If table creation fails.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA foreign_keys=ON;")

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS traders (
                    discord_id          INTEGER PRIMARY KEY,
                    username            TEXT    NOT NULL,
                    balance             REAL    NOT NULL DEFAULT 0.0,
                    account_type        TEXT    NOT NULL DEFAULT 'Micro'
                                        CHECK (account_type IN ('Micro', 'Standard')),
                    private_channel_id  INTEGER,
                    total_profit        REAL    NOT NULL DEFAULT 0.0,
                    win_rate            REAL    NOT NULL DEFAULT 0.0,
                    created_at          TEXT    NOT NULL
                );
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair         TEXT    NOT NULL,
                    action       TEXT    NOT NULL CHECK (action IN ('BUY', 'SELL')),
                    entry_price  REAL    NOT NULL,
                    sl           REAL    NOT NULL,
                    tp1          REAL    NOT NULL,
                    tp2          REAL,
                    tp3          REAL,
                    status       TEXT    NOT NULL DEFAULT 'ACTIVE'
                                 CHECK (status IN
                                     ('ACTIVE', 'TP1_HIT', 'TP2_HIT', 'TP3_HIT',
                                      'CLOSED', 'SL_HIT')),
                    timestamp    TEXT    NOT NULL
                );
                """
            )

            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);"
            )

            # --- migration: `signals.confidence` / `signals.recommended_strategy` --- #
            # Added so replayed/rebroadcast signal cards (e.g. the /makeaccount
            # welcome-room replay in main.py) can show the REAL model confidence
            # and strategy instead of a hardcoded placeholder. SQLite has no
            # "ADD COLUMN IF NOT EXISTS", so check PRAGMA table_info first, same
            # pattern as the Phase 7 `traders.currency` migration above.
            async with db.execute("PRAGMA table_info(signals)") as cursor:
                signal_cols = {row[1] async for row in cursor}
            if "confidence" not in signal_cols:
                await db.execute("ALTER TABLE signals ADD COLUMN confidence REAL")
                logger.info("Migrated signals table: added 'confidence' column.")
            if "recommended_strategy" not in signal_cols:
                await db.execute("ALTER TABLE signals ADD COLUMN recommended_strategy TEXT")
                logger.info("Migrated signals table: added 'recommended_strategy' column.")

            # --- Phase 7 migration: `traders.currency` -------------------- #
            # `balance` and `created_at` already exist on `traders` from the
            # original schema above, so only `currency` is new. SQLite has
            # no "ADD COLUMN IF NOT EXISTS", so we check PRAGMA table_info
            # first — this keeps init_db() safe to call repeatedly against
            # a database created before Phase 7 without dropping data.
            async with db.execute("PRAGMA table_info(traders)") as cursor:
                trader_cols = {row[1] async for row in cursor}
            if "currency" not in trader_cols:
                await db.execute(
                    "ALTER TABLE traders ADD COLUMN currency TEXT NOT NULL DEFAULT 'USD'"
                )
                logger.info("Migrated traders table: added 'currency' column.")

            # --- Phase 7: transactions ledger ------------------------------ #
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    discord_id    INTEGER NOT NULL,
                    type          TEXT    NOT NULL
                                  CHECK (type IN
                                      ('DEPOSIT', 'WITHDRAWAL', 'ADJUSTMENT',
                                       'SUBSCRIPTION_FEE')),
                    amount        REAL    NOT NULL,
                    balance_after REAL    NOT NULL,
                    admin_id      INTEGER,
                    note          TEXT,
                    timestamp     TEXT    NOT NULL,
                    FOREIGN KEY (discord_id) REFERENCES traders(discord_id)
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_discord_id "
                "ON transactions(discord_id);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp "
                "ON transactions(timestamp);"
            )

            # --- Phase 8: message_logs (interaction / conversation memory) - #
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS message_logs (
                    message_id            TEXT PRIMARY KEY,
                    channel_id            TEXT,
                    user_id               TEXT,
                    command_name          TEXT,
                    prompt_text           TEXT,
                    response_text         TEXT,
                    market_snapshot_json  TEXT,
                    timestamp             TEXT    NOT NULL
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_logs_user_id "
                "ON message_logs(user_id);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_logs_timestamp "
                "ON message_logs(timestamp);"
            )

            # --- Phase 8: alert_logs (triggered-alert memory + outcomes) --- #
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_logs (
                    alert_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id     TEXT,
                    message_id     TEXT,
                    event_type     TEXT    NOT NULL
                                   CHECK (event_type IN
                                       ('NEWS_WARNING', 'NEWS_RELEASE',
                                        'LIQUIDITY_SWEEP', 'BIAS_SCENARIO',
                                        'SECOND_OPINION')),
                    symbol         TEXT,
                    payload_json   TEXT,
                    price_at_alert REAL,
                    outcome_30m    REAL,
                    outcome_2h     REAL,
                    timestamp      TEXT    NOT NULL
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_alert_logs_symbol "
                "ON alert_logs(symbol);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_alert_logs_timestamp "
                "ON alert_logs(timestamp);"
            )

            # --- migration: `alert_logs.event_type` CHECK widened to include
            # 'SECOND_OPINION' (the new LLM signal-veto audit log — see
            # second_opinion_engine.py). SQLite has no ALTER TABLE ... ADD/DROP
            # CONSTRAINT, so a pre-existing database (created before this
            # column value existed) needs its table rebuilt to accept the new
            # value: rename old -> create new with the widened CHECK -> copy
            # every row across -> drop old. Detected by checking the stored
            # table SQL for the new value; a freshly created database already
            # has it from the CREATE TABLE above and this block is a no-op.
            async with db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='alert_logs'"
            ) as cursor:
                alert_logs_sql_row = await cursor.fetchone()
            existing_alert_logs_sql = alert_logs_sql_row[0] if alert_logs_sql_row else ""
            if existing_alert_logs_sql and "SECOND_OPINION" not in existing_alert_logs_sql:
                logger.info(
                    "Migrating alert_logs table: widening event_type CHECK to include 'SECOND_OPINION'."
                )
                await db.execute("ALTER TABLE alert_logs RENAME TO alert_logs_old")
                await db.execute(
                    """
                    CREATE TABLE alert_logs (
                        alert_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        channel_id     TEXT,
                        message_id     TEXT,
                        event_type     TEXT    NOT NULL
                                       CHECK (event_type IN
                                           ('NEWS_WARNING', 'NEWS_RELEASE',
                                            'LIQUIDITY_SWEEP', 'BIAS_SCENARIO',
                                            'SECOND_OPINION')),
                        symbol         TEXT,
                        payload_json   TEXT,
                        price_at_alert REAL,
                        outcome_30m    REAL,
                        outcome_2h     REAL,
                        timestamp      TEXT    NOT NULL
                    );
                    """
                )
                await db.execute(
                    """
                    INSERT INTO alert_logs (alert_id, channel_id, message_id, event_type,
                                             symbol, payload_json, price_at_alert,
                                             outcome_30m, outcome_2h, timestamp)
                    SELECT alert_id, channel_id, message_id, event_type,
                           symbol, payload_json, price_at_alert,
                           outcome_30m, outcome_2h, timestamp
                    FROM alert_logs_old;
                    """
                )
                await db.execute("DROP TABLE alert_logs_old")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alert_logs_symbol ON alert_logs(symbol);"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alert_logs_timestamp ON alert_logs(timestamp);"
                )
                logger.info("Migrated alert_logs table successfully.")

            # --- Phase 10: user_preferences (/makeaccount modal profile) --- #
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id           INTEGER PRIMARY KEY,
                    broker_server     TEXT    NOT NULL,
                    preferred_pairs   TEXT    NOT NULL,
                    country           TEXT    NOT NULL,
                    starting_balance  REAL    NOT NULL DEFAULT 1000.0,
                    created_at        TEXT    NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES traders(discord_id)
                );
                """
            )

            await db.commit()
            logger.info("Database initialized at '%s'.", db_path)

    except aiosqlite.Error as exc:
        logger.exception("Failed to initialize database.")
        raise DatabaseError(f"init_db failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Trader operations
# --------------------------------------------------------------------------- #

async def add_trader(
    discord_id: int,
    username: str,
    balance: float = 0.0,
    account_type: str = AccountType.MICRO,
    private_channel_id: Optional[int] = None,
    db_path: str = DB_PATH,
) -> Trader:
    """
    Insert a new trader record.

    Args:
        discord_id: Unique Discord user ID (primary key).
        username: Discord display name at time of registration.
        balance: Starting virtual/real balance.
        account_type: 'Micro' or 'Standard'.
        private_channel_id: Discord channel ID for the trader's private
            signal channel, if already provisioned.
        db_path: Path to the SQLite database file.

    Returns:
        The newly created Trader record.

    Raises:
        TraderAlreadyExistsError: If discord_id is already registered.
        DatabaseError: On any other database failure.
    """
    if account_type not in (AccountType.MICRO, AccountType.STANDARD):
        raise ValueError(f"Invalid account_type: {account_type!r}")

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        async with aiosqlite.connect(db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO traders
                        (discord_id, username, balance, account_type,
                         private_channel_id, total_profit, win_rate, created_at)
                    VALUES (?, ?, ?, ?, ?, 0.0, 0.0, ?)
                    """,
                    (discord_id, username, balance, account_type,
                     private_channel_id, created_at),
                )
                await db.commit()
            except aiosqlite.IntegrityError as exc:
                raise TraderAlreadyExistsError(
                    f"Trader with discord_id={discord_id} already exists."
                ) from exc

        return Trader(
            discord_id=discord_id,
            username=username,
            balance=balance,
            account_type=account_type,
            private_channel_id=private_channel_id,
            total_profit=0.0,
            win_rate=0.0,
            created_at=created_at,
        )

    except TraderAlreadyExistsError:
        raise
    except aiosqlite.Error as exc:
        logger.exception("Failed to add trader %s.", discord_id)
        raise DatabaseError(f"add_trader failed: {exc}") from exc


async def get_trader(discord_id: int, db_path: str = DB_PATH) -> Optional[Trader]:
    """
    Fetch a trader by Discord ID.

    Args:
        discord_id: Unique Discord user ID.
        db_path: Path to the SQLite database file.

    Returns:
        A Trader instance, or None if no matching record exists.

    Raises:
        DatabaseError: On database failure.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traders WHERE discord_id = ?", (discord_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        return Trader(
            discord_id=row["discord_id"],
            username=row["username"],
            balance=row["balance"],
            account_type=row["account_type"],
            private_channel_id=row["private_channel_id"],
            total_profit=row["total_profit"],
            win_rate=row["win_rate"],
            created_at=row["created_at"],
        )

    except aiosqlite.Error as exc:
        logger.exception("Failed to fetch trader %s.", discord_id)
        raise DatabaseError(f"get_trader failed: {exc}") from exc


async def update_balance(
    discord_id: int,
    delta: float,
    db_path: str = DB_PATH,
) -> float:
    """
    Adjust a trader's balance by `delta` (positive to credit, negative to debit).
    This is done atomically via a single UPDATE statement to avoid
    read-modify-write races between concurrent commands.

    Args:
        discord_id: Unique Discord user ID.
        delta: Amount to add to the current balance (use a negative value
            to subtract).
        db_path: Path to the SQLite database file.

    Returns:
        The trader's new balance after the update.

    Raises:
        TraderNotFoundError: If discord_id does not exist.
        DatabaseError: On any other database failure.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "UPDATE traders SET balance = balance + ? WHERE discord_id = ?",
                (delta, discord_id),
            )
            await db.commit()

            if cursor.rowcount == 0:
                raise TraderNotFoundError(
                    f"No trader found with discord_id={discord_id}."
                )

            async with db.execute(
                "SELECT balance FROM traders WHERE discord_id = ?", (discord_id,)
            ) as result_cursor:
                row = await result_cursor.fetchone()

        return float(row[0])

    except TraderNotFoundError:
        raise
    except aiosqlite.Error as exc:
        logger.exception("Failed to update balance for trader %s.", discord_id)
        raise DatabaseError(f"update_balance failed: {exc}") from exc


async def update_trader_stats(
    discord_id: int,
    total_profit: Optional[float] = None,
    win_rate: Optional[float] = None,
    db_path: str = DB_PATH,
) -> None:
    """
    Update a trader's performance stats (total_profit and/or win_rate).
    Only supplied (non-None) fields are updated.

    Args:
        discord_id: Unique Discord user ID.
        total_profit: New cumulative profit value, if updating.
        win_rate: New win rate (0-100), if updating.
        db_path: Path to the SQLite database file.

    Raises:
        TraderNotFoundError: If discord_id does not exist.
        DatabaseError: On any other database failure.
        ValueError: If neither field is provided.
    """
    fields: list[str] = []
    values: list[Any] = []

    if total_profit is not None:
        fields.append("total_profit = ?")
        values.append(total_profit)
    if win_rate is not None:
        fields.append("win_rate = ?")
        values.append(win_rate)

    if not fields:
        raise ValueError("At least one of total_profit or win_rate must be provided.")

    values.append(discord_id)

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                f"UPDATE traders SET {', '.join(fields)} WHERE discord_id = ?",
                tuple(values),
            )
            await db.commit()

            if cursor.rowcount == 0:
                raise TraderNotFoundError(
                    f"No trader found with discord_id={discord_id}."
                )

    except TraderNotFoundError:
        raise
    except aiosqlite.Error as exc:
        logger.exception("Failed to update stats for trader %s.", discord_id)
        raise DatabaseError(f"update_trader_stats failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Signal operations
# --------------------------------------------------------------------------- #

async def log_signal(
    pair: str,
    action: str,
    entry_price: float,
    sl: float,
    tp1: float,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    confidence: Optional[float] = None,
    recommended_strategy: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """
    Insert a new trade signal with status ACTIVE.

    Args:
        pair: Currency pair, e.g. 'EURUSD'.
        action: 'BUY' or 'SELL'.
        entry_price: Entry price for the signal.
        sl: Stop-loss price.
        tp1: First take-profit target (required).
        tp2: Second take-profit target (optional).
        tp3: Third take-profit target (optional).
        confidence: Model confidence (0-100) at the time the signal was
            generated, if known. Persisted so anything that later replays
            this signal (e.g. the /makeaccount welcome-room cards) can show
            the real number instead of guessing/hardcoding one.
        recommended_strategy: The strategy label ("Momentum" / "Mean
            Reversion" / "Breakout" / a bias-engine scenario type) attached
            to this signal at generation time, if known.
        db_path: Path to the SQLite database file.

    Returns:
        The auto-generated signal_id of the newly inserted row.

    Raises:
        ValueError: If action is not BUY/SELL.
        DatabaseError: On database failure.
    """
    if action not in (SignalAction.BUY, SignalAction.SELL):
        raise ValueError(f"Invalid action: {action!r}")

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO signals
                    (pair, action, entry_price, sl, tp1, tp2, tp3, status,
                     timestamp, confidence, recommended_strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pair, action, entry_price, sl, tp1, tp2, tp3,
                 SignalStatus.ACTIVE, timestamp, confidence, recommended_strategy),
            )
            await db.commit()
            signal_id = cursor.lastrowid

        logger.info(
            "Logged signal #%s: %s %s @ %s (confidence=%s, strategy=%s)",
            signal_id, action, pair, entry_price, confidence, recommended_strategy,
        )
        return signal_id

    except aiosqlite.Error as exc:
        logger.exception("Failed to log signal for %s.", pair)
        raise DatabaseError(f"log_signal failed: {exc}") from exc


async def update_signal_status(
    signal_id: int,
    status: str,
    db_path: str = DB_PATH,
) -> None:
    """
    Update the status of an existing signal (e.g. ACTIVE -> TP1_HIT -> CLOSED).

    Args:
        signal_id: The signal's primary key.
        status: New status; must be one of the SignalStatus constants.
        db_path: Path to the SQLite database file.

    Raises:
        ValueError: If status is not a recognized value.
        SignalNotFoundError: If signal_id does not exist.
        DatabaseError: On any other database failure.
    """
    valid_statuses = {
        SignalStatus.ACTIVE, SignalStatus.TP1_HIT, SignalStatus.TP2_HIT,
        SignalStatus.TP3_HIT, SignalStatus.CLOSED, SignalStatus.SL_HIT,
    }
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status!r}")

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "UPDATE signals SET status = ? WHERE signal_id = ?",
                (status, signal_id),
            )
            await db.commit()

            if cursor.rowcount == 0:
                raise SignalNotFoundError(f"No signal found with signal_id={signal_id}.")

        logger.info("Signal #%s status updated to %s.", signal_id, status)

    except SignalNotFoundError:
        raise
    except aiosqlite.Error as exc:
        logger.exception("Failed to update status for signal %s.", signal_id)
        raise DatabaseError(f"update_signal_status failed: {exc}") from exc


async def get_active_signals(db_path: str = DB_PATH) -> list[Signal]:
    """
    Fetch all signals currently in a non-terminal state
    (ACTIVE, TP1_HIT, TP2_HIT, or TP3_HIT), ordered by most recent first.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A list of Signal records. Empty list if none are active.

    Raises:
        DatabaseError: On database failure.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM signals
                WHERE status IN ('ACTIVE', 'TP1_HIT', 'TP2_HIT', 'TP3_HIT')
                ORDER BY timestamp DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            Signal(
                signal_id=row["signal_id"],
                pair=row["pair"],
                action=row["action"],
                entry_price=row["entry_price"],
                sl=row["sl"],
                tp1=row["tp1"],
                tp2=row["tp2"],
                tp3=row["tp3"],
                status=row["status"],
                timestamp=row["timestamp"],
                confidence=row["confidence"] if "confidence" in row.keys() else None,
                recommended_strategy=row["recommended_strategy"] if "recommended_strategy" in row.keys() else None,
            )
            for row in rows
        ]

    except aiosqlite.Error as exc:
        logger.exception("Failed to fetch active signals.")
        raise DatabaseError(f"get_active_signals failed: {exc}") from exc


async def get_signal(signal_id: int, db_path: str = DB_PATH) -> Optional[Signal]:
    """
    Fetch a single signal by its ID.

    Args:
        signal_id: The signal's primary key.
        db_path: Path to the SQLite database file.

    Returns:
        A Signal instance, or None if not found.

    Raises:
        DatabaseError: On database failure.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM signals WHERE signal_id = ?", (signal_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        return Signal(
            signal_id=row["signal_id"],
            pair=row["pair"],
            action=row["action"],
            entry_price=row["entry_price"],
            sl=row["sl"],
            tp1=row["tp1"],
            tp2=row["tp2"],
            tp3=row["tp3"],
            status=row["status"],
            timestamp=row["timestamp"],
            confidence=row["confidence"] if "confidence" in row.keys() else None,
            recommended_strategy=row["recommended_strategy"] if "recommended_strategy" in row.keys() else None,
        )

    except aiosqlite.Error as exc:
        logger.exception("Failed to fetch signal %s.", signal_id)
        raise DatabaseError(f"get_signal failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Phase 7: Financial / account balance operations
#
# These build directly on the existing `traders.balance` column (already
# used for private-room lot sizing) and add a `transactions` ledger so
# every credit/debit is auditable. Nothing here touches signals or trader
# stats — it's additive and isolated from Phases 1-6.
# --------------------------------------------------------------------------- #

async def update_user_balance(
    discord_id: int,
    amount: float,
    tx_type: str,
    admin_id: Optional[int] = None,
    note: Optional[str] = None,
    allow_negative: bool = False,
    db_path: str = DB_PATH,
) -> tuple[bool, float, str]:
    """
    Atomically adjust a trader's balance and record the movement in the
    transactions ledger.

    The balance update and the ledger insert are committed together on the
    same connection, and the update itself is a single conditional UPDATE
    (not a read-then-write), so two concurrent admin commands against the
    same user can't race each other into an inconsistent balance or double
    -spend past a negative-balance guard.

    Args:
        discord_id: Target trader's Discord ID.
        amount: Signed change to apply — positive credits the account,
            negative debits it (e.g. pass -50.0 to deduct $50).
        tx_type: One of TransactionType.{DEPOSIT, WITHDRAWAL, ADJUSTMENT,
            SUBSCRIPTION_FEE}.
        admin_id: Discord ID of the staff member performing the action, if
            any (None for automated/system adjustments).
        note: Optional free-text memo, e.g. "Refund for duplicate fee".
        allow_negative: If False (default), the update is rejected when it
            would take the balance below 0.
        db_path: Path to the SQLite database file.

    Returns:
        (success, resulting_balance, message) where:
            success=True  -> resulting_balance is the NEW balance.
            success=False -> resulting_balance is the CURRENT (unchanged)
                              balance (0.0 if the trader doesn't exist),
                              and message explains why the update was
                              rejected.

    Raises:
        ValueError: If tx_type is not a recognized TransactionType or
            amount is zero / not finite.
        DatabaseError: On an underlying SQLite failure (the transaction is
            not committed in this case, so no partial write occurs).
    """
    if tx_type not in VALID_TRANSACTION_TYPES:
        raise ValueError(f"Invalid tx_type: {tx_type!r}")
    if amount == 0 or not math.isfinite(amount):
        raise ValueError(f"amount must be a non-zero, finite number, got {amount!r}")

    try:
        async with aiosqlite.connect(db_path) as db:
            if allow_negative:
                cursor = await db.execute(
                    "UPDATE traders SET balance = balance + ? WHERE discord_id = ?",
                    (amount, discord_id),
                )
            else:
                # The WHERE clause enforces the negative-balance guard as
                # part of the same atomic statement as the write itself.
                cursor = await db.execute(
                    "UPDATE traders SET balance = balance + ? "
                    "WHERE discord_id = ? AND balance + ? >= 0",
                    (amount, discord_id, amount),
                )

            if cursor.rowcount == 0:
                # Disambiguate "no such trader" vs. "would go negative" with
                # a follow-up read purely for the error message.
                async with db.execute(
                    "SELECT balance FROM traders WHERE discord_id = ?", (discord_id,)
                ) as lookup_cursor:
                    row = await lookup_cursor.fetchone()

                if row is None:
                    return False, 0.0, f"No trader account found for discord_id={discord_id}."

                current_balance = float(row[0])
                return False, current_balance, (
                    f"Insufficient balance: current ${current_balance:.2f}, requested "
                    f"change ${amount:.2f} would go negative (pass allow_negative=True "
                    f"to override)."
                )

            async with db.execute(
                "SELECT balance FROM traders WHERE discord_id = ?", (discord_id,)
            ) as cursor2:
                row = await cursor2.fetchone()
            new_balance = float(row[0])

            await db.execute(
                """
                INSERT INTO transactions
                    (discord_id, type, amount, balance_after, admin_id, note, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    discord_id, tx_type, amount, new_balance, admin_id, note,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

        logger.info(
            "Balance update: discord_id=%s type=%s amount=%.2f new_balance=%.2f admin_id=%s",
            discord_id, tx_type, amount, new_balance, admin_id,
        )
        return True, new_balance, "OK"

    except aiosqlite.Error as exc:
        logger.exception("update_user_balance failed for discord_id=%s.", discord_id)
        raise DatabaseError(f"update_user_balance failed: {exc}") from exc


async def get_user_balance(discord_id: int, db_path: str = DB_PATH) -> Optional[dict[str, Any]]:
    """
    Fetch a trader's balance snapshot for display (e.g. in /balance).

    Args:
        discord_id: Unique Discord user ID.
        db_path: Path to the SQLite database file.

    Returns:
        A dict with keys: discord_id, username, balance, currency,
        account_type, total_profit, win_rate, created_at — or None if the
        trader has no account on file.

    Raises:
        DatabaseError: On database failure.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traders WHERE discord_id = ?", (discord_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        return {
            "discord_id": row["discord_id"],
            "username": row["username"],
            "balance": row["balance"],
            "currency": row["currency"],
            "account_type": row["account_type"],
            "total_profit": row["total_profit"],
            "win_rate": row["win_rate"],
            "created_at": row["created_at"],
        }

    except aiosqlite.Error as exc:
        logger.exception("get_user_balance failed for discord_id=%s.", discord_id)
        raise DatabaseError(f"get_user_balance failed: {exc}") from exc


async def get_transaction_history(
    discord_id: int,
    limit: int = 10,
    db_path: str = DB_PATH,
) -> list[dict[str, Any]]:
    """
    Fetch a trader's most recent ledger entries, newest first.

    Args:
        discord_id: Unique Discord user ID.
        limit: Maximum number of rows to return (default 10).
        db_path: Path to the SQLite database file.

    Returns:
        A list of dicts with keys: id, discord_id, type, amount,
        balance_after, admin_id, note, timestamp. Empty list if the trader
        has no transactions (or doesn't exist).

    Raises:
        DatabaseError: On database failure.
    """
    if limit <= 0:
        return []

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT id, discord_id, type, amount, balance_after, admin_id, note, timestamp
                FROM transactions
                WHERE discord_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (discord_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    except aiosqlite.Error as exc:
        logger.exception("get_transaction_history failed for discord_id=%s.", discord_id)
        raise DatabaseError(f"get_transaction_history failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# User preference operations (Phase 10 — /makeaccount modal profile)
# --------------------------------------------------------------------------- #

def _normalize_preferred_pairs(preferred_pairs: "str | list[str]") -> str:
    """
    Accepts either a raw comma-separated string (as typed into the Modal's
    TextInput) or an already-split list, and normalizes it to a clean,
    de-duplicated, comma-separated, uppercase string for storage —
    e.g. " xauusd, gbpusd ,xauusd" -> "XAUUSD, GBPUSD".
    """
    if isinstance(preferred_pairs, str):
        raw_items = preferred_pairs.split(",")
    else:
        raw_items = list(preferred_pairs)

    cleaned = [item.strip().upper() for item in raw_items if item and item.strip()]
    deduped = list(dict.fromkeys(cleaned))  # order-preserved de-dup
    return ", ".join(deduped)


async def save_user_preferences(
    user_id: int,
    broker_server: str,
    preferred_pairs: "str | list[str]",
    country: str,
    starting_balance: float = 1000.0,
    db_path: str = DB_PATH,
) -> UserPreferences:
    """
    Insert or update (upsert) a trader's saved profile from the
    `/makeaccount` modal.

    Uses INSERT ... ON CONFLICT(user_id) DO UPDATE, so re-submitting the
    modal (e.g. a future "update my profile" flow) safely overwrites the
    existing row instead of raising a primary-key conflict.

    Args:
        user_id: Discord user ID (matches traders.discord_id).
        broker_server: Free-text broker name & server, e.g. "Exness-Real10".
        preferred_pairs: Comma-separated string or list of symbols, e.g.
            "XAUUSD, GBPUSD" or ["XAUUSD", "GBPUSD"]. Normalized to a
            de-duplicated, uppercase, comma-separated string for storage.
        country: Free-text country/region, e.g. "Malaysia".
        starting_balance: Starting demo balance in USD (default 1000.0).
        db_path: Path to the SQLite database file.

    Returns:
        The saved UserPreferences record.

    Raises:
        ValueError: If broker_server/country are blank, or starting_balance
            is not positive.
        DatabaseError: On any other database failure.
    """
    broker_server = (broker_server or "").strip()
    country = (country or "").strip()
    if not broker_server:
        raise ValueError("broker_server cannot be blank.")
    if not country:
        raise ValueError("country cannot be blank.")
    if starting_balance <= 0:
        raise ValueError(f"starting_balance must be > 0, got {starting_balance!r}.")

    normalized_pairs = _normalize_preferred_pairs(preferred_pairs)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT INTO user_preferences
                    (user_id, broker_server, preferred_pairs, country,
                     starting_balance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    broker_server    = excluded.broker_server,
                    preferred_pairs  = excluded.preferred_pairs,
                    country          = excluded.country,
                    starting_balance = excluded.starting_balance
                """,
                (user_id, broker_server, normalized_pairs, country,
                 starting_balance, created_at),
            )
            await db.commit()

        return UserPreferences(
            user_id=user_id,
            broker_server=broker_server,
            preferred_pairs=[p.strip() for p in normalized_pairs.split(",") if p.strip()],
            country=country,
            starting_balance=starting_balance,
            created_at=created_at,
        )

    except aiosqlite.Error as exc:
        logger.exception("save_user_preferences failed for user_id=%s.", user_id)
        raise DatabaseError(f"save_user_preferences failed: {exc}") from exc


async def get_traders_for_news_delivery(db_path: str = DB_PATH) -> list[dict[str, Any]]:
    """
    Fetch every trader that has both a private signal room and a saved
    `/makeaccount` profile, for routing personalized news explanations to
    their room based on their preferred pairs.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A list of dicts with keys: discord_id, private_channel_id,
        preferred_pairs (list[str]). Empty list if none qualify or on
        failure (failures are logged, not raised — news delivery should
        degrade gracefully, not break the news loop).
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT t.discord_id AS discord_id,
                       t.private_channel_id AS private_channel_id,
                       p.preferred_pairs AS preferred_pairs
                FROM traders t
                JOIN user_preferences p ON p.user_id = t.discord_id
                WHERE t.private_channel_id IS NOT NULL
                """
            ) as cursor:
                rows = await cursor.fetchall()

        results = []
        for row in rows:
            pairs = [p.strip() for p in (row["preferred_pairs"] or "").split(",") if p.strip()]
            results.append({
                "discord_id": row["discord_id"],
                "private_channel_id": row["private_channel_id"],
                "preferred_pairs": pairs,
            })
        return results

    except aiosqlite.Error as exc:
        logger.exception("get_traders_for_news_delivery failed.")
        return []


async def get_user_preferences(user_id: int, db_path: str = DB_PATH) -> Optional[UserPreferences]:
    """
    Fetch a trader's saved `/makeaccount` profile.

    Args:
        user_id: Discord user ID (matches traders.discord_id).
        db_path: Path to the SQLite database file.

    Returns:
        A UserPreferences instance (with preferred_pairs already split into
        a clean list[str]), or None if the user has no saved profile.

    Raises:
        DatabaseError: On database failure.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        pairs = [p.strip() for p in (row["preferred_pairs"] or "").split(",") if p.strip()]
        return UserPreferences(
            user_id=row["user_id"],
            broker_server=row["broker_server"],
            preferred_pairs=pairs,
            country=row["country"],
            starting_balance=row["starting_balance"],
            created_at=row["created_at"],
        )

    except aiosqlite.Error as exc:
        logger.exception("get_user_preferences failed for user_id=%s.", user_id)
        raise DatabaseError(f"get_user_preferences failed: {exc}") from exc