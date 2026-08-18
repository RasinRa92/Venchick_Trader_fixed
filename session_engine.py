"""
session_engine.py
==================
Session & Daily Levels Engine for XAUUSD and major Forex pairs.
Complements structure_engine.py (BOS/CHOCH/trend bias) and smc_engine.py
(liquidity/FVG/order blocks) by tracking the "when" and "where" of
institutional reference points:

    - Global session tracking (Asian / London / New York / Overlap), UTC
    - Previous Day/Week/Month High & Low (PDH/PDL, PWH/PWL, PMH/PML)
    - Asian Session High & Low (ASH / ASL)
    - Distance in pips from live price to every reference level
    - Asian range sweep check during London/NY hours (ICT "Judas Swing")

This module is fully standalone and self-contained: it does not import
from, modify, or otherwise touch any other project file (main.py,
mt5_engine.py, ai_engine.py, news_engine.py, chart_engine.py,
database.py, structure_engine.py, smc_engine.py). It talks to the MT5
terminal API directly, exactly like mt5_engine.py, structure_engine.py,
and smc_engine.py do, and assumes the terminal has already been
initialized/logged in elsewhere in the app (e.g. via
mt5_engine.init_mt5() at bot startup).

Timezone handling: all session math runs on timezone-aware UTC
datetimes (zoneinfo.ZoneInfo("UTC"), falling back to datetime.timezone.utc
if the local tzdata database is unavailable — e.g. a bare Windows box
without the `tzdata` package installed).

Fallback behavior: session tracking (current_utc_time, active_sessions,
primary_session, time_remaining_in_session) is pure Python/time-based
and always works, even with MT5 completely offline. Only the MT5-backed
fields (levels, level_distances, asian_range_swept) degrade to safe
None/False defaults — with an "error" key set — if MT5 is unavailable.

Dependencies:
    pip install MetaTrader5 pandas

Public API:
    determine_session_context(symbol) -> dict
    async get_xauusd_session_context(symbol="XAUUSD") -> dict
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
    UTC = ZoneInfo("UTC")
except Exception:  # pragma: no cover - only hit on tzdata-less environments
    UTC = timezone.utc

try:
    import MetaTrader5 as mt5
except ImportError as e:
    raise ImportError(
        "MetaTrader5 package not found. Install with: pip install MetaTrader5 "
        "(Windows only — the official package requires a running MT5 terminal)."
    ) from e

logger = logging.getLogger("session_engine")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# (start_hour_utc, end_hour_utc) — half-open windows, no wraparound needed
# for these four (all defined within a single UTC calendar day).
SESSIONS: Dict[str, Tuple[int, int]] = {
    "Asian": (0, 9),
    "London": (7, 16),
    "New York": (12, 21),
    "London/New York Overlap": (12, 16),
}

# Highest-liquidity session wins as "primary" when more than one is open.
SESSION_PRIORITY: List[str] = [
    "London/New York Overlap",
    "New York",
    "London",
    "Asian",
]

NO_SESSION_LABEL = "No Major Session (Off-Hours)"

# Sessions during which an Asian-range sweep counts as a classic
# ICT "Judas Swing" (a raid of the Asian high/low during the London or
# New York killzones).
LONDON_NY_SESSIONS = ("London", "New York", "London/New York Overlap")

LEVEL_KEYS = ["PDH", "PDL", "PWH", "PWL", "PMH", "PML", "ASH", "ASL"]

# Bars of M15 data to pull for the Asian-session range / post-session
# sweep check. 200 M15 bars ≈ ~2 trading days, comfortably covering
# today's full Asian window plus the current session so far.
INTRADAY_LOOKBACK_COUNT = 200


# ---------------------------------------------------------------------------
# 1. Session tracking (pure time math — no MT5 dependency)
# ---------------------------------------------------------------------------

def _is_session_active(start_hour: int, end_hour: int, now_utc: datetime) -> bool:
    t = now_utc.time()
    start = datetime.min.time().replace(hour=start_hour % 24)
    end = datetime.min.time().replace(hour=end_hour % 24)
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # wraparound window (not used by defaults, kept for safety)


def _get_active_sessions(now_utc: datetime) -> List[str]:
    return [name for name, (start, end) in SESSIONS.items() if _is_session_active(start, end, now_utc)]


def _pick_primary_session(active_sessions: List[str]) -> str:
    for name in SESSION_PRIORITY:
        if name in active_sessions:
            return name
    return NO_SESSION_LABEL


def _time_remaining_in_session(primary_session: str, now_utc: datetime) -> Optional[str]:
    if primary_session not in SESSIONS:
        return None

    _, end_hour = SESSIONS[primary_session]
    end_dt = now_utc.replace(hour=end_hour % 24, minute=0, second=0, microsecond=0)
    if end_dt <= now_utc:
        end_dt += timedelta(days=1)

    remaining_minutes = int((end_dt - now_utc).total_seconds() // 60)
    hours, minutes = divmod(remaining_minutes, 60)
    return f"{hours}h {minutes}m"


# ---------------------------------------------------------------------------
# 2. Safe defaults (session fields always populate; MT5 fields degrade)
# ---------------------------------------------------------------------------

def _default_session_context(symbol: str, error: Optional[str] = None) -> Dict[str, Any]:
    now_utc = datetime.now(UTC)
    active_sessions = _get_active_sessions(now_utc)
    primary_session = _pick_primary_session(active_sessions)

    return {
        "symbol": symbol,
        "current_utc_time": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "active_sessions": active_sessions,
        "primary_session": primary_session,
        "time_remaining_in_session": _time_remaining_in_session(primary_session, now_utc),
        "current_price": None,
        "levels": {k: None for k in LEVEL_KEYS},
        "level_distances": {k: None for k in LEVEL_KEYS},
        "asian_range_swept": {"ash_swept": False, "asl_swept": False},
        "error": error,
        "generated_at": now_utc.isoformat(),
    }


# ---------------------------------------------------------------------------
# 3. MT5 helpers — price, pip size, prior-period high/low, Asian range
# ---------------------------------------------------------------------------

def _get_current_price(symbol: str) -> Optional[float]:
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None and tick.bid:
            return float(tick.bid)

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates is not None and len(rates) > 0:
            return float(rates[0]["close"])
        return None
    except Exception:
        logger.exception("session_engine: failed to read live price for %s.", symbol)
        return None


def _get_pip_size(symbol: str) -> Optional[float]:
    """
    Same convention as mt5_engine.get_live_spread: a "pip" is 10x the
    point for 5/3-digit symbols (most FX pairs, Gold on many brokers),
    1x the point for 4/2-digit symbols.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return info.point * 10 if info.digits in (3, 5) else info.point
    except Exception:
        logger.exception("session_engine: failed to resolve pip size for %s.", symbol)
        return None


def _fetch_prev_period_high_low(symbol: str, timeframe: int) -> Optional[Tuple[float, float]]:
    """
    Fetch the most recently *completed* candle at `timeframe` (start
    position 1, skipping the still-forming current-period candle) and
    return (high, low).
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        r = rates[0]
        return float(r["high"]), float(r["low"])
    except Exception:
        logger.exception(
            "session_engine: failed to fetch prior-period high/low for %s (%s).", symbol, timeframe
        )
        return None


def _fetch_intraday_df(symbol: str, count: int = INTRADAY_LOOKBACK_COUNT) -> Optional[pd.DataFrame]:
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        if not {"time", "high", "low"}.issubset(df.columns):
            return None
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df
    except Exception:
        logger.exception("session_engine: failed to fetch intraday M15 data for %s.", symbol)
        return None


def _asian_session_range(intraday_df: Optional[pd.DataFrame], now_utc: datetime) -> Optional[Tuple[float, float]]:
    """
    High/low of today's UTC Asian window (00:00–09:00). If the Asian
    session is still in progress, this is the partial range so far.
    """
    if intraday_df is None or intraday_df.empty:
        return None

    today = now_utc.date()
    mask = (intraday_df["time"].dt.date == today) & (intraday_df["time"].dt.hour < SESSIONS["Asian"][1])
    session_df = intraday_df[mask]
    if session_df.empty:
        return None

    return float(session_df["high"].max()), float(session_df["low"].min())


def _check_asian_sweep(
    intraday_df: Optional[pd.DataFrame],
    ash: Optional[float],
    asl: Optional[float],
    active_sessions: List[str],
    now_utc: datetime,
) -> Dict[str, bool]:
    """
    ICT "Judas Swing" check: has price, at any point since today's Asian
    session closed (09:00 UTC) while London and/or New York is open,
    traded beyond the Asian High (ash_swept) or Asian Low (asl_swept)?
    Using the full post-session high/low (not just the current tick)
    catches a wick-and-reverse raid even if price has since snapped
    back inside the range.
    """
    if ash is None or asl is None:
        return {"ash_swept": False, "asl_swept": False}
    if not any(s in active_sessions for s in LONDON_NY_SESSIONS):
        return {"ash_swept": False, "asl_swept": False}
    if intraday_df is None or intraday_df.empty:
        return {"ash_swept": False, "asl_swept": False}

    today = now_utc.date()
    asian_end_hour = SESSIONS["Asian"][1]
    post_session_mask = (intraday_df["time"].dt.date == today) & (intraday_df["time"].dt.hour >= asian_end_hour)
    post_session_df = intraday_df[post_session_mask]

    if post_session_df.empty:
        return {"ash_swept": False, "asl_swept": False}

    return {
        "ash_swept": bool((post_session_df["high"] > ash).any()),
        "asl_swept": bool((post_session_df["low"] < asl).any()),
    }


def _pip_distance(level: Optional[float], price: Optional[float], pip_size: Optional[float]) -> Optional[float]:
    """
    Signed distance in pips from `price` to `level`: positive means the
    level sits above current price, negative means below.
    """
    if level is None or price is None or not pip_size:
        return None
    return round((level - price) / pip_size, 1)


# ---------------------------------------------------------------------------
# 4. Public API — single symbol
# ---------------------------------------------------------------------------

def determine_session_context(symbol: str) -> Dict[str, Any]:
    """
    Build the full session + institutional-levels context for `symbol`:
        current_utc_time, active_sessions, primary_session,
        time_remaining_in_session, levels (PDH/PDL/PWH/PWL/PMH/PML/ASH/ASL),
        level_distances (pips), asian_range_swept.

    Session fields always populate (pure UTC time math). MT5-backed
    fields degrade to safe None/False defaults with an "error" key set
    if MT5 rates are unavailable — this function never raises.
    """
    now_utc = datetime.now(UTC)
    active_sessions = _get_active_sessions(now_utc)
    primary_session = _pick_primary_session(active_sessions)
    time_remaining = _time_remaining_in_session(primary_session, now_utc)

    context = _default_session_context(symbol)
    context.update({
        "current_utc_time": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "active_sessions": active_sessions,
        "primary_session": primary_session,
        "time_remaining_in_session": time_remaining,
    })

    try:
        current_price = _get_current_price(symbol)
        pip_size = _get_pip_size(symbol)

        pdh_pdl = _fetch_prev_period_high_low(symbol, mt5.TIMEFRAME_D1)
        pwh_pwl = _fetch_prev_period_high_low(symbol, mt5.TIMEFRAME_W1)
        pmh_pml = _fetch_prev_period_high_low(symbol, mt5.TIMEFRAME_MN1)
        intraday_df = _fetch_intraday_df(symbol)
        asian_range = _asian_session_range(intraday_df, now_utc)

        levels = {
            "PDH": pdh_pdl[0] if pdh_pdl else None,
            "PDL": pdh_pdl[1] if pdh_pdl else None,
            "PWH": pwh_pwl[0] if pwh_pwl else None,
            "PWL": pwh_pwl[1] if pwh_pwl else None,
            "PMH": pmh_pml[0] if pmh_pml else None,
            "PML": pmh_pml[1] if pmh_pml else None,
            "ASH": asian_range[0] if asian_range else None,
            "ASL": asian_range[1] if asian_range else None,
        }
        levels = {k: (round(v, 5) if v is not None else None) for k, v in levels.items()}

        level_distances = {k: _pip_distance(v, current_price, pip_size) for k, v in levels.items()}

        asian_range_swept = _check_asian_sweep(
            intraday_df, levels["ASH"], levels["ASL"], active_sessions, now_utc
        )

        context.update({
            "current_price": round(current_price, 5) if current_price is not None else None,
            "levels": levels,
            "level_distances": level_distances,
            "asian_range_swept": asian_range_swept,
            "error": None,
        })
        return context

    except Exception as e:
        logger.exception("determine_session_context failed for %s.", symbol)
        context["error"] = str(e)
        return context


# ---------------------------------------------------------------------------
# 5. Public API — async entrypoint
# ---------------------------------------------------------------------------

async def get_xauusd_session_context(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Async entrypoint. Runs the (blocking) MT5 lookups off the event loop
    and returns:
        current_utc_time, active_sessions, primary_session,
        time_remaining_in_session, levels, level_distances,
        asian_range_swept.

    Never raises — any failure degrades to a safe default structure
    (session fields still populate; MT5 fields fall back to
    None/False) with an "error" key set.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, determine_session_context, symbol)
    except Exception as e:
        logger.exception("get_xauusd_session_context failed for %s.", symbol)
        return _default_session_context(symbol, error=str(e))


# ---------------------------------------------------------------------------
# Standalone test entrypoint — python session_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def _run() -> None:
        if not mt5.initialize():
            print(f"mt5.initialize() failed: {mt5.last_error()} — showing session-only (offline) context.")
            print(json.dumps(_default_session_context("XAUUSD", error="MT5 not initialized"), indent=2, default=str))
            return

        try:
            print("=== Session context test: XAUUSD ===")
            result = asyncio.run(get_xauusd_session_context(symbol="XAUUSD"))
            print(json.dumps(result, indent=2, default=str))
        finally:
            mt5.shutdown()

    _run()
