"""
structure_engine.py
====================
Institutional Market Structure Engine for XAUUSD and major Forex pairs.

Implements fractal-based swing detection, Break of Structure (BOS) /
Change of Character (CHOCH) classification, trend-bias scoring, and
Premium/Discount (equilibrium) zone mapping directly off MT5 candle
rates (`mt5.copy_rates_from_pos`).

This module is fully standalone and self-contained: it does not import
from, modify, or otherwise touch any other project file (main.py,
mt5_engine.py, ai_engine.py, news_engine.py, chart_engine.py,
database.py). It talks to the MT5 terminal API directly, exactly like
mt5_engine.py does, and assumes the terminal has already been
initialized/logged in elsewhere in the app (e.g. via
mt5_engine.init_mt5() at bot startup). If MT5 is not initialized, the
public functions fail safe and return default/neutral structures
instead of raising.

Dependencies:
    pip install MetaTrader5 pandas numpy

Public API:
    determine_market_structure(symbol, timeframe, count=200) -> dict
    async analyze_xauusd_structure(timeframes=[...]) -> dict
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as e:
    raise ImportError(
        "MetaTrader5 package not found. Install with: pip install MetaTrader5 "
        "(Windows only — the official package requires a running MT5 terminal)."
    ) from e

logger = logging.getLogger("structure_engine")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# 5-bar fractal by default: 2 bars of confirmation on each side of the pivot.
FRACTAL_WINDOW = 2

# Distance from equilibrium (as a fraction of the swing range) that still
# counts as "EQUILIBRIUM" rather than PREMIUM/DISCOUNT.
EQUILIBRIUM_TOLERANCE_PCT = 0.05

# How many trailing same-direction BOS events (since the last CHOCH) are
# needed to upgrade BULLISH/BEARISH to STRONG_BULLISH/STRONG_BEARISH.
STRONG_TREND_BOS_THRESHOLD = 2

DEFAULT_TIMEFRAMES_ATTR = ("TIMEFRAME_D1", "TIMEFRAME_H4", "TIMEFRAME_H1", "TIMEFRAME_M15")

_TIMEFRAME_NAME_MAP = {
    getattr(mt5, attr): attr.replace("TIMEFRAME_", "")
    for attr in dir(mt5)
    if attr.startswith("TIMEFRAME_")
}


def _timeframe_name(timeframe: int) -> str:
    return _TIMEFRAME_NAME_MAP.get(timeframe, str(timeframe))


# ---------------------------------------------------------------------------
# Safe defaults (returned whenever data is unavailable / an error occurs)
# ---------------------------------------------------------------------------

def _default_structure(symbol: str, timeframe: int, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": _timeframe_name(timeframe),
        "trend_bias": "NEUTRAL",
        "last_bos": None,
        "last_choch": None,
        "recent_swing_high": None,
        "recent_swing_low": None,
        "equilibrium_price": None,
        "zone": "UNKNOWN",
        "current_price": None,
        "candles_analyzed": 0,
        "error": error,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 1. Rate fetching
# ---------------------------------------------------------------------------

def _fetch_rates_df(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    Pull `count` candles for `symbol`/`timeframe` via
    mt5.copy_rates_from_pos and return a clean OHLC DataFrame, or None
    if unavailable. Never raises.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(
                "structure_engine: no rates for %s (%s): %s",
                symbol, _timeframe_name(timeframe), mt5.last_error(),
            )
            return None

        df = pd.DataFrame(rates)
        if not {"time", "open", "high", "low", "close"}.issubset(df.columns):
            logger.error("structure_engine: unexpected rates schema for %s.", symbol)
            return None

        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close"]].reset_index(drop=True)
        return df

    except Exception:
        logger.exception("structure_engine: exception fetching rates for %s.", symbol)
        return None


# ---------------------------------------------------------------------------
# 2. Fractal swing high / swing low detection
# ---------------------------------------------------------------------------

def _detect_swing_points(df: pd.DataFrame, window: int = FRACTAL_WINDOW) -> List[Dict[str, Any]]:
    """
    N-bar fractal swing detection. A bar at index i is a swing HIGH if its
    high is strictly greater than the highs of `window` bars on both
    sides; a swing LOW mirrors this on the lows. Returns points in
    chronological order.
    """
    swings: List[Dict[str, Any]] = []
    n = len(df)
    if n < (2 * window + 1):
        return swings

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"].to_numpy()

    for i in range(window, n - window):
        left_highs = highs[i - window:i]
        right_highs = highs[i + 1:i + 1 + window]
        left_lows = lows[i - window:i]
        right_lows = lows[i + 1:i + 1 + window]

        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swings.append({
                "type": "HIGH",
                "index": i,
                "price": float(highs[i]),
                "time": pd.Timestamp(times[i]).to_pydatetime().isoformat(),
            })

        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swings.append({
                "type": "LOW",
                "index": i,
                "price": float(lows[i]),
                "time": pd.Timestamp(times[i]).to_pydatetime().isoformat(),
            })

    swings.sort(key=lambda s: s["index"])
    return swings


# ---------------------------------------------------------------------------
# 3. BOS / CHOCH structural walk
# ---------------------------------------------------------------------------

def _walk_structure(df: pd.DataFrame, swings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Walk forward through the candles, tracking the most recent unbroken
    swing high/low, and classify each break of a reference swing point
    as BOS (continuation) or CHOCH (reversal) based on the prevailing
    trend at the time of the break.
    """
    swing_highs = [s for s in swings if s["type"] == "HIGH"]
    swing_lows = [s for s in swings if s["type"] == "LOW"]

    events: List[Dict[str, Any]] = []
    trend: Optional[str] = None
    trailing_bos_count = 0

    active_high: Optional[Dict[str, Any]] = None
    active_low: Optional[Dict[str, Any]] = None
    sh_idx = 0
    sl_idx = 0

    closes = df["close"].to_numpy()
    times = df["time"].to_numpy()
    n = len(df)

    for i in range(n):
        while sh_idx < len(swing_highs) and swing_highs[sh_idx]["index"] < i:
            active_high = swing_highs[sh_idx]
            sh_idx += 1
        while sl_idx < len(swing_lows) and swing_lows[sl_idx]["index"] < i:
            active_low = swing_lows[sl_idx]
            sl_idx += 1

        close = closes[i]
        event_time = pd.Timestamp(times[i]).to_pydatetime().isoformat()

        if active_high is not None and close > active_high["price"]:
            is_choch = trend in (None, "BEARISH")
            events.append({
                "kind": "CHOCH" if is_choch else "BOS",
                "direction": "BULLISH",
                "price": float(active_high["price"]),
                "break_price": float(close),
                "timestamp": event_time,
            })
            trend = "BULLISH"
            trailing_bos_count = 0 if is_choch else trailing_bos_count + 1
            active_high = None  # consumed; wait for a fresh swing high to form

        elif active_low is not None and close < active_low["price"]:
            is_choch = trend in (None, "BULLISH")
            events.append({
                "kind": "CHOCH" if is_choch else "BOS",
                "direction": "BEARISH",
                "price": float(active_low["price"]),
                "break_price": float(close),
                "timestamp": event_time,
            })
            trend = "BEARISH"
            trailing_bos_count = 0 if is_choch else trailing_bos_count + 1
            active_low = None  # consumed; wait for a fresh swing low to form

    last_bos = next((e for e in reversed(events) if e["kind"] == "BOS"), None)
    last_choch = next((e for e in reversed(events) if e["kind"] == "CHOCH"), None)

    return {
        "trend": trend,
        "trailing_bos_count": trailing_bos_count,
        "last_bos": _strip_kind(last_bos),
        "last_choch": _strip_kind(last_choch),
    }


def _strip_kind(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if event is None:
        return None
    return {
        "price": event["price"],
        "timestamp": event["timestamp"],
        "direction": event["direction"],
    }


def _classify_trend_bias(trend: Optional[str], trailing_bos_count: int) -> str:
    if trend == "BULLISH":
        return "STRONG_BULLISH" if trailing_bos_count >= STRONG_TREND_BOS_THRESHOLD else "BULLISH"
    if trend == "BEARISH":
        return "STRONG_BEARISH" if trailing_bos_count >= STRONG_TREND_BOS_THRESHOLD else "BEARISH"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# 4. Premium / Discount (equilibrium) zone
# ---------------------------------------------------------------------------

def _classify_zone(current_price: float, swing_high: float, swing_low: float) -> Dict[str, Any]:
    equilibrium = (swing_high + swing_low) / 2.0
    price_range = swing_high - swing_low

    if price_range <= 0:
        return {"equilibrium_price": equilibrium, "zone": "UNKNOWN"}

    tolerance = price_range * EQUILIBRIUM_TOLERANCE_PCT
    delta = current_price - equilibrium

    if abs(delta) <= tolerance:
        zone = "EQUILIBRIUM"
    elif delta > 0:
        zone = "PREMIUM"
    else:
        zone = "DISCOUNT"

    return {"equilibrium_price": round(equilibrium, 5), "zone": zone}


# ---------------------------------------------------------------------------
# 5. Public API — single timeframe
# ---------------------------------------------------------------------------

def determine_market_structure(symbol: str, timeframe: int, count: int = 200) -> Dict[str, Any]:
    """
    Analyze `symbol` on `timeframe` (an mt5.TIMEFRAME_* constant) over the
    last `count` candles and return a dict with:
        trend_bias, last_bos, last_choch, recent_swing_high,
        recent_swing_low, equilibrium_price, zone, current_price.

    Always returns a well-formed dict (never raises); on any failure a
    safe NEUTRAL/None default structure is returned with an "error" key
    describing what went wrong.
    """
    try:
        df = _fetch_rates_df(symbol, timeframe, count)
        if df is None or df.empty:
            return _default_structure(symbol, timeframe, error="No rates available.")

        swings = _detect_swing_points(df)
        walk = _walk_structure(df, swings)

        swing_highs = [s for s in swings if s["type"] == "HIGH"]
        swing_lows = [s for s in swings if s["type"] == "LOW"]

        recent_swing_high = swing_highs[-1]["price"] if swing_highs else float(df["high"].max())
        recent_swing_low = swing_lows[-1]["price"] if swing_lows else float(df["low"].min())
        current_price = float(df["close"].iat[-1])

        zone_info = _classify_zone(current_price, recent_swing_high, recent_swing_low)
        trend_bias = _classify_trend_bias(walk["trend"], walk["trailing_bos_count"])

        result = _default_structure(symbol, timeframe)
        result.update({
            "trend_bias": trend_bias,
            "last_bos": walk["last_bos"],
            "last_choch": walk["last_choch"],
            "recent_swing_high": round(recent_swing_high, 5),
            "recent_swing_low": round(recent_swing_low, 5),
            "current_price": round(current_price, 5),
            "candles_analyzed": len(df),
            "error": None,
            **zone_info,
        })
        return result

    except Exception as e:
        logger.exception("determine_market_structure failed for %s.", symbol)
        return _default_structure(symbol, timeframe, error=str(e))


# ---------------------------------------------------------------------------
# 6. Public API — async multi-timeframe entrypoint
# ---------------------------------------------------------------------------

async def analyze_xauusd_structure(
    timeframes: Optional[List[int]] = None,
    symbol: str = "XAUUSD",
    count: int = 200,
) -> Dict[str, Any]:
    """
    Run determine_market_structure() across multiple timeframes for
    `symbol` (default XAUUSD), off the event loop, and return a combined
    dict keyed by timeframe name plus a simple multi-timeframe confluence
    summary. Never raises — per-timeframe failures degrade to safe
    defaults rather than aborting the whole call.
    """
    if timeframes is None:
        timeframes = [mt5.TIMEFRAME_D1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_M15]

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, determine_market_structure, symbol, tf, count)
        for tf in timeframes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    by_timeframe: Dict[str, Any] = {}
    for tf, res in zip(timeframes, results):
        name = _timeframe_name(tf)
        if isinstance(res, Exception):
            logger.exception("analyze_xauusd_structure: timeframe %s failed.", name, exc_info=res)
            by_timeframe[name] = _default_structure(symbol, tf, error=str(res))
        else:
            by_timeframe[name] = res

    biases = [v["trend_bias"] for v in by_timeframe.values() if v.get("trend_bias")]
    bullish = sum(1 for b in biases if "BULLISH" in b)
    bearish = sum(1 for b in biases if "BEARISH" in b)

    if bullish > bearish:
        confluence = "BULLISH"
    elif bearish > bullish:
        confluence = "BEARISH"
    else:
        confluence = "MIXED"

    return {
        "symbol": symbol,
        "timeframes": by_timeframe,
        "confluence_bias": confluence,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Standalone test entrypoint — python structure_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def _run() -> None:
        if not mt5.initialize():
            print(f"mt5.initialize() failed: {mt5.last_error()}")
            return

        try:
            print("=== Single-timeframe test: XAUUSD H4 ===")
            single = determine_market_structure("XAUUSD", mt5.TIMEFRAME_H4, count=200)
            print(json.dumps(single, indent=2, default=str))

            print("\n=== Multi-timeframe test: XAUUSD (D1/H4/H1/M15) ===")
            multi = asyncio.run(analyze_xauusd_structure(symbol="XAUUSD"))
            print(json.dumps(multi, indent=2, default=str))
        finally:
            mt5.shutdown()

    _run()
