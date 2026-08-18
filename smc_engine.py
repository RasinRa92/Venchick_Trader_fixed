"""
smc_engine.py
=============
Liquidity & Smart Money Concepts (SMC) Engine for XAUUSD and major Forex
pairs. Complements structure_engine.py (Break of Structure / CHOCH /
trend bias) by mapping the liquidity and imbalance side of price action:

    - Buy-Side / Sell-Side Liquidity pools (BSL / SSL) from swing points
    - Equal Highs / Equal Lows (EQH / EQL) clusters
    - Liquidity sweep (stop-hunt / raid) detection
    - Fair Value Gaps (FVG) — active vs. mitigated
    - Order Blocks (OB) — active vs. mitigated

This module is fully standalone and self-contained: it does not import
from, modify, or otherwise touch any other project file (main.py,
mt5_engine.py, ai_engine.py, news_engine.py, chart_engine.py,
database.py, structure_engine.py). It talks to the MT5 terminal API
directly, exactly like mt5_engine.py and structure_engine.py do, and
assumes the terminal has already been initialized/logged in elsewhere
in the app (e.g. via mt5_engine.init_mt5() at bot startup). If MT5 is
not initialized, the public functions fail safe and return empty
default structures instead of raising.

Dependencies:
    pip install MetaTrader5 pandas numpy

Public API:
    determine_smc_structure(symbol, timeframe, count=200) -> dict
    async analyze_xauusd_smc(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, count=200) -> dict
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

logger = logging.getLogger("smc_engine")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# 5-bar fractal by default: 2 bars of confirmation on each side of the pivot.
FRACTAL_WINDOW = 2

# Two swing highs/lows within this % of each other are considered "equal"
# (EQH / EQL) — a resting liquidity cluster. ~0.05% ≈ ~$1 on $2000 gold;
# widen this per-symbol if you want the classic "~50 pip" Gold reading.
EQUAL_LEVEL_TOLERANCE_PCT = 0.0005

# How many bars back from an impulsive break/FVG to search for the
# opposite-colored candle that forms the Order Block.
ORDER_BLOCK_LOOKBACK = 10

# Only surface sweeps that happened within this many trailing bars as
# "recent" (older sweeps are still valid history but not "recent").
RECENT_SWEEP_LOOKBACK = 50

_TIMEFRAME_NAME_MAP = {
    getattr(mt5, attr): attr.replace("TIMEFRAME_", "")
    for attr in dir(mt5)
    if attr.startswith("TIMEFRAME_")
}


def _timeframe_name(timeframe: int) -> str:
    return _TIMEFRAME_NAME_MAP.get(timeframe, str(timeframe))


# ---------------------------------------------------------------------------
# Safe defaults
# ---------------------------------------------------------------------------

def _default_smc(symbol: str, timeframe: int, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": _timeframe_name(timeframe),
        "active_bsl": [],
        "active_ssl": [],
        "equal_highs": [],
        "equal_lows": [],
        "recent_sweeps": [],
        "unfilled_fvgs": [],
        "order_blocks": [],
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
                "smc_engine: no rates for %s (%s): %s",
                symbol, _timeframe_name(timeframe), mt5.last_error(),
            )
            return None

        df = pd.DataFrame(rates)
        if not {"time", "open", "high", "low", "close"}.issubset(df.columns):
            logger.error("smc_engine: unexpected rates schema for %s.", symbol)
            return None

        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close"]].reset_index(drop=True)
        return df

    except Exception:
        logger.exception("smc_engine: exception fetching rates for %s.", symbol)
        return None


def _iso_time(df: pd.DataFrame, index: int) -> str:
    return pd.Timestamp(df["time"].iat[index]).to_pydatetime().isoformat()


# ---------------------------------------------------------------------------
# 2. Fractal swing high / swing low detection (shared basis for BSL/SSL,
#    EQH/EQL, and Order Block context)
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

    for i in range(window, n - window):
        left_highs = highs[i - window:i]
        right_highs = highs[i + 1:i + 1 + window]
        left_lows = lows[i - window:i]
        right_lows = lows[i + 1:i + 1 + window]

        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swings.append({"type": "HIGH", "index": i, "price": float(highs[i]), "time": _iso_time(df, i)})

        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swings.append({"type": "LOW", "index": i, "price": float(lows[i]), "time": _iso_time(df, i)})

    swings.sort(key=lambda s: s["index"])
    return swings


# ---------------------------------------------------------------------------
# 3. Liquidity pools (BSL / SSL) + sweep detection
# ---------------------------------------------------------------------------

def _detect_liquidity_and_sweeps(df: pd.DataFrame, swings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For every swing high/low, scan forward to see what happens the first
    time price trades through that level:
      - Wicks through and closes back on the origin side  -> SWEEP
        (a liquidity raid; the pool is consumed, no longer "active").
      - Trades through and closes beyond it                -> BROKEN
        (a clean structural break; the pool is consumed, not a sweep).
      - Never touched again in the window                  -> ACTIVE
        (resting liquidity — still un-swept BSL/SSL).
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    active_bsl: List[Dict[str, Any]] = []
    active_ssl: List[Dict[str, Any]] = []
    sweeps: List[Dict[str, Any]] = []

    for s in swings:
        idx, price = s["index"], s["price"]

        if s["type"] == "HIGH":
            status = "ACTIVE"
            sweep_j = None
            for j in range(idx + 1, n):
                if highs[j] > price:
                    status = "SWEPT" if closes[j] < price else "BROKEN"
                    sweep_j = j
                    break

            if status == "ACTIVE":
                active_bsl.append({"price": round(price, 5), "time": s["time"], "index": idx})
            elif status == "SWEPT":
                sweeps.append({
                    "swept_level_type": "BSL",
                    "level_price": round(price, 5),
                    "sweep_price": round(float(highs[sweep_j]), 5),
                    "is_swept": True,
                    "timestamp": _iso_time(df, sweep_j),
                    "index": sweep_j,
                })

        else:  # "LOW"
            status = "ACTIVE"
            sweep_j = None
            for j in range(idx + 1, n):
                if lows[j] < price:
                    status = "SWEPT" if closes[j] > price else "BROKEN"
                    sweep_j = j
                    break

            if status == "ACTIVE":
                active_ssl.append({"price": round(price, 5), "time": s["time"], "index": idx})
            elif status == "SWEPT":
                sweeps.append({
                    "swept_level_type": "SSL",
                    "level_price": round(price, 5),
                    "sweep_price": round(float(lows[sweep_j]), 5),
                    "is_swept": True,
                    "timestamp": _iso_time(df, sweep_j),
                    "index": sweep_j,
                })

    sweeps.sort(key=lambda e: e["index"])
    recent_cutoff = max(0, n - RECENT_SWEEP_LOOKBACK)
    recent_sweeps = [e for e in sweeps if e["index"] >= recent_cutoff]

    return {"active_bsl": active_bsl, "active_ssl": active_ssl, "recent_sweeps": recent_sweeps}


# ---------------------------------------------------------------------------
# 4. Equal Highs / Equal Lows (EQH / EQL)
# ---------------------------------------------------------------------------

def _cluster_equal_levels(
    df: pd.DataFrame,
    swings: List[Dict[str, Any]],
    kind: str,
    tolerance_pct: float = EQUAL_LEVEL_TOLERANCE_PCT,
) -> List[Dict[str, Any]]:
    """
    Group swing highs (kind='HIGH') or swing lows (kind='LOW') that sit
    within `tolerance_pct` of each other into equal-level clusters
    (EQH/EQL). Only clusters of 2+ swings qualify. A cluster is reported
    only if it hasn't since been swept/broken (checked the same way as
    single BSL/SSL levels, from the cluster's last member forward).
    """
    points = [s for s in swings if s["type"] == kind]
    if len(points) < 2:
        return []

    points_sorted = sorted(points, key=lambda s: s["price"])
    clusters: List[List[Dict[str, Any]]] = []
    current = [points_sorted[0]]

    for p in points_sorted[1:]:
        anchor = current[0]["price"]
        if anchor != 0 and abs(p["price"] - anchor) / abs(anchor) <= tolerance_pct:
            current.append(p)
        else:
            if len(current) >= 2:
                clusters.append(current)
            current = [p]
    if len(current) >= 2:
        clusters.append(current)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    results: List[Dict[str, Any]] = []
    for cluster in clusters:
        level = sum(p["price"] for p in cluster) / len(cluster)
        last_idx = max(p["index"] for p in cluster)

        active = True
        for j in range(last_idx + 1, n):
            if kind == "HIGH" and highs[j] > level:
                active = False
                break
            if kind == "LOW" and lows[j] < level:
                active = False
                break

        if active:
            results.append({
                "price": round(level, 5),
                "touches": len(cluster),
                "swing_times": [p["time"] for p in cluster],
            })

    return results


# ---------------------------------------------------------------------------
# 5. Fair Value Gaps (FVG)
# ---------------------------------------------------------------------------

def _detect_fvgs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Classic 3-candle imbalance: candle1 (i-2), candle2 (i-1), candle3 (i).
    Bullish FVG when high(candle1) < low(candle3); bearish FVG when
    low(candle1) > high(candle3). Each gap is scanned forward for
    mitigation: "PARTIAL" once price trades back into the gap, "FULL"
    once price trades all the way through it.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    fvgs: List[Dict[str, Any]] = []

    for i in range(2, n):
        c1, c3 = i - 2, i

        if highs[c1] < lows[c3]:
            gap_bottom, gap_top = float(highs[c1]), float(lows[c3])
            fvgs.append(_build_fvg(df, "BULLISH", gap_bottom, gap_top, c1, i))

        elif lows[c1] > highs[c3]:
            gap_bottom, gap_top = float(highs[c3]), float(lows[c1])
            fvgs.append(_build_fvg(df, "BEARISH", gap_bottom, gap_top, c1, i))

    return fvgs


def _build_fvg(df: pd.DataFrame, direction: str, gap_bottom: float, gap_top: float, c1: int, c3: int) -> Dict[str, Any]:
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    n = len(df)

    fill_status = "NONE"
    for j in range(c3 + 1, n):
        if direction == "BULLISH":
            if lows[j] <= gap_bottom:
                fill_status = "FULL"
                break
            if lows[j] <= gap_top:
                fill_status = "PARTIAL"
        else:  # BEARISH
            if highs[j] >= gap_top:
                fill_status = "FULL"
                break
            if highs[j] >= gap_bottom:
                fill_status = "PARTIAL"

    return {
        "direction": direction,
        "gap_top": round(gap_top, 5),
        "gap_bottom": round(gap_bottom, 5),
        "formed_time": _iso_time(df, c3),
        "candle1_index": c1,
        "candle3_index": c3,
        "fill_status": fill_status,          # NONE / PARTIAL / FULL
        "is_filled": fill_status != "NONE",
    }


# ---------------------------------------------------------------------------
# 6. Order Blocks (OB)
# ---------------------------------------------------------------------------

def _detect_structure_breaks(df: pd.DataFrame, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightweight break detector (no BOS/CHOCH distinction needed here —
    just "did price close beyond the most recent unconsumed swing point,
    and where did the impulsive leg begin"). Used only to anchor Order
    Block search; structure_engine.py owns the full BOS/CHOCH narrative.
    """
    swing_highs = [s for s in swings if s["type"] == "HIGH"]
    swing_lows = [s for s in swings if s["type"] == "LOW"]
    closes = df["close"].to_numpy()
    n = len(df)

    breaks: List[Dict[str, Any]] = []
    active_high, active_low = None, None
    sh_idx = sl_idx = 0

    for i in range(n):
        while sh_idx < len(swing_highs) and swing_highs[sh_idx]["index"] < i:
            active_high = swing_highs[sh_idx]
            sh_idx += 1
        while sl_idx < len(swing_lows) and swing_lows[sl_idx]["index"] < i:
            active_low = swing_lows[sl_idx]
            sl_idx += 1

        if active_high is not None and closes[i] > active_high["price"]:
            breaks.append({"direction": "BULLISH", "trigger_index": i})
            active_high = None
        elif active_low is not None and closes[i] < active_low["price"]:
            breaks.append({"direction": "BEARISH", "trigger_index": i})
            active_low = None

    return breaks


def _find_ob_candle(df: pd.DataFrame, from_index: int, direction: str, lookback: int) -> Optional[int]:
    """
    Search backward from `from_index` (exclusive) for the nearest
    opposite-colored candle: a down-close candle anchors a BULLISH order
    block, an up-close candle anchors a BEARISH order block.
    """
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    floor = max(0, from_index - lookback)

    for k in range(from_index - 1, floor - 1, -1):
        is_down = closes[k] < opens[k]
        is_up = closes[k] > opens[k]
        if direction == "BULLISH" and is_down:
            return k
        if direction == "BEARISH" and is_up:
            return k
    return None


def _detect_order_blocks(
    df: pd.DataFrame,
    swings: List[Dict[str, Any]],
    fvgs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Order Block = the last opposite-colored candle immediately preceding
    an impulsive move that produced either a structural break (BOS) or a
    Fair Value Gap in the same direction. Deduplicated by candle index,
    then filtered down to only unmitigated (active) blocks.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    candidates: Dict[int, Dict[str, Any]] = {}  # candle index -> OB dict

    for br in _detect_structure_breaks(df, swings):
        ob_idx = _find_ob_candle(df, br["trigger_index"], br["direction"], ORDER_BLOCK_LOOKBACK)
        if ob_idx is not None and ob_idx not in candidates:
            candidates[ob_idx] = {
                "type": br["direction"],
                "ob_high": float(highs[ob_idx]),
                "ob_low": float(lows[ob_idx]),
                "formed_index": ob_idx,
                "formed_time": _iso_time(df, ob_idx),
                "source": "BOS",
            }

    for fvg in fvgs:
        ob_idx = _find_ob_candle(df, fvg["candle1_index"] + 1, fvg["direction"], ORDER_BLOCK_LOOKBACK)
        if ob_idx is not None and ob_idx not in candidates:
            candidates[ob_idx] = {
                "type": fvg["direction"],
                "ob_high": float(highs[ob_idx]),
                "ob_low": float(lows[ob_idx]),
                "formed_index": ob_idx,
                "formed_time": _iso_time(df, ob_idx),
                "source": "FVG",
            }

    active_obs: List[Dict[str, Any]] = []
    for ob in candidates.values():
        mitigated = False
        for j in range(ob["formed_index"] + 1, n):
            if ob["type"] == "BULLISH" and lows[j] <= ob["ob_high"]:
                mitigated = True
                break
            if ob["type"] == "BEARISH" and highs[j] >= ob["ob_low"]:
                mitigated = True
                break

        if not mitigated:
            active_obs.append({
                "type": ob["type"],
                "ob_high": round(ob["ob_high"], 5),
                "ob_low": round(ob["ob_low"], 5),
                "formed_time": ob["formed_time"],
                "source": ob["source"],
                "status": "ACTIVE",
            })

    active_obs.sort(key=lambda o: o["formed_time"])
    return active_obs


# ---------------------------------------------------------------------------
# 7. Public API — single timeframe
# ---------------------------------------------------------------------------

def determine_smc_structure(symbol: str, timeframe: int, count: int = 200) -> Dict[str, Any]:
    """
    Analyze `symbol` on `timeframe` (an mt5.TIMEFRAME_* constant) over the
    last `count` candles and return a dict with:
        active_bsl, active_ssl, equal_highs, equal_lows, recent_sweeps,
        unfilled_fvgs, order_blocks.

    Always returns a well-formed dict (never raises); on any failure a
    safe empty-list default structure is returned with an "error" key
    describing what went wrong.
    """
    try:
        df = _fetch_rates_df(symbol, timeframe, count)
        if df is None or df.empty:
            return _default_smc(symbol, timeframe, error="No rates available.")

        swings = _detect_swing_points(df)
        liquidity = _detect_liquidity_and_sweeps(df, swings)
        equal_highs = _cluster_equal_levels(df, swings, kind="HIGH")
        equal_lows = _cluster_equal_levels(df, swings, kind="LOW")
        fvgs = _detect_fvgs(df)
        unfilled_fvgs = [f for f in fvgs if not f["is_filled"]]
        order_blocks = _detect_order_blocks(df, swings, fvgs)

        result = _default_smc(symbol, timeframe)
        result.update({
            "active_bsl": liquidity["active_bsl"],
            "active_ssl": liquidity["active_ssl"],
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            "recent_sweeps": liquidity["recent_sweeps"],
            "unfilled_fvgs": unfilled_fvgs,
            "order_blocks": order_blocks,
            "candles_analyzed": len(df),
            "error": None,
        })
        return result

    except Exception as e:
        logger.exception("determine_smc_structure failed for %s.", symbol)
        return _default_smc(symbol, timeframe, error=str(e))


# ---------------------------------------------------------------------------
# 8. Public API — async entrypoint
# ---------------------------------------------------------------------------

async def analyze_xauusd_smc(
    symbol: str = "XAUUSD",
    timeframe: int = mt5.TIMEFRAME_H1,
    count: int = 200,
) -> Dict[str, Any]:
    """
    Async entrypoint. Runs the (blocking) MT5 + numpy/pandas analysis off
    the event loop and returns:
        active_bsl, active_ssl, equal_highs, equal_lows, recent_sweeps,
        unfilled_fvgs, order_blocks.

    Never raises — any failure degrades to a safe empty-list default
    structure with an "error" key set.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, determine_smc_structure, symbol, timeframe, count)
    except Exception as e:
        logger.exception("analyze_xauusd_smc failed for %s.", symbol)
        return _default_smc(symbol, timeframe, error=str(e))


# ---------------------------------------------------------------------------
# Standalone test entrypoint — python smc_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def _run() -> None:
        if not mt5.initialize():
            print(f"mt5.initialize() failed: {mt5.last_error()}")
            return

        try:
            print("=== SMC test: XAUUSD H1 ===")
            result = asyncio.run(analyze_xauusd_smc(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, count=200))
            print(json.dumps(result, indent=2, default=str))
        finally:
            mt5.shutdown()

    _run()
