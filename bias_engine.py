"""
bias_engine.py
===============
Phase 5 — Multi-Timeframe Bias & Trade Idea Engine for XAUUSD and major
Forex pairs.

Aggregates the live outputs of four independent, already-shipped engines
into a single confluence read and a set of conditional, scenario-based
trade ideas:

    - structure_engine.py  : BOS/CHOCH trend bias + Premium/Discount zone
    - smc_engine.py         : liquidity pools, sweeps, FVGs, order blocks
    - session_engine.py     : session clock + PDH/PDL/PWH/PWL/ASH/ASL
    - news_engine.py        : news freeze window + emergency killswitch

This module is fully standalone and self-contained: it does not import
from, modify, or otherwise touch main.py, mt5_engine.py, ai_engine.py,
chart_engine.py, or database.py. It only *reads* from the four engines
listed above via their existing public APIs.

STRICT COMPLIANCE / RISK NOTE
------------------------------
This engine NEVER emits a fixed "BUY NOW" / "SELL NOW" signal and NEVER
promises a guaranteed outcome or profit. Every entry in
`active_scenarios` is a conditional, structure-based "IF ... THEN look
for ..." trade idea that still requires the stated trigger condition to
print on the chart before it is actionable. Nothing returned by this
module should be presented to an end user as a directive to place a
trade.

KNOWN LIMITATION — reversal scenarios
--------------------------------------
The spec for `LIQUIDITY_RAID_REVERSAL` scenarios references "extreme RSI
divergence." None of the four upstream engines (structure/smc/session/
news) currently compute RSI or any other oscillator, so that leg is not
literally available yet. As a documented stand-in, this engine instead
requires a liquidity raid of a PDH/PDL/PWH/PWL level (from
session_engine + smc_engine sweep detection) *combined with* a 15M
Market Structure Shift / CHOCH in the opposing direction (from
structure_engine). This is a defensible institutional proxy for the
same idea (a raid that immediately fails and reverses), but it is not
an RSI check. Swap in a real RSI comparison here once an indicator
engine exists.

Dependencies:
    pip install MetaTrader5 pandas numpy aiohttp

Public API:
    async evaluate_xauusd_bias_and_scenarios(symbol="XAUUSD") -> dict
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import MetaTrader5 as mt5
except ImportError as e:
    raise ImportError(
        "MetaTrader5 package not found. Install with: pip install MetaTrader5 "
        "(Windows only — the official package requires a running MT5 terminal)."
    ) from e

# ---------------------------------------------------------------------------
# Upstream engines — imported defensively so a missing/broken module
# degrades this engine to safe neutral output instead of crashing on import.
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
    import news_engine
except Exception:  # pragma: no cover - defensive import
    news_engine = None  # type: ignore[assignment]

logger = logging.getLogger("bias_engine")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# How close (in price units) a liquidity sweep must land to a session level
# (PDH/PDL/PWH/PWL/ASH/ASL) to be considered "that level got swept."
# XAUUSD trades in whole-dollar-ish swings, so $0.50 is a tight-but-sane
# match band; overridable per-symbol via _level_match_tolerance().
DEFAULT_LEVEL_MATCH_TOLERANCE = 0.50

# How far ahead (minutes) to look for a pending high-impact USD event when
# deciding whether a scenario's risk_rating should be auto-flagged HIGH.
HIGH_IMPACT_LOOKAHEAD_MINUTES = 120

BULLISH_BIASES = ("BULLISH", "STRONG_BULLISH")
BEARISH_BIASES = ("BEARISH", "STRONG_BEARISH")


# ---------------------------------------------------------------------------
# Safe defaults (used whenever an upstream engine is missing, errors, or
# returns partial/empty data)
# ---------------------------------------------------------------------------

def _default_tf_structure() -> Dict[str, Any]:
    return {
        "trend_bias": "NEUTRAL",
        "zone": "UNKNOWN",
        "last_choch": None,
        "last_bos": None,
        "current_price": None,
    }


def _default_smc() -> Dict[str, Any]:
    return {
        "active_bsl": [],
        "active_ssl": [],
        "unfilled_fvgs": [],
        "order_blocks": [],
        "recent_sweeps": [],
    }


def _default_session() -> Dict[str, Any]:
    return {
        "levels": {},
        "level_distances": {},
        "asian_range_swept": {"ash_swept": False, "asl_swept": False},
        "current_price": None,
    }


def _default_news_overlay(error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "active_killswitch": False,
        "active_news_freeze": False,
        "freeze_event": None,
        "high_impact_pending_within_2h": False,
        "next_major_event": None,
        "upcoming_usd_events_today": [],
        "recent_releases": [],
        "error": error,
    }


def _fallback_result(symbol: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Last-resort payload if the whole aggregation blows up unexpectedly."""
    return {
        "symbol": symbol,
        "timeframe_biases": {
            "daily_bias": "NEUTRAL",
            "h4_bias": "NEUTRAL",
            "h1_bias": "NEUTRAL",
            "m15_bias": "NEUTRAL",
        },
        "overall_bias": "NEUTRAL",
        "confluence_score": 1,
        "active_scenarios": [],
        "risk_overlay": _default_news_overlay(error=error),
        "error": error,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 1. Safe, concurrent upstream engine calls
# ---------------------------------------------------------------------------

async def _fetch_structure(symbol: str) -> Dict[str, Dict[str, Any]]:
    """Returns dict keyed by 'D1'/'H4'/'H1'/'M15' -> structure dict."""
    if structure_engine is None:
        logger.warning("bias_engine: structure_engine unavailable, using neutral defaults.")
        return {}
    try:
        result = await structure_engine.analyze_xauusd_structure(symbol=symbol)
        return result.get("timeframes", {}) or {}
    except Exception:
        logger.exception("bias_engine: structure_engine call failed for %s.", symbol)
        return {}


async def _fetch_smc(symbol: str, timeframe: int) -> Dict[str, Any]:
    if smc_engine is None:
        logger.warning("bias_engine: smc_engine unavailable, using empty defaults.")
        return _default_smc()
    try:
        result = await smc_engine.analyze_xauusd_smc(symbol=symbol, timeframe=timeframe)
        if not result or result.get("error"):
            merged = _default_smc()
            merged.update({k: v for k, v in result.items() if k in merged and v})
            return merged
        return result
    except Exception:
        logger.exception("bias_engine: smc_engine call failed for %s.", symbol)
        return _default_smc()


async def _fetch_session(symbol: str) -> Dict[str, Any]:
    if session_engine is None:
        logger.warning("bias_engine: session_engine unavailable, using empty defaults.")
        return _default_session()
    try:
        result = await session_engine.get_xauusd_session_context(symbol=symbol)
        return result or _default_session()
    except Exception:
        logger.exception("bias_engine: session_engine call failed for %s.", symbol)
        return _default_session()


async def _fetch_news_overlay() -> Dict[str, Any]:
    if news_engine is None:
        logger.warning("bias_engine: news_engine unavailable, assuming no freeze/killswitch.")
        return _default_news_overlay(error="news_engine unavailable")
    try:
        context = await news_engine.get_institutional_news_context()
    except Exception:
        logger.exception("bias_engine: news_engine.get_institutional_news_context failed.")
        context = {}

    try:
        upcoming_high_impact = await news_engine.get_upcoming_high_impact_usd_events(
            within_minutes=HIGH_IMPACT_LOOKAHEAD_MINUTES
        )
    except Exception:
        logger.exception("bias_engine: news_engine.get_upcoming_high_impact_usd_events failed.")
        upcoming_high_impact = []

    overlay = _default_news_overlay()
    overlay.update({
        "active_killswitch": bool(context.get("active_killswitch", False)),
        "active_news_freeze": bool(context.get("active_news_freeze", False)),
        "next_major_event": context.get("next_major_event"),
        "upcoming_usd_events_today": context.get("upcoming_usd_events_today", []),
        "recent_releases": context.get("recent_releases", []),
        "high_impact_pending_within_2h": len(upcoming_high_impact) > 0,
        "error": None,
    })
    return overlay


# ---------------------------------------------------------------------------
# 2. Multi-timeframe bias confluence
# ---------------------------------------------------------------------------

def _is_bullish(bias: Optional[str]) -> bool:
    return bias in BULLISH_BIASES


def _is_bearish(bias: Optional[str]) -> bool:
    return bias in BEARISH_BIASES


def _compute_overall_bias(daily: str, h4: str, h1: str) -> str:
    """
    Rules (per Phase 5 spec):
      STRONG_BULLISH: 1D, 4H, 1H all Bullish/Strong Bullish.
      BULLISH:        4H and 1H Bullish, 1D neutral or bullish.
      BEARISH:        4H and 1H Bearish, 1D neutral or bearish.
      STRONG_BEARISH: 1D, 4H, 1H all Bearish/Strong Bearish.
      NEUTRAL:        4H/1H conflict, or nothing above applies (includes
                       "resting inside equilibrium without CHOCH" — that
                       case simply never satisfies the directional
                       conditions above and falls through here).
    """
    if _is_bullish(daily) and _is_bullish(h4) and _is_bullish(h1):
        return "STRONG_BULLISH"
    if _is_bearish(daily) and _is_bearish(h4) and _is_bearish(h1):
        return "STRONG_BEARISH"

    daily_ok_bull = daily == "NEUTRAL" or _is_bullish(daily)
    daily_ok_bear = daily == "NEUTRAL" or _is_bearish(daily)

    if _is_bullish(h4) and _is_bullish(h1) and daily_ok_bull:
        return "BULLISH"
    if _is_bearish(h4) and _is_bearish(h1) and daily_ok_bear:
        return "BEARISH"

    return "NEUTRAL"


def _compute_confluence_score(daily: str, h4: str, h1: str, m15: str, overall_bias: str) -> int:
    """
    1-5 score reflecting how many of the 4 core timeframes agree with the
    overall direction. Explicit 4H/1H conflict is capped at 1 regardless
    of anything else, since that is the strongest possible disagreement
    signal.
    """
    if (_is_bullish(h4) and _is_bearish(h1)) or (_is_bearish(h4) and _is_bullish(h1)):
        return 1

    if overall_bias in BULLISH_BIASES:
        aligned = sum(1 for b in (daily, h4, h1, m15) if _is_bullish(b))
    elif overall_bias in BEARISH_BIASES:
        aligned = sum(1 for b in (daily, h4, h1, m15) if _is_bearish(b))
    else:
        aligned = 0

    # aligned: 0-4 -> score: 1-5
    return max(1, min(5, aligned + 1))


# ---------------------------------------------------------------------------
# 3. Scenario helpers
# ---------------------------------------------------------------------------

def _level_match_tolerance(current_price: Optional[float]) -> float:
    if not current_price:
        return DEFAULT_LEVEL_MATCH_TOLERANCE
    # ~15 bps of price, floored at the default tolerance — keeps the band
    # sane across wildly different priced symbols (XAUUSD vs. e.g. USDJPY).
    return max(DEFAULT_LEVEL_MATCH_TOLERANCE, current_price * 0.0015)


def _match_session_level(price: float, levels: Dict[str, Optional[float]], tolerance: float) -> Optional[str]:
    best_name, best_dist = None, tolerance
    for name, level_price in levels.items():
        if level_price is None:
            continue
        dist = abs(price - level_price)
        if dist <= tolerance and dist <= best_dist:
            best_name, best_dist = name, dist
    return best_name


def _has_ssl_sweep(*smc_results: Dict[str, Any]) -> bool:
    return any(
        s.get("swept_level_type") == "SSL"
        for smc in smc_results
        for s in smc.get("recent_sweeps", [])
    )


def _has_bsl_sweep(*smc_results: Dict[str, Any]) -> bool:
    return any(
        s.get("swept_level_type") == "BSL"
        for smc in smc_results
        for s in smc.get("recent_sweeps", [])
    )


def _first_active_ob(smc_results: List[Dict[str, Any]], direction: str) -> Optional[Dict[str, Any]]:
    for smc in smc_results:
        for ob in smc.get("order_blocks", []):
            if ob.get("type") == direction:
                return ob
    return None


def _first_unfilled_fvg(smc_results: List[Dict[str, Any]], direction: str) -> Optional[Dict[str, Any]]:
    for smc in smc_results:
        for fvg in smc.get("unfilled_fvgs", []):
            if fvg.get("direction") == direction:
                return fvg
    return None


def _risk_rating(base_ok: bool, news_overlay: Dict[str, Any]) -> str:
    """
    LOW/MEDIUM by structural quality, auto-upgraded to HIGH whenever a
    news freeze is currently active or a high-impact USD event is
    pending within the lookahead window (STRICT COMPLIANCE requirement).
    """
    if news_overlay.get("active_news_freeze") or news_overlay.get("high_impact_pending_within_2h"):
        return "HIGH"
    if news_overlay.get("active_killswitch"):
        return "HIGH"
    return "LOW" if base_ok else "MEDIUM"


def _target_levels(levels: Dict[str, Optional[float]], keys: List[str], extra: Optional[List[float]] = None) -> List[Any]:
    targets: List[Any] = [f"{k} ({levels[k]})" for k in keys if levels.get(k) is not None]
    if extra:
        targets.extend(round(p, 5) for p in extra)
    return targets


# ---------------------------------------------------------------------------
# 4. Scenario builders
# ---------------------------------------------------------------------------

def _build_bullish_continuation(
    overall_bias: str,
    h1_structure: Dict[str, Any],
    smc_results: List[Dict[str, Any]],
    session_ctx: Dict[str, Any],
    news_overlay: Dict[str, Any],
    confluence_score: int,
) -> Optional[Dict[str, Any]]:
    if overall_bias not in BULLISH_BIASES:
        return None
    if h1_structure.get("zone") != "DISCOUNT":
        return None

    ob = _first_active_ob(smc_results, "BULLISH")
    fvg = None if ob else _first_unfilled_fvg(smc_results, "BULLISH")
    if ob is None and fvg is None:
        return None

    asl_swept = bool(session_ctx.get("asian_range_swept", {}).get("asl_swept"))
    if not (asl_swept or _has_ssl_sweep(*smc_results)):
        return None

    if ob is not None:
        zone_desc = f"Bullish Order Block [{ob['ob_low']} - {ob['ob_high']}]"
        invalidation_price = ob["ob_low"]
    else:
        zone_desc = f"Bullish FVG [{fvg['gap_bottom']} - {fvg['gap_top']}]"
        invalidation_price = fvg["gap_bottom"]

    levels = session_ctx.get("levels", {})
    active_bsl_prices = [lvl["price"] for smc in smc_results for lvl in smc.get("active_bsl", [])]

    return {
        "scenario_type": "BULLISH_DISCOUNT_REJOIN",
        "trigger_condition": (
            f"Wait for a 15M candle close confirming CHOCH back above {zone_desc} "
            f"after Asian Low / SSL sweep, before considering long continuation."
        ),
        "invalidation_level": f"Invalidated on 15M close below {invalidation_price}",
        "target_levels": _target_levels(levels, ["PDH", "ASH", "PWH"], active_bsl_prices[:2]),
        "risk_rating": _risk_rating(confluence_score >= 4, news_overlay),
    }


def _build_bearish_continuation(
    overall_bias: str,
    h1_structure: Dict[str, Any],
    smc_results: List[Dict[str, Any]],
    session_ctx: Dict[str, Any],
    news_overlay: Dict[str, Any],
    confluence_score: int,
) -> Optional[Dict[str, Any]]:
    if overall_bias not in BEARISH_BIASES:
        return None
    if h1_structure.get("zone") != "PREMIUM":
        return None

    ob = _first_active_ob(smc_results, "BEARISH")
    fvg = None if ob else _first_unfilled_fvg(smc_results, "BEARISH")
    if ob is None and fvg is None:
        return None

    ash_swept = bool(session_ctx.get("asian_range_swept", {}).get("ash_swept"))
    if not (ash_swept or _has_bsl_sweep(*smc_results)):
        return None

    if ob is not None:
        zone_desc = f"Bearish Order Block [{ob['ob_low']} - {ob['ob_high']}]"
        invalidation_price = ob["ob_high"]
    else:
        zone_desc = f"Bearish FVG [{fvg['gap_bottom']} - {fvg['gap_top']}]"
        invalidation_price = fvg["gap_top"]

    levels = session_ctx.get("levels", {})
    active_ssl_prices = [lvl["price"] for smc in smc_results for lvl in smc.get("active_ssl", [])]

    return {
        "scenario_type": "BEARISH_PREMIUM_REJOIN",
        "trigger_condition": (
            f"Wait for a 15M candle close confirming CHOCH back below {zone_desc} "
            f"after Asian High / BSL sweep, before considering short continuation."
        ),
        "invalidation_level": f"Invalidated on 15M close above {invalidation_price}",
        "target_levels": _target_levels(levels, ["PDL", "ASL", "PWL"], active_ssl_prices[:2]),
        "risk_rating": _risk_rating(confluence_score >= 4, news_overlay),
    }


def _build_liquidity_raid_reversal(
    m15_structure: Dict[str, Any],
    smc_results: List[Dict[str, Any]],
    session_ctx: Dict[str, Any],
    news_overlay: Dict[str, Any],
    confluence_score: int,
) -> Optional[Dict[str, Any]]:
    """
    See module-level KNOWN LIMITATION note: this substitutes "M15 MSS/
    CHOCH immediately following a PDH/PDL/PWH/PWL liquidity raid" for the
    RSI-divergence leg described in the spec, since no upstream engine
    currently computes RSI.
    """
    current_price = session_ctx.get("current_price") or m15_structure.get("current_price")
    levels = session_ctx.get("levels", {})
    raid_levels = {k: levels.get(k) for k in ("PDH", "PDL", "PWH", "PWL")}

    tolerance = _level_match_tolerance(current_price)
    m15_choch = m15_structure.get("last_choch")
    if not m15_choch:
        return None

    all_sweeps = [s for smc in smc_results for s in smc.get("recent_sweeps", [])]
    for sweep in all_sweeps:
        matched_level = _match_session_level(sweep.get("sweep_price", sweep.get("level_price")), raid_levels, tolerance)
        if matched_level is None:
            continue

        swept_type = sweep.get("swept_level_type")  # "BSL" (highs) or "SSL" (lows)
        # A high-side raid (BSL) should be followed by a BEARISH CHOCH to
        # count as a reversal setup, and vice versa for a low-side raid.
        expected_choch_direction = "BEARISH" if swept_type == "BSL" else "BULLISH"
        if m15_choch.get("direction") != expected_choch_direction:
            continue

        reversal_direction = "BEARISH" if swept_type == "BSL" else "BULLISH"
        target_keys = ["PDL", "ASL", "PWL"] if reversal_direction == "BEARISH" else ["PDH", "ASH", "PWH"]

        return {
            "scenario_type": "LIQUIDITY_RAID_REVERSAL",
            "trigger_condition": (
                f"Wait for 15M candle close confirming {reversal_direction} CHOCH at "
                f"{m15_choch.get('price')} following the {matched_level} liquidity raid "
                f"at {sweep.get('sweep_price')}."
            ),
            "invalidation_level": (
                f"Invalidated on 15M close back beyond the raid extreme at {sweep.get('sweep_price')}"
            ),
            "target_levels": _target_levels(levels, target_keys),
            "risk_rating": _risk_rating(confluence_score >= 3, news_overlay),
        }

    return None


# ---------------------------------------------------------------------------
# 5. Main entrypoint
# ---------------------------------------------------------------------------

async def evaluate_xauusd_bias_and_scenarios(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Query structure_engine, smc_engine, session_engine, and news_engine
    concurrently and return:
        timeframe_biases : {daily_bias, h4_bias, h1_bias, m15_bias}
        overall_bias     : STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
        confluence_score : int 1-5
        active_scenarios : list[dict] — conditional, never-directive trade ideas
        risk_overlay     : current news freeze / killswitch state

    Never raises. Any missing/partial upstream data degrades to safe
    neutral defaults so the caller always gets a well-formed dict back.
    """
    try:
        (
            structure_by_tf,
            h1_smc,
            m15_smc,
            session_ctx,
            news_overlay,
        ) = await asyncio.gather(
            _fetch_structure(symbol),
            _fetch_smc(symbol, mt5.TIMEFRAME_H1),
            _fetch_smc(symbol, mt5.TIMEFRAME_M15),
            _fetch_session(symbol),
            _fetch_news_overlay(),
        )

        d1_structure = structure_by_tf.get("D1", _default_tf_structure())
        h4_structure = structure_by_tf.get("H4", _default_tf_structure())
        h1_structure = structure_by_tf.get("H1", _default_tf_structure())
        m15_structure = structure_by_tf.get("M15", _default_tf_structure())

        daily_bias = d1_structure.get("trend_bias", "NEUTRAL") or "NEUTRAL"
        h4_bias = h4_structure.get("trend_bias", "NEUTRAL") or "NEUTRAL"
        h1_bias = h1_structure.get("trend_bias", "NEUTRAL") or "NEUTRAL"
        m15_bias = m15_structure.get("trend_bias", "NEUTRAL") or "NEUTRAL"

        overall_bias = _compute_overall_bias(daily_bias, h4_bias, h1_bias)
        confluence_score = _compute_confluence_score(daily_bias, h4_bias, h1_bias, m15_bias, overall_bias)

        smc_results = [h1_smc, m15_smc]

        active_scenarios: List[Dict[str, Any]] = []
        for builder_result in (
            _build_bullish_continuation(
                overall_bias, h1_structure, smc_results, session_ctx, news_overlay, confluence_score
            ),
            _build_bearish_continuation(
                overall_bias, h1_structure, smc_results, session_ctx, news_overlay, confluence_score
            ),
            _build_liquidity_raid_reversal(
                m15_structure, smc_results, session_ctx, news_overlay, confluence_score
            ),
        ):
            if builder_result is not None:
                active_scenarios.append(builder_result)

        return {
            "symbol": symbol,
            "timeframe_biases": {
                "daily_bias": daily_bias,
                "h4_bias": h4_bias,
                "h1_bias": h1_bias,
                "m15_bias": m15_bias,
            },
            "overall_bias": overall_bias,
            "confluence_score": confluence_score,
            "active_scenarios": active_scenarios,
            "risk_overlay": news_overlay,
            "error": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception("evaluate_xauusd_bias_and_scenarios failed for %s.", symbol)
        return _fallback_result(symbol, error=str(e))


# ---------------------------------------------------------------------------
# Standalone test entrypoint — python bias_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def _run() -> None:
        if not mt5.initialize():
            print(f"mt5.initialize() failed: {mt5.last_error()}")
            return

        try:
            print("=== Phase 5 test: XAUUSD bias & scenario engine ===")
            result = asyncio.run(evaluate_xauusd_bias_and_scenarios(symbol="XAUUSD"))
            print(json.dumps(result, indent=2, default=str))
        finally:
            mt5.shutdown()

    _run()
