"""
second_opinion_engine.py
=========================
A Groq-powered "second opinion" veto/filter layer that runs on top of an
already-confluence-confirmed signal, right before it is broadcast.

Design principle (mirrors ai_engine.py's LIVE DATA CONTEXT pattern): the
LLM is never given tool-calling access and is never allowed to invent or
alter a number. Every entry/SL/TP price level and every confidence value
comes 100% from the deterministic pipeline (mt5_engine.analyze_market_structure
+ bias_engine.evaluate_xauusd_bias_and_scenarios, already merged by
main.py's _analyze_with_confluence()). This module's ONLY job is a
qualitative pass/fail on top of that, informed by context a rules-based
system might miss — e.g. a killswitch headline that fired minutes ago and
hasn't fully cleared, a freeze window that just ended, or this symbol's
own very recent alert history contradicting the signal about to fire.

The model is asked for strict JSON:
    {"proceed": bool, "confidence_adjustment": int, "reason": str}
`confidence_adjustment` is only ever allowed to move the deterministic
confidence DOWN (clamped to [-20, 0]) — this layer can make the bot more
cautious, never more confident, and never touches entry/SL/TP.

Same paranoia level as ai_engine.py: the Groq call is wrapped in a
timeout, every failure mode (no client, no API key, timeout, network
error, malformed/unparseable JSON) degrades to the safe default —
proceed=True, confidence_adjustment=0 — i.e. the bot behaves exactly as
it did before this layer existed. A broken or misconfigured second
opinion must never silently stop the bot from generating any signals.

Environment variables:
    GROQ_SECOND_OPINION_API_KEY   optional — if set, used for this
                                  module's Groq calls instead of the main
                                  GROQ_API_KEY (lets you rate-limit /
                                  bill / rotate the second-opinion layer
                                  independently of Venchick AI chat). If
                                  unset, falls back to GROQ_API_KEY, the
                                  same key ai_engine.py uses.

Public API:
    async get_second_opinion(symbol: str, analysis: dict,
                              bias_context: dict | None) -> SecondOpinionVerdict
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from groq import AsyncGroq

import news_engine

# memory_engine is optional/defensive, exactly like every other engine in
# ai_engine.py — if it's missing or its DB layer is down, we simply lose
# the audit-log row for this verdict, never the verdict itself.
try:
    import memory_engine
except Exception:  # pragma: no cover - defensive import
    memory_engine = None  # type: ignore[assignment]

logger = logging.getLogger("forex_bot.second_opinion_engine")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_SECOND_OPINION_API_KEY = os.getenv("GROQ_SECOND_OPINION_API_KEY", "") or GROQ_API_KEY

MODEL_NAME = "llama-3.3-70b-versatile"

# Signal generation must never stall waiting on this layer — the scanner
# loop cadence is 10s (SIGNAL_SCAN_INTERVAL_SECONDS in main.py), so this
# has to resolve well inside that or fall back.
SECOND_OPINION_TIMEOUT_SECONDS = 8.0

# confidence_adjustment is only ever allowed to move confidence DOWN, and
# only within this range — a "second opinion" is a caution layer, not a
# second source of confidence.
MIN_CONFIDENCE_ADJUSTMENT = -20
MAX_CONFIDENCE_ADJUSTMENT = 0

MAX_REASON_CHARS = 200

# How many of this symbol's own recently-logged alert/verdict rows to pull
# for "does this contradict something very recent" context.
_ALERT_HISTORY_LIMIT = 5

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_client: Optional[AsyncGroq] = None
_client_init_attempted = False


@dataclass
class SecondOpinionVerdict:
    proceed: bool
    confidence_adjustment: int  # always <= 0
    reason: str
    source: str  # "llm" or "fallback"


def _fallback_verdict(reason: str) -> SecondOpinionVerdict:
    """The safe default this layer degrades to on ANY failure — identical
    to how the bot behaved before this layer existed."""
    return SecondOpinionVerdict(proceed=True, confidence_adjustment=0, reason=reason, source="fallback")


def _get_client() -> Optional[AsyncGroq]:
    global _client, _client_init_attempted
    if _client is not None:
        return _client
    if _client_init_attempted:
        return None
    _client_init_attempted = True

    if not GROQ_SECOND_OPINION_API_KEY:
        logger.error(
            "second_opinion_engine: no API key configured (GROQ_SECOND_OPINION_API_KEY / "
            "GROQ_API_KEY) — second-opinion pass will always fall back to the deterministic decision."
        )
        return None

    try:
        _client = AsyncGroq(api_key=GROQ_SECOND_OPINION_API_KEY)
        return _client
    except Exception:
        logger.exception("second_opinion_engine: failed to initialize Groq client.")
        return None


# --------------------------------------------------------------------------- #
# Context gathering — each source individually safe, never raises.
# --------------------------------------------------------------------------- #

async def _safe_news_overlay() -> Dict[str, Any]:
    """Used only when bias_context has no risk_overlay of its own (e.g.
    bias_engine was unavailable) — same institutional summary ai_engine.py
    already relies on for chat, so this module doesn't need its own news
    logic."""
    try:
        return await news_engine.get_institutional_news_context()
    except Exception:
        logger.exception("second_opinion_engine: get_institutional_news_context failed")
        return {
            "active_killswitch": False,
            "active_news_freeze": False,
            "upcoming_usd_events_today": [],
            "next_major_event": None,
            "recent_releases": [],
            "note": "news context unavailable",
        }


async def _safe_alert_history(symbol: str) -> list[Dict[str, Any]]:
    if memory_engine is None:
        return []
    try:
        return await memory_engine.get_recent_alert_history(symbol, limit=_ALERT_HISTORY_LIMIT)
    except Exception:
        logger.exception("second_opinion_engine: get_recent_alert_history failed for %s", symbol)
        return []


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _system_prompt() -> str:
    return (
        "You are a risk-review analyst performing a final qualitative check before a trading "
        "signal is broadcast to subscribers. A deterministic system has ALREADY computed the "
        "action, entry, stop-loss, take-profits, and confidence — those numbers are final and "
        "not yours to change. Your ONLY job is to look at context a rules engine might miss "
        "(a killswitch headline that fired minutes ago and hasn't fully cleared, a news freeze "
        "window that just ended, this symbol's own very recent alert history contradicting the "
        "signal, anything else in the context that reads as acutely risky right now) and decide "
        "whether the signal should still go out.\n\n"
        "STRICT RULES:\n"
        "1. You NEVER invent, restate as different, or 'correct' any price, level, percentage, or "
        "other number. Use the numbers given to you only to reason about risk, never output new ones.\n"
        "2. You may only ever make the bot MORE cautious, never less. confidence_adjustment must be "
        "an integer from -20 to 0 (0 = no change). You cannot increase confidence.\n"
        "3. Respond with STRICT JSON ONLY — no preamble, no markdown fences, no commentary before "
        "or after — in exactly this shape:\n"
        '   {"proceed": true or false, "confidence_adjustment": <int from -20 to 0>, '
        '"reason": "<plain-English reason, at most 200 characters>"}\n'
        "4. Default to proceed=true, confidence_adjustment=0 unless something in the context "
        "genuinely warrants caution — this is a targeted veto for acute risk, not a second "
        "guess of the deterministic model's own read on the market.\n"
        "5. Keep 'reason' short, concrete, and specific to what you saw in the context (cite the "
        "actual event/condition), never generic."
    )


def _format_scenarios(scenarios: Optional[list]) -> str:
    if not scenarios:
        return "  (none)"
    lines = []
    for s in scenarios[:2]:
        lines.append(f"  - [{s.get('scenario_type')}] trigger: {s.get('trigger_condition')}")
    return "\n".join(lines)


def _format_recent_releases(releases: Optional[list]) -> str:
    if not releases:
        return "  (none in the recent window)"
    lines = []
    for r in releases[:3]:
        lines.append(
            f"  - {r.get('event')} ({r.get('impact')}) at {r.get('time')}: "
            f"actual={r.get('actual')} forecast={r.get('forecast')} previous={r.get('previous')}"
        )
    return "\n".join(lines)


def _format_alert_history(history: list[Dict[str, Any]]) -> str:
    if not history:
        return "  (no prior logged alerts for this symbol)"
    lines = []
    for entry in history[:_ALERT_HISTORY_LIMIT]:
        payload = entry.get("payload") or {}
        lines.append(
            f"  - {entry.get('timestamp')} [{entry.get('event_type')}]: "
            f"{payload.get('reason') or payload.get('event') or payload}"
        )
    return "\n".join(lines)


def _build_context_text(
    symbol: str,
    analysis: Dict[str, Any],
    bias_context: Optional[Dict[str, Any]],
    news_overlay: Dict[str, Any],
    alert_history: list[Dict[str, Any]],
) -> str:
    base_section = (
        f"DETERMINISTIC SIGNAL (already computed, final — do not alter):\n"
        f"  Symbol: {symbol}\n"
        f"  Action: {analysis.get('action')}\n"
        f"  Entry: {analysis.get('entry')}\n"
        f"  SL: {analysis.get('sl')}\n"
        f"  TP1/TP2/TP3: {analysis.get('tp1')} / {analysis.get('tp2')} / {analysis.get('tp3')}\n"
        f"  Confidence (0-100): {analysis.get('confidence')}\n"
        f"  Recommended strategy: {analysis.get('recommended_strategy')}\n"
    )

    if bias_context:
        bias_section = (
            f"MULTI-ENGINE CONFLUENCE READ:\n"
            f"  Overall bias: {bias_context.get('overall_bias')}\n"
            f"  Confluence score (1-5): {bias_context.get('confluence_score')}\n"
            f"  Timeframe biases: {bias_context.get('timeframe_biases')}\n"
            f"  Active conditional scenarios:\n{_format_scenarios(bias_context.get('active_scenarios'))}\n"
        )
    else:
        bias_section = (
            "MULTI-ENGINE CONFLUENCE READ: not available for this signal — the base "
            "deterministic model fired on its own.\n"
        )

    news_section = (
        f"NEWS / RISK STATUS:\n"
        f"  Active emergency killswitch: {news_overlay.get('active_killswitch')}\n"
        f"  Active news freeze window: {news_overlay.get('active_news_freeze')}\n"
        f"  Next major USD event: {news_overlay.get('next_major_event')}\n"
        f"  Recent releases:\n{_format_recent_releases(news_overlay.get('recent_releases'))}\n"
    )

    history_section = (
        f"THIS SYMBOL'S RECENT LOGGED ALERT/VERDICT HISTORY (newest first):\n"
        f"{_format_alert_history(alert_history)}\n"
    )

    return f"{base_section}\n{bias_section}\n{news_section}\n{history_section}"


# --------------------------------------------------------------------------- #
# Defensive parsing — never trust the LLM's output shape.
# --------------------------------------------------------------------------- #

def _parse_verdict(raw_text: str) -> SecondOpinionVerdict:
    if not raw_text:
        return _fallback_verdict("second opinion returned an empty response - deterministic decision used")

    match = _JSON_BLOCK_RE.search(raw_text)
    candidate = match.group(0) if match else raw_text

    try:
        data = json.loads(candidate)
    except Exception:
        logger.warning("second_opinion_engine: could not parse JSON from response: %r", raw_text[:300])
        return _fallback_verdict("second opinion response was malformed - deterministic decision used")

    if not isinstance(data, dict):
        return _fallback_verdict("second opinion response was not a JSON object - deterministic decision used")

    proceed_raw = data.get("proceed", True)
    proceed = bool(proceed_raw) if isinstance(proceed_raw, (bool, int)) else True

    adjustment_raw = data.get("confidence_adjustment", 0)
    try:
        adjustment = int(adjustment_raw)
    except (TypeError, ValueError):
        adjustment = 0
    adjustment = max(MIN_CONFIDENCE_ADJUSTMENT, min(MAX_CONFIDENCE_ADJUSTMENT, adjustment))

    reason_raw = data.get("reason", "")
    reason = str(reason_raw).strip() if reason_raw is not None else ""
    if not reason:
        reason = "no reason provided"
    reason = reason[:MAX_REASON_CHARS]

    return SecondOpinionVerdict(proceed=proceed, confidence_adjustment=adjustment, reason=reason, source="llm")


# --------------------------------------------------------------------------- #
# Audit logging — every verdict, proceed or veto, is logged.
# --------------------------------------------------------------------------- #

async def _safe_log_verdict(symbol: str, analysis: Dict[str, Any], verdict: SecondOpinionVerdict) -> None:
    logger.info(
        "second_opinion[%s]: proceed=%s adjustment=%s source=%s reason=%s",
        symbol, verdict.proceed, verdict.confidence_adjustment, verdict.source, verdict.reason,
    )

    if memory_engine is None:
        return
    try:
        await memory_engine.log_alert_event(
            channel_id=None,
            message_id=None,
            event_type=memory_engine.AlertEventType.SECOND_OPINION,
            symbol=symbol,
            payload_dict={
                "proceed": verdict.proceed,
                "confidence_adjustment": verdict.confidence_adjustment,
                "reason": verdict.reason,
                "source": verdict.source,
                "base_action": analysis.get("action"),
                "base_confidence": analysis.get("confidence"),
            },
            current_price=analysis.get("entry"),
        )
    except Exception:
        logger.exception("second_opinion_engine: failed to log verdict via memory_engine for %s", symbol)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def get_second_opinion(
    symbol: str,
    analysis: Dict[str, Any],
    bias_context: Optional[Dict[str, Any]] = None,
) -> SecondOpinionVerdict:
    """
    Runs the LLM second-opinion pass on an already-confluence-confirmed
    signal, right before it would be broadcast.

    Args:
        symbol: e.g. "XAUUSD".
        analysis: the deterministic analysis dict from
            main.py's _analyze_with_confluence() (action/entry/sl/tp*/
            confidence/recommended_strategy). Read-only from this
            function's perspective — never mutated here.
        bias_context: the raw bias_engine.evaluate_xauusd_bias_and_scenarios()
            result for this symbol, if main.py already fetched one during
            confluence evaluation (pass None if unavailable — this
            function degrades gracefully and pulls a fresh news overlay
            on its own in that case).

    Returns:
        A SecondOpinionVerdict. On ANY failure (missing/misconfigured API
        key, timeout, network error, malformed model output), returns the
        safe fallback: proceed=True, confidence_adjustment=0 — i.e. the
        signal proceeds exactly as it would have before this layer
        existed. This function never raises.
    """
    client = _get_client()
    if client is None:
        verdict = _fallback_verdict(
            "second opinion not configured (missing GROQ_SECOND_OPINION_API_KEY / GROQ_API_KEY) "
            "- deterministic decision used"
        )
        await _safe_log_verdict(symbol, analysis, verdict)
        return verdict

    news_overlay = (bias_context or {}).get("risk_overlay") or await _safe_news_overlay()
    alert_history = await _safe_alert_history(symbol)
    context_text = _build_context_text(symbol, analysis, bias_context, news_overlay, alert_history)

    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": context_text},
                ],
                temperature=0.2,
                max_completion_tokens=200,
            ),
            timeout=SECOND_OPINION_TIMEOUT_SECONDS,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
    except asyncio.TimeoutError:
        logger.warning("second_opinion_engine: Groq call timed out for %s after %ss", symbol, SECOND_OPINION_TIMEOUT_SECONDS)
        verdict = _fallback_verdict("second opinion timed out - deterministic decision used")
        await _safe_log_verdict(symbol, analysis, verdict)
        return verdict
    except Exception:
        logger.exception("second_opinion_engine: Groq call failed for %s", symbol)
        verdict = _fallback_verdict("second opinion call failed - deterministic decision used")
        await _safe_log_verdict(symbol, analysis, verdict)
        return verdict

    verdict = _parse_verdict(raw_text)
    await _safe_log_verdict(symbol, analysis, verdict)
    return verdict


# --------------------------------------------------------------------------- #
# Standalone smoke test — python second_opinion_engine.py
#
# Runs the parser/formatter/fallback logic against a synthetic signal
# without requiring MT5 or a live Groq key (it will simply exercise the
# "no API key configured" fallback path if GROQ_SECOND_OPINION_API_KEY /
# GROQ_API_KEY aren't set, or make a real call if they are).
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def _run() -> None:
        fake_analysis = {
            "action": "BUY",
            "entry": 2385.40,
            "sl": 2379.00,
            "tp1": 2392.00,
            "tp2": 2398.00,
            "tp3": 2404.00,
            "confidence": 78,
            "recommended_strategy": "Momentum",
        }
        verdict = await get_second_opinion("XAUUSD", fake_analysis, bias_context=None)
        print("Verdict:", verdict)

    asyncio.run(_run())
