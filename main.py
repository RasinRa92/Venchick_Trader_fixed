"""
main.py
=======
Primary entry point for the Forex Signal Discord Bot.

Wires together:
    - database.py     : aiosqlite persistence (traders + signals)
    - mt5_engine.py    : MT5 market data, structure analysis, lot sizing
    - news_engine.py   : economic calendar freeze + emergency killswitch
    - chart_engine.py  : in-memory PNG candlestick chart rendering
    - ai_engine.py      : Groq-powered "Venchick AI" chat analyst

Requires: discord.py>=2.0
    pip install -U discord.py python-dotenv groq

Environment variables (see .env.example):
    DISCORD_TOKEN                required
    GUILD_ID                     required, int
    UNDER_200_CHANNEL_ID         required, int
    OVER_200_CHANNEL_ID          required, int
    HIGH_WINRATE_CHANNEL_ID      required, int
    ALERTS_CHANNEL_ID            optional, int (killswitch / freeze notices)
    PRIVATE_ROOM_CATEGORY_ID     optional, int (category for /makeaccount rooms)
    FINNHUB_API_KEY / NEWSAPI_API_KEY   used by news_engine.py
    GROQ_API_KEY                  required, used by ai_engine.py (Venchick AI)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks

import config
import database
import mt5_engine
import news_engine
import ai_engine
from chart_engine import generate_signal_chart

# bias_engine combines structure_engine (BOS/CHOCH/trend), smc_engine
# (liquidity/order blocks/FVGs), session_engine (PDH/PDL/ASH/ASL), and
# news_engine into one confluence read. Imported defensively — same
# pattern ai_engine.py already uses for its optional engines — so a
# missing/broken module degrades signal generation back to the plain
# mt5_engine model instead of preventing the bot from starting.
try:
    import bias_engine
except Exception:  # pragma: no cover - defensive import
    bias_engine = None  # type: ignore[assignment]

# Second-opinion LLM veto/filter pass, applied right before a signal that
# already passed the deterministic confluence check is broadcast. Imported
# defensively like bias_engine — a missing/broken module simply means
# _open_and_broadcast_signal() skips straight to broadcasting, exactly as
# it did before this layer existed.
try:
    import second_opinion_engine
except Exception:  # pragma: no cover - defensive import
    second_opinion_engine = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional; env vars can be set at the OS level instead

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("forex_bot.main")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
GUILD_ID = int(os.getenv("GUILD_ID", "0"))

UNDER_200_CHANNEL_ID = int(os.getenv("UNDER_200_CHANNEL_ID", "0"))
OVER_200_CHANNEL_ID = int(os.getenv("OVER_200_CHANNEL_ID", "0"))
HIGH_WINRATE_CHANNEL_ID = int(os.getenv("HIGH_WINRATE_CHANNEL_ID", "0"))
ALERTS_CHANNEL_ID = int(os.getenv("ALERTS_CHANNEL_ID", "0")) or None
PRIVATE_ROOM_CATEGORY_ID = int(os.getenv("PRIVATE_ROOM_CATEGORY_ID", "0")) or None

# Phase 4: news warning / post-release explanations go to the same
# subscriber-tier channels the signal broadcaster already targets, so no
# new .env config is required.
NEWS_ALERT_CHANNEL_IDS = [
    cid for cid in (UNDER_200_CHANNEL_ID, OVER_200_CHANNEL_ID, HIGH_WINRATE_CHANNEL_ID) if cid
]

# Venchick Trader is a Gold-only system. The broker's exact Gold symbol is
# resolved from MT5 at startup via mt5_engine.resolve_gold_symbol() and
# accessed throughout via mt5_engine.get_gold_symbol(). The MT5_GOLD_SYMBOL
# env var (read by config.get_gold_symbol_candidate()) seeds that resolution.
SIGNAL_SCAN_INTERVAL_SECONDS = 10
PRICE_TICKER_INTERVAL_SECONDS = 5
NEWS_EVENT_SCAN_INTERVAL_SECONDS = 60  # Phase 4: pre-news / post-release scan cadence

# A signal is routed to #high-winrate-trades on top of the normal tier
# channels when its model confidence clears this bar.
HIGH_WINRATE_CONFIDENCE_THRESHOLD = 75

# Minimum bias_engine confluence_score (1-5: structure + liquidity +
# session + news all weighed together) required for a bias-engine-backed
# signal to actually fire. Below this, the setup isn't considered
# confident enough even if the base mt5_engine model liked it.
MIN_CONFLUENCE_SCORE = 3

# Flat reference dollar-risk used ONLY for the lot-size example printed on
# broadcast cards (real per-user sizing happens in /makeaccount + private
# channel cards, using the trader's actual balance).
REFERENCE_RISK_MICRO = 10.0
REFERENCE_RISK_STANDARD = 100.0
DEFAULT_RISK_PERCENT = 1.0  # % of balance risked per trade for private cards

DB_PATH = database.DB_PATH

# --------------------------------------------------------------------------- #
# In-memory runtime state
# --------------------------------------------------------------------------- #

# signal_id -> discord.Message (the original broadcast card), so the ticker
# loop can open/reuse a thread on it for TP/SL updates. Not persisted across
# restarts by design (thread_id isn't in the signals schema) — a restart
# simply stops posting follow-ups for signals opened before the restart.
SIGNAL_MESSAGES: dict[int, discord.Message] = {}
SIGNAL_THREADS: dict[int, discord.Thread] = {}

_last_killswitch_state = False


def _sl_pips(symbol: str, entry: float, sl: float) -> float:
    """Stop-loss distance in pips using broker-accurate point size from MT5.

    Uses mt5_engine.pip_size() which reads point/digits from symbol_info,
    giving the exact pip unit for the broker's Gold contract. Falls back to
    0.01 (correct for XAUUSD on most brokers) if MT5 data is unavailable.
    """
    try:
        pip = mt5_engine.pip_size(symbol)
    except Exception:
        pip = 0.01  # Gold (XAUUSD) default: point = 0.01 on standard brokers
    return abs(entry - sl) / pip


# --------------------------------------------------------------------------- #
# Discord bot setup
# --------------------------------------------------------------------------- #

intents = discord.Intents.default()
intents.guilds = True
intents.members = True  # needed to name private channels after the member
intents.message_content = True  # required to read message text for Venchick AI (on_message)


class ForexBot(commands.Bot):
    async def setup_hook(self) -> None:
        if GUILD_ID:
            guild_obj = discord.Object(id=GUILD_ID)
            self.tree.copy_global_to(guild=guild_obj)
            await self.tree.sync(guild=guild_obj)
        else:
            await self.tree.sync()


bot = ForexBot(command_prefix="!", intents=intents)


# --------------------------------------------------------------------------- #
# UI: "Took Trade" button attached to every broadcast signal card
# --------------------------------------------------------------------------- #

class SignalView(discord.ui.View):
    """Persistent view attached to every signal broadcast card."""

    def __init__(self, signal_id: int):
        super().__init__(timeout=None)
        self.signal_id = signal_id
        # custom_id must be unique + deterministic so the button keeps
        # working across bot restarts once re-registered as persistent.
        self.took_trade.custom_id = f"took_trade:{signal_id}"

    @discord.ui.button(label="✅ Took Trade", style=discord.ButtonStyle.success)
    async def took_trade(self, interaction: discord.Interaction, button: discord.ui.Button):
        signal = await database.get_signal(self.signal_id, db_path=DB_PATH)
        if signal is None:
            await interaction.response.send_message("That signal no longer exists.", ephemeral=True)
            return

        await interaction.response.send_message(
            f"Marked signal **#{self.signal_id}** ({signal.pair} {signal.action}) as taken. "
            f"Use `/logtrade` when it closes to record the result.",
            ephemeral=True,
        )


# --------------------------------------------------------------------------- #
# Embed / chart builders
# --------------------------------------------------------------------------- #

def _build_signal_embed(
    signal_id: int,
    analysis: dict,
    *,
    lot_micro: Optional[float] = None,
    lot_standard: Optional[float] = None,
    title_prefix: str = "",
) -> discord.Embed:
    action = analysis["action"]
    color = discord.Color.green() if action == "BUY" else discord.Color.red()

    embed = discord.Embed(
        title=f"{title_prefix}{analysis['symbol']} — {action} SIGNAL (#{signal_id})",
        description=f"Strategy: **{analysis['recommended_strategy']}**  ·  Confidence: **{analysis['confidence']:.0f}%**",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Entry", value=f"`{analysis['entry']:.5f}`", inline=True)
    embed.add_field(name="Stop Loss", value=f"`{analysis['sl']:.5f}`", inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="TP1", value=f"`{analysis['tp1']:.5f}`", inline=True)
    if analysis.get("tp2"):
        embed.add_field(name="TP2", value=f"`{analysis['tp2']:.5f}`", inline=True)
    if analysis.get("tp3"):
        embed.add_field(name="TP3", value=f"`{analysis['tp3']:.5f}`", inline=True)

    if lot_micro is not None:
        embed.add_field(name="Example Micro Lot", value=f"`{lot_micro}`", inline=True)
    if lot_standard is not None:
        embed.add_field(name="Example Standard Lot", value=f"`{lot_standard}`", inline=True)

    if analysis.get("notes"):
        embed.add_field(name="Notes", value="\n".join(f"• {n}" for n in analysis["notes"]), inline=False)

    embed.set_footer(text=f"RSI(H1): {analysis.get('rsi_h1')}  ·  ATR(H1): {analysis.get('atr_h1')}")
    return embed


async def _build_chart_file(symbol: str, analysis: dict, filename: str = "signal.png") -> Optional[discord.File]:
    df = await mt5_engine.get_market_data(symbol, "M15", count=80)
    if df is None:
        return None

    chart_df = df.set_index("time")[["open", "high", "low", "close"]].rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    buf = generate_signal_chart(
        symbol,
        chart_df,
        entry=analysis["entry"],
        sl=analysis["sl"],
        tp1=analysis["tp1"],
        tp2=analysis.get("tp2"),
        tp3=analysis.get("tp3"),
    )
    return discord.File(buf, filename=filename)


# --------------------------------------------------------------------------- #
# Confluence-aware analysis: mt5_engine gives the price levels (entry/SL/
# TP), bias_engine folds in structure (BOS/CHOCH), liquidity (order blocks/
# FVGs/sweeps), session levels, and news risk into one confluence score and
# a set of conditional scenarios. A signal only fires when both agree.
# --------------------------------------------------------------------------- #

async def _analyze_with_confluence(symbol: str) -> Optional[dict]:
    """
    Returns an analysis dict shaped exactly like
    mt5_engine.analyze_market_structure()'s output (so every existing
    caller — embeds, charts, lot sizing — keeps working unchanged), but
    with `confidence` and `notes` upgraded using bias_engine's
    multi-engine confluence read when bias_engine is available.

    Falls back to the plain mt5_engine analysis (unchanged) if bias_engine
    is unavailable, errors, has no opinion, or disagrees with the base
    model's direction — "no confluence" means "no signal", not "guess".
    """
    base = await mt5_engine.analyze_market_structure(symbol)
    if not base or base.get("action") not in ("BUY", "SELL"):
        return base

    if bias_engine is None:
        return base

    try:
        bias_result = await bias_engine.evaluate_xauusd_bias_and_scenarios(symbol=symbol)
    except Exception:
        logger.exception("_analyze_with_confluence: bias_engine failed for %s, using base analysis.", symbol)
        return base

    # Stash the raw bias_engine result under a private key so the
    # second-opinion pass in _open_and_broadcast_signal() has the full
    # confluence/news context to work with, without re-querying
    # bias_engine a second time. This key is popped off before the
    # signal is logged to the DB or embedded — see _open_and_broadcast_signal.
    base["_bias_context"] = bias_result

    overall_bias = bias_result.get("overall_bias", "NEUTRAL")
    confluence_score = bias_result.get("confluence_score", 1)
    scenarios = bias_result.get("active_scenarios") or []

    wants_bullish = base["action"] == "BUY"
    bias_agrees = (
        (wants_bullish and overall_bias in ("BULLISH", "STRONG_BULLISH"))
        or (not wants_bullish and overall_bias in ("BEARISH", "STRONG_BEARISH"))
    )

    # Require the deeper engines to actually agree with the base model's
    # direction AND to have a live, confirmed conditional scenario before
    # trusting them — otherwise a real trade opportunity from the base
    # model would get silently dropped just because bias_engine had
    # nothing to say (e.g. a fast momentum move with no confirmed
    # liquidity-raid/structure scenario yet).
    if not bias_agrees or not scenarios or confluence_score < MIN_CONFLUENCE_SCORE:
        return base

    enriched = dict(base)
    # confluence_score is 1-5; blend it with the base model's own
    # confidence rather than overriding it outright, so a strong base
    # read plus strong confluence can push toward 100 while a marginal
    # base read with only borderline confluence doesn't get inflated.
    confluence_confidence = round(confluence_score / 5 * 100)
    enriched["confidence"] = round((base.get("confidence", 0) + confluence_confidence) / 2)
    enriched["notes"] = list(base.get("notes", []))
    enriched["notes"].append(
        f"Confluence confirmed ({confluence_score}/5): {overall_bias} bias across structure, "
        f"liquidity, and session engines."
    )
    for scenario in scenarios[:2]:
        enriched["notes"].append(
            f"Scenario [{scenario.get('scenario_type')}]: {scenario.get('trigger_condition')}"
        )
    return enriched


# --------------------------------------------------------------------------- #
# Background loop #1: scan pairs -> news freeze -> evaluate -> broadcast
# --------------------------------------------------------------------------- #

@tasks.loop(seconds=SIGNAL_SCAN_INTERVAL_SECONDS)
async def signal_scanner_loop():
    global _last_killswitch_state

    killswitch = await news_engine.check_emergency_killswitch()
    if killswitch.triggered != _last_killswitch_state:
        _last_killswitch_state = killswitch.triggered
        if killswitch.triggered:
            await _post_alert(
                f"🛑 **EMERGENCY KILLSWITCH TRIGGERED** — {killswitch.reason}\n"
                f"Headline: {killswitch.matched_headline}\n"
                f"All new signal generation is halted until this clears."
            )
        else:
            await _post_alert("✅ Killswitch cleared — resuming normal signal scanning.")

    if killswitch.triggered:
        return  # hard stop, no new trades while a shock event is live

    frozen, event = await news_engine.is_news_freeze_active()
    if frozen:
        logger.info("News freeze active (%s) — skipping this scan cycle.", event.event_name if event else "?")
        return

    if not await mt5_engine.ensure_connected():
        logger.warning("signal_scanner_loop: MT5 not connected, skipping cycle.")
        return

    # Gold-only: resolve the single instrument this bot operates on.
    try:
        gold = mt5_engine.get_gold_symbol()
    except RuntimeError:
        logger.warning(
            "signal_scanner_loop: Gold symbol not yet resolved — skipping cycle. "
            "Resolution will be retried on the next cycle once MT5 connects."
        )
        return

    active_signals = await database.get_active_signals(db_path=DB_PATH)
    if any(s.pair == gold for s in active_signals):
        return  # A Gold signal is already active — do not stack another.

    try:
        analysis = await _analyze_with_confluence(gold)
    except Exception:
        logger.exception("signal_scanner_loop: analysis failed for %s", gold)
        return

    if not analysis or analysis["action"] not in ("BUY", "SELL"):
        return

    await _open_and_broadcast_signal(gold, analysis)


async def _open_and_broadcast_signal(symbol: str, analysis: dict) -> None:
    # Pop the private bias_engine context stashed by _analyze_with_confluence()
    # (if any) before it can leak into database.log_signal(), the embed, or
    # anywhere else that might serialize `analysis` — it's only needed here.
    bias_context = analysis.pop("_bias_context", None)

    # Second opinion: one more qualitative pass over an already-confluence-
    # confirmed signal, right before it's logged/broadcast. This NEVER
    # touches entry/SL/TP and can only ever move confidence DOWN or veto
    # outright — see second_opinion_engine.py for the full design rationale.
    # A missing module, missing API key, timeout, or any other failure
    # degrades to "proceed exactly as the deterministic pipeline decided",
    # so this layer can never silently stop the bot from generating signals.
    if second_opinion_engine is not None:
        try:
            verdict = await second_opinion_engine.get_second_opinion(symbol, analysis, bias_context)
        except Exception:
            logger.exception("second_opinion_engine.get_second_opinion raised unexpectedly for %s", symbol)
            verdict = None

        if verdict is not None:
            if not verdict.proceed:
                logger.info(
                    "Signal for %s VETOED by second opinion: %s", symbol, verdict.reason
                )
                return

            if verdict.confidence_adjustment:
                original_confidence = analysis.get("confidence", 0) or 0
                analysis["confidence"] = max(0, round(original_confidence + verdict.confidence_adjustment))
                analysis["notes"] = list(analysis.get("notes", []))
                analysis["notes"].append(
                    f"Second opinion: {verdict.confidence_adjustment:+d} confidence "
                    f"({original_confidence} -> {analysis['confidence']}) — {verdict.reason}"
                )

    try:
        signal_id = await database.log_signal(
            pair=symbol,
            action=analysis["action"],
            entry_price=analysis["entry"],
            sl=analysis["sl"],
            tp1=analysis["tp1"],
            tp2=analysis.get("tp2"),
            tp3=analysis.get("tp3"),
            confidence=analysis.get("confidence"),
            recommended_strategy=analysis.get("recommended_strategy"),
            db_path=DB_PATH,
        )
    except database.DatabaseError:
        logger.exception("Failed to log signal for %s", symbol)
        return

    sl_pips = _sl_pips(symbol, analysis["entry"], analysis["sl"])
    try:
        lot_micro = mt5_engine.calculate_lot_size_micro(symbol, REFERENCE_RISK_MICRO, sl_pips)
        lot_standard = mt5_engine.calculate_lot_size_standard(symbol, REFERENCE_RISK_STANDARD, sl_pips)
    except Exception:
        lot_micro = lot_standard = None

    embed = _build_signal_embed(signal_id, analysis, lot_micro=lot_micro, lot_standard=lot_standard)
    chart_file = await _build_chart_file(symbol, analysis)
    view = SignalView(signal_id)

    targets = [UNDER_200_CHANNEL_ID, OVER_200_CHANNEL_ID]
    if analysis["confidence"] >= HIGH_WINRATE_CONFIDENCE_THRESHOLD:
        targets.append(HIGH_WINRATE_CHANNEL_ID)

    sent_primary_message: Optional[discord.Message] = None
    for channel_id in targets:
        if not channel_id:
            continue
        channel = bot.get_channel(channel_id)
        if channel is None:
            logger.warning("Channel id %s not found/cached — skipping broadcast target.", channel_id)
            continue

        file_to_send = chart_file
        # discord.File objects are single-use (stream position), so re-open
        # a fresh copy per channel if we already sent one.
        if chart_file is not None and sent_primary_message is not None:
            chart_file.fp.seek(0)
            file_to_send = discord.File(chart_file.fp, filename=chart_file.filename)

        msg = await channel.send(embed=embed, file=file_to_send, view=view)
        if sent_primary_message is None:
            sent_primary_message = msg

    if sent_primary_message is not None:
        SIGNAL_MESSAGES[signal_id] = sent_primary_message

    logger.info("Broadcast signal #%s: %s %s @ %s", signal_id, analysis["action"], symbol, analysis["entry"])


async def _post_alert(text: str) -> None:
    if not ALERTS_CHANNEL_ID:
        logger.info("ALERT (no alerts channel configured): %s", text)
        return
    channel = bot.get_channel(ALERTS_CHANNEL_ID)
    if channel is not None:
        await channel.send(text)


# --------------------------------------------------------------------------- #
# Background loop #2: live price ticker for TP1/TP2/TP3/SL hits
# --------------------------------------------------------------------------- #

@tasks.loop(seconds=PRICE_TICKER_INTERVAL_SECONDS)
async def price_ticker_loop():
    if not await mt5_engine.ensure_connected():
        return

    active_signals = await database.get_active_signals(db_path=DB_PATH)
    for signal in active_signals:
        price = await _get_live_price(signal.pair)
        if price is None:
            continue

        new_status = _evaluate_signal_status(signal, price)
        if new_status is None or new_status == signal.status:
            continue

        try:
            await database.update_signal_status(signal.signal_id, new_status, db_path=DB_PATH)
        except database.DatabaseError:
            logger.exception("Failed to update status for signal #%s", signal.signal_id)
            continue

        await _post_ticker_update(signal, new_status, price)


async def _get_live_price(symbol: str) -> Optional[float]:
    """Best-effort live price via the most recent M1 candle close.

    mt5_engine.py does not currently expose a raw tick getter, so we reuse
    get_market_data() with a 1-candle M1 pull as a low-latency proxy for
    the live price. Swap this for a dedicated get_live_tick(symbol) in
    mt5_engine.py if sub-minute precision becomes important.
    """
    df = await mt5_engine.get_market_data(symbol, "M1", count=1)
    if df is None or df.empty:
        return None
    return float(df.iloc[-1]["close"])


def _evaluate_signal_status(signal: database.Signal, price: float) -> Optional[str]:
    """Returns the new status if a level was crossed, else None."""
    is_buy = signal.action == database.SignalAction.BUY

    # Stop loss always takes priority over take-profit checks.
    hit_sl = price <= signal.sl if is_buy else price >= signal.sl
    if hit_sl:
        return database.SignalStatus.SL_HIT

    # Walk take-profits from the furthest hit backwards so we never
    # regress a status (e.g. skip straight to TP3_HIT on a fast candle).
    tp_levels = [
        (signal.tp3, database.SignalStatus.TP3_HIT),
        (signal.tp2, database.SignalStatus.TP2_HIT),
        (signal.tp1, database.SignalStatus.TP1_HIT),
    ]
    status_order = [
        database.SignalStatus.ACTIVE,
        database.SignalStatus.TP1_HIT,
        database.SignalStatus.TP2_HIT,
        database.SignalStatus.TP3_HIT,
    ]
    current_rank = status_order.index(signal.status) if signal.status in status_order else 0

    for level, status in tp_levels:
        if level is None:
            continue
        hit = price >= level if is_buy else price <= level
        if hit and status_order.index(status) > current_rank:
            return status

    return None


async def _post_ticker_update(signal: database.Signal, new_status: str, price: float) -> None:
    message = SIGNAL_MESSAGES.get(signal.signal_id)
    if message is None:
        logger.info(
            "Signal #%s hit %s @ %s but no cached broadcast message (likely opened before restart).",
            signal.signal_id, new_status, price,
        )
        return

    thread = SIGNAL_THREADS.get(signal.signal_id)
    if thread is None:
        try:
            thread = await message.create_thread(name=f"{signal.pair} #{signal.signal_id} updates")
            SIGNAL_THREADS[signal.signal_id] = thread
        except discord.HTTPException:
            logger.exception("Failed to create update thread for signal #%s", signal.signal_id)
            return

    icon = "🟥" if new_status == database.SignalStatus.SL_HIT else "🟩"
    label = new_status.replace("_", " ")
    await thread.send(f"{icon} **{signal.pair} #{signal.signal_id}** — {label} hit @ `{price:.5f}`")

    if new_status in (database.SignalStatus.SL_HIT, database.SignalStatus.TP3_HIT):
        try:
            await database.update_signal_status(signal.signal_id, database.SignalStatus.CLOSED, db_path=DB_PATH)
        except database.DatabaseError:
            logger.exception("Failed to close signal #%s", signal.signal_id)
        SIGNAL_MESSAGES.pop(signal.signal_id, None)
        SIGNAL_THREADS.pop(signal.signal_id, None)


# --------------------------------------------------------------------------- #
# Background loop #3 (Phase 4): USD news pre-warnings + post-release AI notes
#
# Isolated from signal_scanner_loop / price_ticker_loop on purpose - a news
# fetch/parse/AI hiccup here must never affect trade scanning or TP/SL
# tracking. Every step below is wrapped so a failure just skips this cycle.
# --------------------------------------------------------------------------- #

@tasks.loop(seconds=NEWS_EVENT_SCAN_INTERVAL_SECONDS)
async def check_news_events_task():
    try:
        alerts = await news_engine.check_pending_news_alerts()
    except Exception:
        logger.exception("check_news_events_task: check_pending_news_alerts failed")
        return

    for alert in alerts:
        try:
            if alert.kind == "pre_warning":
                embed = _build_news_warning_embed(alert.event, alert.minutes_until or 0.0)
                await _broadcast_news_embed(embed)
            elif alert.kind == "post_release":
                # Flagship broadcast to the shared tier channels (gold has
                # historically been this bot's headline instrument).
                try:
                    _post_gold = mt5_engine.get_gold_symbol()
                except RuntimeError:
                    _post_gold = "XAUUSD"
                embed = await _build_post_release_embed(alert.event, symbol=_post_gold)
                await _broadcast_news_embed(embed)
                # Real-time, AI-tailored per-trader delivery: each trader
                # with a saved profile gets the impact read for THEIR
                # preferred pairs in their own private room.
                await _deliver_personalized_post_release(alert.event)
        except Exception:
            logger.exception(
                "check_news_events_task: failed to process %s alert for %s",
                alert.kind, alert.event.event_name,
            )


def _build_news_warning_embed(event: "news_engine.EconomicEvent", minutes_until: float) -> discord.Embed:
    embed = discord.Embed(
        title="⚠️ High-Impact USD News Warning",
        description=f"**{event.event_name}** releases in ~{max(0, round(minutes_until))} minute(s).",
        color=discord.Color.orange(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Release Time (UTC)", value=event.event_time.strftime("%Y-%m-%d %H:%M UTC"), inline=True)
    embed.add_field(name="Impact", value=event.impact.upper(), inline=True)
    embed.add_field(name="Forecast", value=str(event.forecast) if event.forecast is not None else "—", inline=True)
    embed.add_field(name="Previous", value=str(event.previous) if event.previous is not None else "—", inline=True)
    embed.set_footer(text="Venchick Trader · Institutional News Engine")
    return embed


def _affected_pairs(currency: str) -> list[str]:
    """Returns [Gold symbol] when the currency is USD, otherwise [].

    Venchick Trader is a Gold-only system. Gold (XAUUSD) is priced in USD,
    so only USD economic events are relevant. Any other currency produces an
    empty list, preventing news delivery for irrelevant instruments.
    """
    if (currency or "").upper() == "USD":
        try:
            return [mt5_engine.get_gold_symbol()]
        except RuntimeError:
            return ["XAUUSD"]  # Safe pre-resolution fallback
    return []


async def _build_post_release_embed(
    event: "news_engine.EconomicEvent",
    symbol: str = "XAUUSD",
    user_id: Optional[int] = None,
) -> discord.Embed:
    """Builds an AI-generated post-release impact card tailored to `symbol`.
    Pass `user_id` when this is being delivered into a specific trader's
    room so ai_engine can pull their saved profile into the read."""
    prompt = (
        f"The USD economic event '{event.event_name}' was just released.\n"
        f"Actual: {event.actual} | Forecast: {event.forecast} | Previous: {event.previous}\n"
        f"In 2-3 concise sentences, explain the likely short-term impact on {symbol} "
        "for an active trader. Be direct, no filler or hedging disclaimers."
    )
    try:
        explanation = await ai_engine.generate_ai_response(
            prompt,
            {"channel_id": None, "username": "news_engine", "is_private_room": user_id is not None,
             "balance": None, "account_type": None, "user_id": user_id},
        )
    except Exception:
        logger.exception("check_news_events_task: AI explanation failed for %s (%s)", event.event_name, symbol)
        explanation = "AI explanation unavailable right now — compare the actual vs. forecast manually."

    embed = discord.Embed(
        title=f"📊 {event.event_name} — Released ({symbol} impact)",
        description=explanation,
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Actual", value=str(event.actual), inline=True)
    embed.add_field(name="Forecast", value=str(event.forecast) if event.forecast is not None else "—", inline=True)
    embed.add_field(name="Previous", value=str(event.previous) if event.previous is not None else "—", inline=True)
    embed.set_footer(text="Venchick Trader · Institutional News Engine (AI-generated)")
    return embed


async def _deliver_personalized_post_release(event: "news_engine.EconomicEvent") -> None:
    """
    Phase 4+ upgrade: instead of a single generic gold-only take, this
    sends every trader with a saved /makeaccount profile a version of the
    post-release explanation tailored to the pairs THEY said they trade —
    using ai_engine's real per-user context now that user_id is threaded
    through (see the on_message channel_context fix). Best-effort per
    trader: one trader's delivery failing (closed room, rate limit, etc.)
    never blocks the others.
    """
    try:
        traders = await database.get_traders_for_news_delivery(db_path=DB_PATH)
    except Exception:
        logger.exception("check_news_events_task: failed to load traders for personalized news delivery")
        return

    if not traders:
        return

    # Gold-only: every USD news event is relevant to all Gold traders.
    # Deliver a Gold-specific post-release explainer to each trader's private room.
    try:
        gold = mt5_engine.get_gold_symbol()
    except RuntimeError:
        gold = "XAUUSD"

    for trader in traders:
        room = bot.get_channel(trader["private_channel_id"])
        if room is None:
            continue
        try:
            embed = await _build_post_release_embed(
                event, symbol=gold, user_id=trader["discord_id"]
            )
            await room.send(embed=embed)
        except discord.HTTPException:
            logger.exception(
                "check_news_events_task: failed to deliver Gold news to trader %s",
                trader["discord_id"],
            )
        except Exception:
            logger.exception(
                "check_news_events_task: unexpected error building Gold news for trader %s",
                trader["discord_id"],
            )


async def _broadcast_news_embed(embed: discord.Embed) -> None:
    if not NEWS_ALERT_CHANNEL_IDS:
        logger.info("check_news_events_task: no subscriber channels configured, skipping broadcast.")
        return
    for channel_id in NEWS_ALERT_CHANNEL_IDS:
        channel = bot.get_channel(channel_id)
        if channel is None:
            logger.warning("check_news_events_task: channel id %s not found/cached — skipping.", channel_id)
            continue
        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            logger.exception("check_news_events_task: failed to send embed to channel %s", channel_id)


# --------------------------------------------------------------------------- #
# Slash command: /makeaccount  (Phase 10 — discord.ui.Modal trading profile)
# --------------------------------------------------------------------------- #

# account_type stays a slash-command choice (not a Modal field) because
# discord.ui.Modal only supports free-text TextInput components — no
# selects/dropdowns — so a fixed Micro/Standard enum is both cleaner and
# safer as a Choice than as a typed string the user could misspell.
class MakeAccountModal(discord.ui.Modal, title="Create Your Trading Account"):
    """
    Collects the trading-profile fields that personalize Venchick AI's
    responses (ai_engine.py's `_gather_user_profile_context()`): broker/
    server, preferred pairs, country (for local session-time callouts),
    and an optional starting balance.

    Submitting a Modal is itself the interaction's *first* response, so
    the "does this user already have an account?" check happens before
    the modal is shown (in the /makeaccount command below); on_submit()
    re-checks for safety against a double-submit race, then does the
    actual channel creation + DB writes behind its own deferred response.
    """

    broker_server = discord.ui.TextInput(
        label="Broker Name & Server",
        placeholder="e.g. Exness-Real10, ICMarkets-Live",
        required=True,
        max_length=100,
    )
    preferred_pairs = discord.ui.TextInput(
        label="Preferred Trading Pairs",
        placeholder="XAUUSD  (Venchick Trader is a Gold-only system)",
        required=True,
        max_length=200,
    )
    country = discord.ui.TextInput(
        label="Country / Region",
        placeholder="e.g. Malaysia, UK, USA",
        required=True,
        max_length=60,
    )
    starting_balance = discord.ui.TextInput(
        label="Starting Balance (USD)",
        placeholder="Default: 1000",
        required=False,
        max_length=15,
    )

    def __init__(self, account_type: str):
        super().__init__()
        self.account_type = account_type

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        # Parse the optional starting-balance text field defensively —
        # it's free text in a Modal, so anything can land here.
        raw_balance = (self.starting_balance.value or "").strip()
        if not raw_balance:
            balance = 1000.0
        else:
            try:
                balance = float(raw_balance.replace(",", "").replace("$", ""))
            except ValueError:
                await interaction.followup.send(
                    f"'{raw_balance}' isn't a valid starting balance — please run `/makeaccount` again "
                    f"and enter a plain number (e.g. `1000`).",
                    ephemeral=True,
                )
                return
        if balance <= 0:
            await interaction.followup.send("Starting balance must be greater than 0.", ephemeral=True)
            return

        # Re-check for a race: two /makeaccount runs submitted concurrently.
        existing = await database.get_trader(interaction.user.id, db_path=DB_PATH)
        if existing is not None:
            await interaction.followup.send("You already have an account registered.", ephemeral=True)
            return

        guild = interaction.guild
        if guild is None:
            await interaction.followup.send("This command must be used inside the server.", ephemeral=True)
            return

        category = guild.get_channel(PRIVATE_ROOM_CATEGORY_ID) if PRIVATE_ROOM_CATEGORY_ID else None
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            interaction.user: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
            guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True, manage_channels=True),
        }

        safe_name = "".join(c for c in interaction.user.name.lower() if c.isalnum() or c == "-") or str(interaction.user.id)
        channel_name = f"room-{safe_name}"

        try:
            room = await guild.create_text_channel(
                name=channel_name,
                category=category,
                overwrites=overwrites,
                topic=f"Private signal room for {interaction.user} ({interaction.user.id})",
                reason=f"/makeaccount by {interaction.user}",
            )
        except discord.Forbidden:
            await interaction.followup.send("I don't have permission to create channels here.", ephemeral=True)
            return

        try:
            await database.add_trader(
                discord_id=interaction.user.id,
                username=str(interaction.user),
                balance=balance,
                account_type=self.account_type,
                private_channel_id=room.id,
                db_path=DB_PATH,
            )
        except database.TraderAlreadyExistsError:
            await interaction.followup.send("You already have an account registered.", ephemeral=True)
            return
        except database.DatabaseError:
            logger.exception("Failed to register trader %s", interaction.user.id)
            await interaction.followup.send("Something went wrong saving your account — please try again.", ephemeral=True)
            return

        # Trading-profile save is best-effort and separate from account
        # creation: a profile-save failure shouldn't undo the account/room
        # that already exists, since retrying /makeaccount would now just
        # hit "you already have an account". Log and continue with a
        # profile-less confirmation instead of leaving the user stuck.
        prefs = None
        try:
            prefs = await database.save_user_preferences(
                user_id=interaction.user.id,
                broker_server=self.broker_server.value,
                preferred_pairs=self.preferred_pairs.value,
                country=self.country.value,
                starting_balance=balance,
                db_path=DB_PATH,
            )
        except (database.DatabaseError, ValueError):
            logger.exception("Failed to save trading profile for %s", interaction.user.id)

        await room.send(
            f"Welcome {interaction.user.mention}! This is your private signal room.\n"
            f"Account type: **{self.account_type}**  ·  Starting balance: **${balance:.2f}**\n\n"
            f"You'll receive custom lot-sized signal cards here whenever a setup fires."
        )

        await _send_private_welcome_signals(room, interaction.user.id, balance, self.account_type)

        embed = discord.Embed(
            title="✅ Trading Account Created",
            description=f"Your private room: {room.mention}",
            color=discord.Color.green(),
        )
        embed.add_field(name="Account Type", value=self.account_type, inline=True)
        embed.add_field(name="Starting Balance", value=f"${balance:.2f}", inline=True)
        if prefs is not None:
            embed.add_field(name="Broker / Server", value=prefs.broker_server, inline=False)
            embed.add_field(
                name="Preferred Pairs",
                value=", ".join(prefs.preferred_pairs) if prefs.preferred_pairs else "—",
                inline=False,
            )
            embed.add_field(name="Country / Region", value=prefs.country, inline=False)
            embed.set_footer(text="Venchick AI will use this profile to personalize session-timing and pair analysis.")
        else:
            embed.set_footer(text="Account created, but your trading profile couldn't be saved — try /makeaccount again later to add it.")

        await interaction.followup.send(embed=embed, ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        logger.exception("MakeAccountModal failed for %s", interaction.user.id)
        message = "Something went wrong creating your account — please try again."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)


@bot.tree.command(name="makeaccount", description="Create your private signal room and register your trading account.")
@app_commands.describe(account_type="Micro or Standard XM account")
@app_commands.choices(account_type=[
    app_commands.Choice(name="Micro", value=database.AccountType.MICRO),
    app_commands.Choice(name="Standard", value=database.AccountType.STANDARD),
])
async def makeaccount(interaction: discord.Interaction, account_type: app_commands.Choice[str]):
    # This check happens BEFORE the modal is shown: sending a modal must be
    # the interaction's first response, so we can't discord.Interaction.response.defer()
    # here and check afterward the way the rest of the bot's commands do.
    existing = await database.get_trader(interaction.user.id, db_path=DB_PATH)
    if existing is not None:
        await interaction.response.send_message(
            f"You already have an account (balance: ${existing.balance:.2f}). "
            f"Your private room is <#{existing.private_channel_id}>." if existing.private_channel_id
            else "You already have an account registered.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:
        await interaction.response.send_message("This command must be used inside the server.", ephemeral=True)
        return

    await interaction.response.send_modal(MakeAccountModal(account_type=account_type.value))


async def _send_private_welcome_signals(room: discord.TextChannel, discord_id: int, balance: float, account_type: str) -> None:
    """Send the trader's currently-active signals into their new private room,
    with lot sizes custom-computed for their actual balance."""
    active_signals = await database.get_active_signals(db_path=DB_PATH)
    if not active_signals:
        return

    dollar_risk = balance * (DEFAULT_RISK_PERCENT / 100.0)
    sizer = mt5_engine.calculate_lot_size_micro if account_type == database.AccountType.MICRO else mt5_engine.calculate_lot_size_standard

    for signal in active_signals[:5]:  # cap to avoid flooding a brand-new room
        sl_pips = _sl_pips(signal.pair, signal.entry_price, signal.sl)
        try:
            lot = sizer(signal.pair, dollar_risk, sl_pips)
        except Exception:
            lot = None

        analysis_like = {
            "symbol": signal.pair,
            "action": signal.action,
            "entry": signal.entry_price,
            "sl": signal.sl,
            "tp1": signal.tp1,
            "tp2": signal.tp2,
            "tp3": signal.tp3,
            "recommended_strategy": signal.recommended_strategy or "—",
            "confidence": signal.confidence if signal.confidence is not None else 0,
            "rsi_h1": "—",
            "atr_h1": "—",
            "notes": [f"Sized for your ${balance:.2f} {account_type} account at {DEFAULT_RISK_PERCENT}% risk."],
        }
        embed = _build_signal_embed(
            signal.signal_id, analysis_like,
            lot_micro=lot if account_type == database.AccountType.MICRO else None,
            lot_standard=lot if account_type == database.AccountType.STANDARD else None,
            title_prefix="🔥 ",
        )
        await room.send(embed=embed, view=SignalView(signal.signal_id))


# --------------------------------------------------------------------------- #
# Slash command: /logtrade
# --------------------------------------------------------------------------- #

@bot.tree.command(name="logtrade", description="Log the result of a signal you took.")
@app_commands.describe(
    signal_id="The signal # shown on the card (e.g. #42 -> 42)",
    result="WIN, LOSS, or BE (break-even)",
    profit="Optional: your actual $ profit/loss on this trade (positive or negative)",
)
@app_commands.choices(result=[
    app_commands.Choice(name="WIN", value="WIN"),
    app_commands.Choice(name="LOSS", value="LOSS"),
    app_commands.Choice(name="BE", value="BE"),
])
async def logtrade(
    interaction: discord.Interaction,
    signal_id: int,
    result: app_commands.Choice[str],
    profit: Optional[float] = None,
):
    await interaction.response.defer(ephemeral=True)

    trader = await database.get_trader(interaction.user.id, db_path=DB_PATH)
    if trader is None:
        await interaction.followup.send("You don't have an account yet — run `/makeaccount` first.", ephemeral=True)
        return

    signal = await database.get_signal(signal_id, db_path=DB_PATH)
    if signal is None:
        await interaction.followup.send(f"No signal found with id #{signal_id}.", ephemeral=True)
        return

    new_balance = trader.balance
    if profit is not None:
        try:
            new_balance = await database.update_balance(interaction.user.id, profit, db_path=DB_PATH)
        except database.DatabaseError:
            logger.exception("Failed to update balance for %s", interaction.user.id)
            await interaction.followup.send("Failed to update your balance — please try again.", ephemeral=True)
            return

    # Rolling win-rate update: pull this trader's historical win rate as a
    # simple counter-free approximation — nudges toward 100/0/50 based on
    # the latest result, weighted so it doesn't whipsaw on a single trade.
    weight = 0.15
    target = 100.0 if result.value == "WIN" else (0.0 if result.value == "LOSS" else 50.0)
    new_win_rate = round(trader.win_rate * (1 - weight) + target * weight, 1)
    new_total_profit = trader.total_profit + (profit or 0.0)

    try:
        await database.update_trader_stats(
            interaction.user.id, total_profit=new_total_profit, win_rate=new_win_rate, db_path=DB_PATH,
        )
    except database.DatabaseError:
        logger.exception("Failed to update stats for %s", interaction.user.id)

    profit_str = f" (${profit:+.2f})" if profit is not None else ""
    await interaction.followup.send(
        f"Logged **{result.value}**{profit_str} on signal #{signal_id} ({signal.pair} {signal.action}).\n"
        f"New balance: **${new_balance:.2f}**  ·  Rolling win rate: **{new_win_rate:.1f}%**",
        ephemeral=True,
    )


# --------------------------------------------------------------------------- #
# Phase 7: Financial / account balance slash commands
# --------------------------------------------------------------------------- #

def _is_staff(interaction: discord.Interaction) -> bool:
    """
    Admin gate for the financial commands below. Reuses Discord's own
    'Administrator' permission (or guild-owner status) rather than
    introducing a separate role/env-config concept — server owners manage
    access the normal way via Server Settings -> Roles.
    """
    if interaction.guild is None:
        return False
    member = interaction.user
    if isinstance(member, discord.Member):
        return member.guild_permissions.administrator or member.id == interaction.guild.owner_id
    return False


async def _dm_balance_update(user: discord.abc.User, title: str, delta: float, new_balance: float, note: Optional[str], color: discord.Color) -> None:
    """Best-effort DM to the affected trader; never raises — a closed DM
    should not block the admin's command from completing."""
    embed = discord.Embed(title=title, color=color)
    embed.add_field(name="Amount", value=f"{'+' if delta >= 0 else '-'}${abs(delta):,.2f}", inline=True)
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)
    if note:
        embed.add_field(name="Note", value=note, inline=False)
    try:
        await user.send(embed=embed)
    except (discord.Forbidden, discord.HTTPException):
        logger.info("Could not DM %s a balance update confirmation (DMs closed).", user.id)


@bot.tree.command(name="addbalance", description="[Admin] Add funds to a trader's account balance.")
@app_commands.describe(
    user="The trader to credit",
    amount="Amount to add (must be positive)",
    note="Optional memo recorded on the ledger",
)
async def addbalance(interaction: discord.Interaction, user: discord.User, amount: float, note: Optional[str] = None):
    await interaction.response.defer(ephemeral=True)

    if not _is_staff(interaction):
        await interaction.followup.send("This command is restricted to server admins.", ephemeral=True)
        return

    if amount <= 0 or not (amount == amount) or amount in (float("inf"), float("-inf")):
        await interaction.followup.send(
            "Amount must be a positive, finite number. Use `/deductbalance` to remove funds.",
            ephemeral=True,
        )
        return

    trader = await database.get_trader(user.id, db_path=DB_PATH)
    if trader is None:
        await interaction.followup.send(
            f"{user.mention} doesn't have a trading account yet (no `/makeaccount` on file).",
            ephemeral=True,
        )
        return

    try:
        success, new_balance, msg = await database.update_user_balance(
            discord_id=user.id,
            amount=amount,
            tx_type=database.TransactionType.DEPOSIT,
            admin_id=interaction.user.id,
            note=note,
            db_path=DB_PATH,
        )
    except (database.DatabaseError, ValueError):
        logger.exception("addbalance failed for %s", user.id)
        await interaction.followup.send("Something went wrong updating the balance — please try again.", ephemeral=True)
        return

    if not success:
        await interaction.followup.send(f"Could not add funds: {msg}", ephemeral=True)
        return

    embed = discord.Embed(
        title="💰 Balance Updated",
        description=f"Added **${amount:,.2f}** to {user.mention}'s account.",
        color=discord.Color.green(),
    )
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)
    embed.add_field(name="Admin", value=interaction.user.mention, inline=True)
    if note:
        embed.add_field(name="Note", value=note, inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)

    await _dm_balance_update(user, "💰 Your balance was updated", amount, new_balance, note, discord.Color.green())


@bot.tree.command(name="deductbalance", description="[Admin] Deduct/withdraw funds from a trader's account balance.")
@app_commands.describe(
    user="The trader to debit",
    amount="Amount to deduct (must be positive)",
    note="Optional memo recorded on the ledger",
    confirm="Allow the balance to go negative if it would otherwise be rejected",
)
async def deductbalance(
    interaction: discord.Interaction,
    user: discord.User,
    amount: float,
    note: Optional[str] = None,
    confirm: bool = False,
):
    await interaction.response.defer(ephemeral=True)

    if not _is_staff(interaction):
        await interaction.followup.send("This command is restricted to server admins.", ephemeral=True)
        return

    if amount <= 0 or not (amount == amount) or amount in (float("inf"), float("-inf")):
        await interaction.followup.send("Amount must be a positive, finite number.", ephemeral=True)
        return

    trader = await database.get_trader(user.id, db_path=DB_PATH)
    if trader is None:
        await interaction.followup.send(
            f"{user.mention} doesn't have a trading account yet (no `/makeaccount` on file).",
            ephemeral=True,
        )
        return

    try:
        success, new_balance, msg = await database.update_user_balance(
            discord_id=user.id,
            amount=-amount,
            tx_type=database.TransactionType.WITHDRAWAL,
            admin_id=interaction.user.id,
            note=note,
            allow_negative=confirm,
            db_path=DB_PATH,
        )
    except (database.DatabaseError, ValueError):
        logger.exception("deductbalance failed for %s", user.id)
        await interaction.followup.send("Something went wrong updating the balance — please try again.", ephemeral=True)
        return

    if not success:
        await interaction.followup.send(
            f"Could not deduct funds: {msg}\nRe-run with `confirm: True` to allow a negative balance.",
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title="💸 Balance Updated",
        description=f"Deducted **${amount:,.2f}** from {user.mention}'s account.",
        color=discord.Color.red(),
    )
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)
    embed.add_field(name="Admin", value=interaction.user.mention, inline=True)
    if note:
        embed.add_field(name="Note", value=note, inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)

    await _dm_balance_update(user, "💸 Your balance was updated", -amount, new_balance, note, discord.Color.red())


@bot.tree.command(name="balance", description="View a trader's balance, account status, and tier.")
@app_commands.describe(user="Trader to look up (admins only for others; defaults to yourself)")
async def balance(interaction: discord.Interaction, user: Optional[discord.User] = None):
    await interaction.response.defer(ephemeral=True)

    target = user or interaction.user
    if user is not None and user.id != interaction.user.id and not _is_staff(interaction):
        await interaction.followup.send("You can only view your own balance.", ephemeral=True)
        return

    info = await database.get_user_balance(target.id, db_path=DB_PATH)
    if info is None:
        who = "You do" if target.id == interaction.user.id else f"{target.mention} does"
        await interaction.followup.send(f"{who} not have a trading account yet — run `/makeaccount` first.", ephemeral=True)
        return

    status = "🟢 Active" if info["balance"] >= 0 else "🔴 Negative"
    embed = discord.Embed(
        title=f"📊 Account Balance — {info['username']}",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Balance", value=f"{info['balance']:,.2f} {info['currency']}", inline=True)
    embed.add_field(name="Account Tier", value=info["account_type"], inline=True)
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Total Profit", value=f"${info['total_profit']:,.2f}", inline=True)
    embed.add_field(name="Win Rate", value=f"{info['win_rate']:.1f}%", inline=True)
    created = info["created_at"][:10] if info["created_at"] else "—"
    embed.set_footer(text=f"Account created {created}")
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="statement", description="View the last 10 transactions on a trader's account.")
@app_commands.describe(user="Trader to look up (admins only for others; defaults to yourself)")
async def statement(interaction: discord.Interaction, user: Optional[discord.User] = None):
    await interaction.response.defer(ephemeral=True)

    target = user or interaction.user
    if user is not None and user.id != interaction.user.id and not _is_staff(interaction):
        await interaction.followup.send("You can only view your own statement.", ephemeral=True)
        return

    trader = await database.get_trader(target.id, db_path=DB_PATH)
    if trader is None:
        who = "You do" if target.id == interaction.user.id else f"{target.mention} does"
        await interaction.followup.send(f"{who} not have a trading account yet — run `/makeaccount` first.", ephemeral=True)
        return

    try:
        history = await database.get_transaction_history(target.id, limit=10, db_path=DB_PATH)
    except database.DatabaseError:
        logger.exception("statement failed for %s", target.id)
        await interaction.followup.send("Something went wrong pulling the statement — please try again.", ephemeral=True)
        return

    if not history:
        who = "you" if target.id == interaction.user.id else target.mention
        await interaction.followup.send(f"No transactions recorded yet for {who}.", ephemeral=True)
        return

    lines = []
    for tx in history:
        sign = "+" if tx["amount"] >= 0 else "-"
        ts = (tx["timestamp"] or "")[:16].replace("T", " ")
        line = f"`{ts}` **{tx['type']}** {sign}${abs(tx['amount']):,.2f} → bal ${tx['balance_after']:,.2f}"
        if tx["note"]:
            line += f"  _{tx['note']}_"
        lines.append(line)

    embed = discord.Embed(
        title=f"🧾 Statement — {trader.username}",
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )
    embed.set_footer(text=f"Showing last {len(history)} transaction(s) · Current balance: ${trader.balance:,.2f}")
    await interaction.followup.send(embed=embed, ephemeral=True)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

@bot.event
async def on_ready():
    logger.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "?")

    await database.init_db(db_path=DB_PATH)

    mt5_ok = await mt5_engine.init_mt5()
    if not mt5_ok:
        logger.error("MT5 failed to connect — signal scanning/ticker loops will keep retrying each cycle.")

    # Resolve the broker's actual Gold symbol now that MT5 is connected.
    # resolve_gold_symbol() makes blocking MT5 calls, so we push it to a
    # thread executor to avoid stalling the Discord event loop — the same
    # pattern used by get_market_data() and every other MT5 read in this bot.
    if mt5_ok:
        gold_candidate = config.get_gold_symbol_candidate()
        try:
            gold_symbol = await asyncio.get_event_loop().run_in_executor(
                None, mt5_engine.resolve_gold_symbol, gold_candidate
            )
            logger.info("Venchick Trader: Gold market symbol = %s", gold_symbol)
        except RuntimeError as exc:
            logger.error(
                "on_ready: Gold symbol resolution failed — %s  "
                "Signal scanning will be skipped each cycle until Gold is found. "
                "Set MT5_GOLD_SYMBOL in .env if your broker uses a non-standard "
                "name (e.g. MT5_GOLD_SYMBOL=XAUUSDm for ICMarkets/Exness).",
                exc,
            )
    else:
        logger.warning(
            "on_ready: MT5 not connected — Gold symbol resolution deferred. "
            "The scanner loop will skip each cycle until MT5 reconnects."
        )

    if not signal_scanner_loop.is_running():
        signal_scanner_loop.start()
    if not price_ticker_loop.is_running():
        price_ticker_loop.start()
    if not check_news_events_task.is_running():
        check_news_events_task.start()

    # Sync the AI's symbol recognition with the resolved Gold symbol so that
    # broker-specific names (e.g. XAUUSDm) are correctly detected in chat.
    try:
        ai_engine.configure(extra_pairs=[mt5_engine.get_gold_symbol()])
    except RuntimeError:
        ai_engine.configure(extra_pairs=["XAUUSD"])  # Fallback if resolution failed

    try:
        gold_log = mt5_engine.get_gold_symbol()
    except RuntimeError:
        gold_log = "UNRESOLVED"
    logger.info("Venchick Trader ready. Gold symbol: %s", gold_log)


@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord gateway.")


async def _build_ai_chart_file(symbol: str) -> Optional[discord.File]:
    """Chart builder for ad-hoc AI chart requests. Reuses _build_chart_file()
    with a live signal analysis when one exists; falls back to a flat
    reference line at the last close so a chart still renders for pairs the
    model currently reads as NEUTRAL."""
    analysis = await mt5_engine.analyze_market_structure(symbol)
    if analysis and analysis.get("action") in ("BUY", "SELL"):
        return await _build_chart_file(symbol, analysis, filename=f"{symbol}_ai.png")

    df = await mt5_engine.get_market_data(symbol, "M15", count=80)
    if df is None or df.empty:
        return None
    last_close = float(df.iloc[-1]["close"])
    fallback_analysis = {
        "symbol": symbol, "entry": last_close, "sl": last_close,
        "tp1": last_close, "tp2": None, "tp3": None,
    }
    return await _build_chart_file(symbol, fallback_analysis, filename=f"{symbol}_ai.png")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Keep standard prefix commands working exactly as before.
    await bot.process_commands(message)

    if bot.user is None:
        return

    mentioned = bot.user.mentioned_in(message)

    trader = await database.get_trader(message.author.id, db_path=DB_PATH)
    is_private_room = trader is not None and trader.private_channel_id == message.channel.id

    # Venchick AI only speaks when directly @mentioned, or inside a
    # subscriber's own private /makeaccount room.
    if not (mentioned or is_private_room):
        return

    channel_context = {
        "channel_id": message.channel.id,
        "username": str(message.author),
        "is_private_room": is_private_room,
        "balance": trader.balance if trader else None,
        "account_type": trader.account_type if trader else None,
        # Required for Phase 9 conversation recall/logging and Phase 10
        # /makeaccount profile personalization — without these, ai_engine
        # silently skips both (see ai_engine.generate_ai_response docstring).
        "user_id": message.author.id,
        "message_id": message.id,
    }

    clean_content = message.content
    if mentioned and bot.user:
        clean_content = clean_content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
    if not clean_content:
        return

    async with message.channel.typing():
        try:
            reply_text = await ai_engine.generate_ai_response(clean_content, channel_context)
        except Exception:
            logger.exception("Venchick AI failed to generate a response")
            await message.reply("Sorry, I couldn't process that right now — try again in a moment.")
            return

        reply_text, chart_symbol = ai_engine.extract_chart_tag(reply_text)

        chart_file = None
        if chart_symbol:
            try:
                chart_file = await _build_ai_chart_file(chart_symbol)
            except Exception:
                logger.exception("Failed to build AI-requested chart for %s", chart_symbol)

    if chart_file:
        await message.reply(reply_text or f"Here's {chart_symbol}:", file=chart_file)
    elif chart_symbol and not chart_file:
        await message.reply(f"{reply_text}\n\n(Couldn't pull a live chart for {chart_symbol} right now.)")
    else:
        await message.reply(reply_text)


async def _shutdown():
    signal_scanner_loop.cancel()
    price_ticker_loop.cancel()
    check_news_events_task.cancel()
    mt5_engine.shutdown_mt5()


def main() -> None:
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")

    try:
        bot.run(DISCORD_TOKEN)
    finally:
        asyncio.run(_shutdown())


if __name__ == "__main__":
    main()