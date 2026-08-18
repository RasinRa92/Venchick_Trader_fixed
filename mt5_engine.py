"""
mt5_engine.py
--------------
MT5 Bridge, Market Data, Technical Indicators, Market Structure Analysis,
and XM Broker Lot Sizing for forex_discord_bot.

Requires: MetaTrader5, pandas, numpy
    pip install MetaTrader5 pandas numpy
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as e:
    raise ImportError(
        "MetaTrader5 package not found. Install with: pip install MetaTrader5 "
        "(Windows only — the official package requires a running MT5 terminal)."
    ) from e

import config as _config

logger = logging.getLogger("mt5_engine")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config — loaded from environment variables via config.py.
# Supported vars: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, MT5_TIMEOUT.
# Falls back to terminal-authenticated mode if no credentials are set.
# ---------------------------------------------------------------------------
MT5_CONFIG = _config.get_mt5_config()

RECONNECT_DELAY_SEC = 5
MAX_RECONNECT_ATTEMPTS = 10

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

_mt5_lock = asyncio.Lock()
_connected = False

# ---------------------------------------------------------------------------
# Gold symbol management
# ---------------------------------------------------------------------------
# Ordered list of common broker-specific Gold symbol names to try when
# auto-detecting the broker's actual symbol for XAUUSD. The first name that
# mt5.symbol_info() confirms as available wins. Extend this list if you
# encounter a broker whose Gold symbol is not covered here.
_GOLD_SYMBOL_CANDIDATES = [
    "XAUUSD",       # RoboForex Pro/Standard, XM, FP Markets, most brokers
    "XAUUSDm",      # ICMarkets, Exness, some RoboForex account types
    "XAUUSD.",      # Some ECN brokers
    "XAUUSD.pro",   # Some pro/raw-spread accounts
    "XAUUSD.ecn",   # Some ECN-labelled accounts
    "XAUUSDc",      # Rare centaccount variant
    "GOLD",         # A small number of brokers use a plain name
]

# Module-level singleton: set once by resolve_gold_symbol(), read everywhere
# via get_gold_symbol(). Never reassigned after the initial resolution.
_resolved_gold_symbol: str | None = None


def _try_select_symbol(name: str) -> bool:
    """Try to confirm that `name` is available in MT5 Market Watch.

    Attempts symbol_info() first; if the symbol is not yet visible, tries
    symbol_select() to add it to Market Watch and checks again.
    Returns True only if symbol_info() returns a non-None result.
    Never raises.
    """
    try:
        info = mt5.symbol_info(name)
        if info is not None:
            return True
        # Symbol may just not be in Market Watch yet — try to select it.
        if mt5.symbol_select(name, True):
            info = mt5.symbol_info(name)
            return info is not None
        return False
    except Exception:
        logger.debug("_try_select_symbol: exception checking '%s'.", name)
        return False


def resolve_gold_symbol(candidate: str | None = None) -> str:
    """
    Locate and lock in the broker's actual Gold (XAUUSD) symbol.

    Resolution order:
      1. The explicit `candidate` name from config (MT5_GOLD_SYMBOL env var).
      2. Each name in _GOLD_SYMBOL_CANDIDATES in order.
      3. A broader MT5 scan for any symbol matching '*XAU*'.

    The resolved name is stored in _resolved_gold_symbol and returned by
    every subsequent call to get_gold_symbol(). Idempotent: if the symbol
    was already resolved, the stored value is returned immediately.

    Args:
        candidate: Preferred symbol name (from config.get_gold_symbol_candidate()).
                   If None or empty, the _GOLD_SYMBOL_CANDIDATES list is tried
                   directly.

    Returns:
        The confirmed broker symbol name for Gold, e.g. 'XAUUSD' or 'XAUUSDm'.

    Raises:
        RuntimeError: If no tradeable Gold symbol can be found in this MT5
                      terminal. Set MT5_GOLD_SYMBOL in .env to specify the
                      exact broker symbol name.
    """
    global _resolved_gold_symbol

    # Already resolved — return immediately (idempotent).
    if _resolved_gold_symbol is not None:
        return _resolved_gold_symbol

    # Build the ordered candidate list (explicit candidate first, no duplicates).
    ordered: list[str] = []
    if candidate and candidate.strip():
        ordered.append(candidate.strip())
    for c in _GOLD_SYMBOL_CANDIDATES:
        if c not in ordered:
            ordered.append(c)

    logger.info(
        "resolve_gold_symbol: searching for Gold symbol. "
        "Will try: %s",
        ", ".join(ordered),
    )

    # --- Pass 1: try each named candidate ---
    for name in ordered:
        if _try_select_symbol(name):
            logger.info("resolve_gold_symbol: Gold symbol confirmed = %s", name)
            _resolved_gold_symbol = name
            return name

    # --- Pass 2: broader *XAU* scan ---
    logger.warning(
        "resolve_gold_symbol: named candidates not found. "
        "Scanning MT5 for *XAU* symbols...",
    )
    try:
        xau_matches = mt5.symbols_get("*XAU*") or []
        for sym_info in xau_matches:
            name = sym_info.name
            if name not in ordered and _try_select_symbol(name):
                logger.info(
                    "resolve_gold_symbol: Gold symbol found via scan = %s", name
                )
                _resolved_gold_symbol = name
                return name
    except Exception:
        logger.exception("resolve_gold_symbol: error scanning *XAU* symbols.")

    # --- Nothing found ---
    tried = ", ".join(ordered)
    raise RuntimeError(
        f"Could not find a tradeable Gold symbol in MT5. "
        f"Tried: [{tried}] plus a *XAU* symbol scan. "
        f"Ensure Gold (XAUUSD) is available in your broker's Market Watch, "
        f"or set MT5_GOLD_SYMBOL in .env with the exact broker symbol name "
        f"(e.g. MT5_GOLD_SYMBOL=XAUUSDm for ICMarkets/Exness)."
    )


def get_gold_symbol() -> str:
    """
    Return the resolved broker Gold symbol (e.g. 'XAUUSD' or 'XAUUSDm').

    This is the SINGLE SOURCE OF TRUTH for the Gold symbol name used by
    all market data, signal generation, charting, lot sizing, and AI context
    gathering. All code that previously hardcoded 'XAUUSD' should call this
    instead so broker-specific naming is handled transparently.

    Raises:
        RuntimeError: If resolve_gold_symbol() has not been called yet
                      (i.e. before MT5 connects and on_ready() runs).
    """
    if _resolved_gold_symbol is None:
        raise RuntimeError(
            "Gold symbol not yet resolved. "
            "Call mt5_engine.resolve_gold_symbol() after MT5 connects "
            "(this happens automatically in on_ready())."
        )
    return _resolved_gold_symbol


def pip_size(symbol: str) -> float:
    """
    Broker-accurate pip size for `symbol`, read from MT5 symbol_info.

    Public wrapper for the internal _pip_size() calculation:
        pip = point * 10  for 5- or 3-digit symbols (most FX pairs)
        pip = point       for 2- or 4-digit symbols (Gold, JPY, metals)

    For Gold/XAUUSD on most brokers: digits=2, point=0.01, pip=0.01.

    Raises:
        ValueError: If `symbol` is not found in MT5 Market Watch.
    """
    return _pip_size(symbol)


# ---------------------------------------------------------------------------
# 1. Connection management
# ---------------------------------------------------------------------------
async def init_mt5(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Asynchronously initialize and log in to the MT5 terminal, with
    auto-reconnect on failure. Blocking MT5 calls are pushed to a thread
    executor so the Discord event loop is never stalled.
    """
    global _connected
    cfg = {**MT5_CONFIG, **(config or {})}

    async with _mt5_lock:
        attempt = 0
        while attempt < MAX_RECONNECT_ATTEMPTS:
            attempt += 1
            ok = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _do_mt5_init(cfg)
            )
            if ok:
                _connected = True
                logger.info("MT5 connected (attempt %d).", attempt)
                return True

            logger.warning(
                "MT5 init failed (attempt %d/%d): %s",
                attempt, MAX_RECONNECT_ATTEMPTS, mt5.last_error(),
            )
            await asyncio.sleep(RECONNECT_DELAY_SEC)

        _connected = False
        logger.error("MT5 connection failed after %d attempts.", MAX_RECONNECT_ATTEMPTS)
        return False


def _do_mt5_init(cfg: Dict[str, Any]) -> bool:
    kwargs = {}
    if cfg.get("path"):
        kwargs["path"] = cfg["path"]
    if not mt5.initialize(timeout=cfg["timeout"], **kwargs):
        return False

    if cfg.get("login") and cfg.get("password") and cfg.get("server"):
        authorized = mt5.login(
            login=cfg["login"], password=cfg["password"],
            server=cfg["server"], timeout=cfg["timeout"],
        )
        if not authorized:
            mt5.shutdown()
            return False

    return mt5.terminal_info() is not None


async def ensure_connected() -> bool:
    """Call before any MT5 operation; reconnects if the terminal dropped."""
    global _connected
    if _connected and mt5.terminal_info() is not None:
        return True
    return await init_mt5()


def shutdown_mt5() -> None:
    global _connected
    mt5.shutdown()
    _connected = False
    logger.info("MT5 connection closed.")


# ---------------------------------------------------------------------------
# 2. Market data + indicators
# ---------------------------------------------------------------------------
async def get_market_data(symbol: str, timeframe: str = "H1", count: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch `count` candles for `symbol`/`timeframe`, return a DataFrame with
    OHLCV plus RSI(14), EMA(20/50/200), ATR(14), and Bollinger Bands(20,2).
    """
    if not await ensure_connected():
        logger.error("get_market_data: MT5 not connected.")
        return None

    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of {list(TIMEFRAME_MAP)}.")

    rates = await asyncio.get_event_loop().run_in_executor(
        None, lambda: mt5.copy_rates_from_pos(symbol, tf, 0, count)
    )
    if rates is None or len(rates) == 0:
        logger.error("No rates returned for %s %s: %s", symbol, timeframe, mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]

    df = _add_indicators(df)
    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA 20 / 50 / 200
    for period in (20, 50, 200):
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # RSI 14 (Wilder's smoothing)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # ATR 14 (Wilder's smoothing)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # Bollinger Bands (20, 2 std)
    mid = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    return df


# ---------------------------------------------------------------------------
# 3. Spread
# ---------------------------------------------------------------------------
async def get_live_spread(symbol: str) -> Optional[float]:
    """
    Return the current live spread for `symbol` in pips (not raw points).
    Handles both 4/5-digit (FX) and 2/3-digit (JPY, metals) quoting.
    """
    if not await ensure_connected():
        return None

    info = await asyncio.get_event_loop().run_in_executor(None, lambda: mt5.symbol_info(symbol))
    if info is None:
        if not mt5.symbol_select(symbol, True):
            logger.error("Symbol %s not found/visible: %s", symbol, mt5.last_error())
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

    point = info.point
    digits = info.digits
    spread_points = info.ask - info.bid

    # A "pip" is 10x the point for 5/3-digit symbols, 1x the point for 4/2-digit.
    pip_size = point * 10 if digits in (3, 5) else point
    spread_pips = spread_points / pip_size
    return round(spread_pips, 2)


# ---------------------------------------------------------------------------
# 4. Market structure / signal analysis
# ---------------------------------------------------------------------------
async def analyze_market_structure(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Multi-timeframe structure analysis combining trend alignment (EMA stack),
    momentum (RSI), volatility (ATR/Bollinger), and swing support/resistance.

    Returns a dict:
        {
          "symbol", "action" ("BUY"/"SELL"/"NEUTRAL"),
          "entry", "sl", "tp1", "tp2", "tp3",
          "recommended_strategy" ("Momentum"/"Mean Reversion"/"Breakout"),
          "confidence", "notes"
        }
    """
    if not await ensure_connected():
        return None

    tf_data = {}
    for tf in ("M15", "H1", "H4", "D1"):
        df = await get_market_data(symbol, tf, count=250)
        if df is None or len(df) < 210:
            logger.warning("Insufficient data for %s on %s.", symbol, tf)
            return None
        tf_data[tf] = df

    htf = tf_data["D1"]
    mtf = tf_data["H4"]
    ltf = tf_data["H1"]
    etf = tf_data["M15"]

    trend_scores = {tf: _trend_score(d) for tf, d in tf_data.items()}
    aligned_bullish = sum(1 for s in trend_scores.values() if s > 0)
    aligned_bearish = sum(1 for s in trend_scores.values() if s < 0)

    last = ltf.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr_14"])
    rsi = float(last["rsi_14"])

    swing_high, swing_low = _recent_swing_levels(mtf, lookback=60)

    # --- structure / bounce evaluation -----------------------------------
    near_support = swing_low is not None and abs(price - swing_low) <= atr * 0.5
    near_resistance = swing_high is not None and abs(price - swing_high) <= atr * 0.5
    bb_upper, bb_lower = float(last["bb_upper"]), float(last["bb_lower"])
    outside_upper_band = price >= bb_upper
    outside_lower_band = price <= bb_lower

    action = "NEUTRAL"
    strategy = "Momentum"
    notes: List[str] = []

    if aligned_bullish >= 3 and near_support and rsi < 55:
        action = "BUY"
        strategy = "Mean Reversion"
        notes.append("Multi-timeframe uptrend with pullback into support.")
    elif aligned_bearish >= 3 and near_resistance and rsi > 45:
        action = "SELL"
        strategy = "Mean Reversion"
        notes.append("Multi-timeframe downtrend with pullback into resistance.")
    elif aligned_bullish >= 3 and price > swing_high * 0.999 if swing_high else False:
        action = "BUY"
        strategy = "Breakout"
        notes.append("Bullish trend breaking above prior swing high.")
    elif aligned_bearish >= 3 and swing_low and price < swing_low * 1.001:
        action = "SELL"
        strategy = "Breakout"
        notes.append("Bearish trend breaking below prior swing low.")
    elif aligned_bullish >= 3 and rsi > 50 and not outside_upper_band:
        action = "BUY"
        strategy = "Momentum"
        notes.append("Aligned uptrend with supportive momentum, room before overbought.")
    elif aligned_bearish >= 3 and rsi < 50 and not outside_lower_band:
        action = "SELL"
        strategy = "Momentum"
        notes.append("Aligned downtrend with supportive momentum, room before oversold.")
    else:
        notes.append("No clear multi-timeframe confluence; standing aside.")

    entry = sl = tp1 = tp2 = tp3 = None
    if action == "BUY":
        entry = price
        sl = round(price - atr * 1.5, _price_digits(symbol))
        risk = price - sl
        tp1 = round(price + risk * 1.0, _price_digits(symbol))
        tp2 = round(price + risk * 2.0, _price_digits(symbol))
        tp3 = round(price + risk * 3.0, _price_digits(symbol))
    elif action == "SELL":
        entry = price
        sl = round(price + atr * 1.5, _price_digits(symbol))
        risk = sl - price
        tp1 = round(price - risk * 1.0, _price_digits(symbol))
        tp2 = round(price - risk * 2.0, _price_digits(symbol))
        tp3 = round(price - risk * 3.0, _price_digits(symbol))

    confidence = round(max(aligned_bullish, aligned_bearish) / 4 * 100, 0)

    return {
        "symbol": symbol,
        "action": action,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "recommended_strategy": strategy,
        "confidence": confidence,
        "rsi_h1": round(rsi, 1),
        "atr_h1": round(atr, _price_digits(symbol)),
        "swing_high_h4": swing_high,
        "swing_low_h4": swing_low,
        "trend_scores": trend_scores,
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _trend_score(df: pd.DataFrame) -> int:
    """+1 bullish EMA stack, -1 bearish, 0 mixed — based on the latest bar."""
    last = df.iloc[-1]
    ema20, ema50, ema200 = last["ema_20"], last["ema_50"], last["ema_200"]
    if ema20 > ema50 > ema200:
        return 1
    if ema20 < ema50 < ema200:
        return -1
    return 0


def _recent_swing_levels(df: pd.DataFrame, lookback: int = 60):
    """Simple swing high/low from the most recent `lookback` bars."""
    window = df.tail(lookback)
    if window.empty:
        return None, None
    return float(window["high"].max()), float(window["low"].min())


def _price_digits(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    return info.digits if info else 5


# ---------------------------------------------------------------------------
# 5. XM lot sizing
# ---------------------------------------------------------------------------
def _pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol {symbol} not found in Market Watch.")
    return info.point * 10 if info.digits in (3, 5) else info.point


def _pip_value_per_lot(symbol: str, lot_size: float = 1.0) -> float:
    """
    Approximate USD value of 1 pip for `lot_size` lots of `symbol`, using
    the current tick, contract size, and account currency conversion where
    the quote currency isn't USD. Falls back to tick_value if conversion
    data is unavailable.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol {symbol} not found in Market Watch.")

    pip = _pip_size(symbol)
    # trade_tick_value already reflects account currency per 1.0 lot per tick move
    if info.trade_tick_size > 0:
        pip_value_per_lot = (pip / info.trade_tick_size) * info.trade_tick_value
    else:
        pip_value_per_lot = info.trade_contract_size * pip

    return pip_value_per_lot * lot_size


def calculate_lot_size_micro(symbol: str, dollar_risk: float, sl_pips: float) -> float:
    """
    XM Micro account lot sizing (1 micro lot = 1,000 base-currency units,
    volume step is typically 0.01 of a *standard* lot = 1 micro lot).

    Returns lot size expressed in XM's standard `volume` units (e.g. 0.01
    = 1 micro lot), rounded down to the broker's volume step.
    """
    if sl_pips <= 0:
        raise ValueError("sl_pips must be > 0")

    pip_value_per_std_lot = _pip_value_per_lot(symbol, lot_size=1.0)
    raw_lots = dollar_risk / (sl_pips * pip_value_per_std_lot)

    info = mt5.symbol_info(symbol)
    step = info.volume_step if info else 0.01
    min_lot = info.volume_min if info else 0.01
    max_lot = info.volume_max if info else 100.0

    lots = max(min_lot, np.floor(raw_lots / step) * step)
    return round(min(lots, max_lot), 2)


def calculate_lot_size_standard(symbol: str, dollar_risk: float, sl_pips: float) -> float:
    """
    XM Standard account lot sizing (1 standard lot = 100,000 base-currency
    units). Same math as micro — the distinction on XM is the account
    type / leverage tier, not the formula — but rounds to the standard
    account's typical 0.01 step with a 0.01 minimum unless the symbol
    reports otherwise.
    """
    return calculate_lot_size_micro(symbol, dollar_risk, sl_pips)


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _main():
        import config as _cfg
        ok = await init_mt5()
        if not ok:
            print("Failed to connect to MT5.")
            return

        # Resolve the broker's Gold symbol before any data fetch.
        gold = resolve_gold_symbol(_cfg.get_gold_symbol_candidate())
        print(f"Gold symbol: {gold}")

        df = await get_market_data(gold, "H1", 300)
        print(df.tail())

        spread = await get_live_spread(gold)
        print(f"Live spread ({gold}): {spread} pips")

        analysis = await analyze_market_structure(gold)
        print(analysis)

        lots = calculate_lot_size_micro(gold, dollar_risk=10, sl_pips=50)
        print(f"Micro lot size for $10 risk / 50 pip SL on {gold}: {lots}")

        shutdown_mt5()

    asyncio.run(_main())