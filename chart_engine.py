"""
chart_engine.py
================
In-Memory PNG Chart Renderer for the Forex Discord Bot.

Renders a dark-themed 15-minute candlestick chart annotated with the
Entry / Stop-Loss / Take-Profit levels for a trade signal, entirely
in memory (no disk I/O) so it can be streamed straight into a
discord.File() for zero-latency uploads.

Dependencies: matplotlib, pandas
    pip install matplotlib pandas
"""

import io
from typing import Optional

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend, required for bots/servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_COLOR = "#131722"       # chart background (TradingView-style dark navy)
GRID_COLOR = "#1e222d"
TEXT_COLOR = "#d1d4dc"
UP_COLOR = "#26a69a"       # bullish candle
DOWN_COLOR = "#ef5350"     # bearish candle
WICK_ALPHA = 0.9

ENTRY_COLOR = "#2196F3"    # blue
SL_COLOR = "#ef5350"       # red
TP_COLOR = "#26a69a"       # green


def _plot_candles(ax, df: pd.DataFrame, width: float = 0.6) -> None:
    """Draw OHLC candlesticks onto the given axes using plain matplotlib
    patches (no external candlestick dependency required)."""
    for i, (_, row) in enumerate(df.iterrows()):
        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]
        color = UP_COLOR if close >= open_ else DOWN_COLOR

        # wick
        ax.add_line(
            Line2D([i, i], [low, high], color=color, linewidth=1, alpha=WICK_ALPHA, zorder=2)
        )
        # body
        body_low = min(open_, close)
        body_height = max(abs(close - open_), 1e-9)
        ax.add_patch(
            Rectangle(
                (i - width / 2, body_low),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                zorder=3,
            )
        )


def _style_axes(ax, fig, df: pd.DataFrame, symbol: str) -> None:
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_title(f"{symbol}  ·  15M Signal Chart", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)

    # x-axis: show a handful of readable time labels instead of every candle
    n = len(df)
    step = max(1, n // 8)
    tick_positions = list(range(0, n, step))
    tick_labels = [df.index[i].strftime("%H:%M") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-1, n)


def _draw_level(ax, price: float, color: str, label: str, x_max: float) -> None:
    ax.axhline(price, color=color, linestyle="--", linewidth=1.1, alpha=0.9, zorder=4)
    ax.text(
        x_max,
        price,
        f" {label} {price:.5f}",
        color=color,
        fontsize=8,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=5,
    )


def generate_signal_chart(
    symbol: str,
    df: pd.DataFrame,
    entry: float,
    sl: float,
    tp1: float,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
) -> io.BytesIO:
    """
    Render a dark-themed 15-minute candlestick chart with Entry/SL/TP levels
    directly to an in-memory PNG buffer (zero disk writes).

    Args:
        symbol: Trading pair, e.g. "EURUSD".
        df:     OHLC(V) data, DatetimeIndex, columns must include
                'Open', 'High', 'Low', 'Close' (Volume optional, unused).
                Expected to already be resampled to the 15-minute timeframe.
        entry:  Entry price -> blue dashed line.
        sl:     Stop-loss price -> red dashed line.
        tp1:    Take-profit 1 -> green dashed line.
        tp2:    Take-profit 2 (optional) -> green dashed line.
        tp3:    Take-profit 3 (optional) -> green dashed line.

    Returns:
        io.BytesIO: PNG image bytes, stream position reset to 0, ready to
        pass straight into discord.File(buffer, filename="signal.png").
    """
    if df is None or df.empty:
        raise ValueError("generate_signal_chart: df is empty or None")

    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"generate_signal_chart: df missing columns {missing}")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    _plot_candles(ax, df)
    _style_axes(ax, fig, df, symbol)

    x_max = len(df) - 1

    # Entry (blue)
    _draw_level(ax, entry, ENTRY_COLOR, "ENTRY", x_max)
    # Stop loss (red)
    _draw_level(ax, sl, SL_COLOR, "SL", x_max)
    # Take profits (green) — only draw the ones provided
    for label, tp in (("TP1", tp1), ("TP2", tp2), ("TP3", tp3)):
        if tp is not None:
            _draw_level(ax, tp, TP_COLOR, label, x_max)

    # Pad y-limits slightly so labels/lines near the edges aren't clipped
    all_levels = [v for v in (entry, sl, tp1, tp2, tp3) if v is not None]
    y_low = min(df["Low"].min(), *all_levels)
    y_high = max(df["High"].max(), *all_levels)
    pad = (y_high - y_low) * 0.08 or (y_high * 0.001)
    ax.set_ylim(y_low - pad, y_high + pad)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_COLOR, edgecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)  # release memory immediately, no lingering figures
    buf.seek(0)

    return buf


# ---------------------------------------------------------------------------
# Quick manual smoke test (run: python chart_engine.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    rng = pd.date_range("2024-01-01 08:00", periods=60, freq="15min")
    price = 1.0850 + np.cumsum(np.random.randn(60)) * 0.0006
    high = price + np.random.rand(60) * 0.0008
    low = price - np.random.rand(60) * 0.0008
    open_ = price + np.random.randn(60) * 0.0003
    close = price + np.random.randn(60) * 0.0003

    sample_df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=rng
    )

    out = generate_signal_chart(
        "EURUSD", sample_df, entry=1.0855, sl=1.0830, tp1=1.0880, tp2=1.0905, tp3=1.0930
    )
    with open("/tmp/_smoke_test.png", "wb") as f:
        f.write(out.getvalue())
    print("OK - wrote /tmp/_smoke_test.png, size:", len(out.getvalue()), "bytes")