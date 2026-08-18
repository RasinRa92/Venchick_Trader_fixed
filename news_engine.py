"""
news_engine.py
--------------
Standalone news/NLP intelligence module for forex_discord_bot.

Responsibilities:
    1. Pull high-impact economic calendar events (NFP, CPI, rate decisions...)
    2. Expose a "news freeze" flag so mt5_engine / main.py can block new
       trades in the +/- 15 minute window around a high-impact release.
    3. Lightweight headline sentiment scoring (Bullish / Bearish / Volatile).
    4. An emergency killswitch that scans breaking headlines for shock
       language (flash crash, circuit breaker, emergency rate move, etc.)
       and flips a global "halt trading" flag main.py can check.

Design notes:
    - Uses aiohttp only (no sync requests calls) so it plays nicely inside
      a discord.py / MT5 async event loop.
    - Calendar + breaking news are cached in-memory with a short TTL to
      avoid hammering the free-tier API rate limits.
    - Sentiment scoring is a dependency-free keyword/heuristic model so the
      module works even if the host box has no internet access to pip
      install a full NLP stack. If you already have `vaderSentiment`
      installed, it will be used automatically as a secondary signal.
    - Every external call is wrapped so a network hiccup degrades to
      "unknown / safe defaults" rather than crashing the bot.

Environment variables:
    FINNHUB_API_KEY       - required for check_economic_calendar()
    NEWSAPI_API_KEY       - optional, used for check_emergency_killswitch()
                            breaking-headline feed (falls back to Finnhub
                            general news if not set)

Public API:
    await check_economic_calendar(force_refresh=False) -> list[EconomicEvent]
    await is_news_freeze_active() -> tuple[bool, EconomicEvent | None]
    analyze_headline_sentiment(headline_text: str) -> str
    await check_emergency_killswitch() -> KillswitchStatus

    -- Phase 4 institutional news additions --
    await fetch_usd_economic_calendar(force_refresh=False) -> list[EconomicEvent]
    await get_upcoming_high_impact_usd_events(within_minutes=60) -> list[EconomicEvent]
    await check_pending_news_alerts() -> list[NewsAlert]
    await get_institutional_news_context() -> dict

Phase 4 notes:
    - ForexFactory does not offer a public API and scraping forexfactory.com
      directly violates their Terms of Service (and their markup changes
      often enough to make a scraper a maintenance liability).
    - fetch_usd_economic_calendar() is a dual-engine pipeline: JBlanked's
      MQL5-sourced calendar (https://www.jblanked.com/news/api/docs/calendar/)
      is tried first, and it falls back to the Finnhub-backed
      check_economic_calendar() if JBlanked is unavailable, unauthenticated,
      rate-limited, or returns no data. Both paths return the same
      EconomicEvent objects, so everything downstream (alerts, AI context)
      is unaffected by which engine actually served the data.
    - JBlanked authenticates via an `Authorization: Api-Key ...` header
      (JBLANKED_API_KEY env var), not by User-Agent. _get_random_headers()
      rotates a realistic desktop-browser User-Agent as generic hygiene
      against blanket CDN/WAF bot rules - it is NOT a substitute for a
      valid API key and will not bypass real rate limits. As of writing,
      JBlanked's free tier is capped at 1 request/day, well under what a
      60-second polling loop needs - a paid plan (or a longer calendar
      cache) is required for JBlanked to actually serve as primary in
      practice.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

import aiohttp

logger = logging.getLogger("news_engine")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
JBLANKED_API_KEY = os.getenv("JBLANKED_API_KEY", "")

FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
NEWSAPI_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"

# JBlanked's MQL5-sourced "today" calendar endpoint - primary source for
# fetch_usd_economic_calendar(). See https://www.jblanked.com/news/api/docs/calendar/
JBLANKED_CALENDAR_URL = "https://www.jblanked.com/news/api/mql5/calendar/today/"
# offset=3 asks JBlanked to return timestamps in GMT/UTC (their offset scheme
# is "hours from GMT-3": GMT-3=0, GMT=3, EST=7, PST=10), so we can parse
# "Date" directly as UTC without extra timezone math.
JBLANKED_UTC_OFFSET_PARAM = 3

# Freeze window around a high-impact event (minutes before / after).
FREEZE_WINDOW_MINUTES = 15

# Phase 4: institutional pre-news warning / post-release detection windows.
PRE_NEWS_WARNING_MINUTES = 30   # warn if a HIGH-impact USD event is this close
POST_RELEASE_WINDOW_MINUTES = 15  # treat an actual as "fresh" for this long

# Cache TTLs
CALENDAR_CACHE_TTL_SECONDS = 5 * 60      # calendar doesn't change every second
NEWS_CACHE_TTL_SECONDS = 30              # breaking news needs to be fresher

# Third-tier free fallback: TradingEconomics' public "guest:guest" demo
# credentials (no signup, no key required) — heavily rate-limited and
# meant for evaluation, not production traffic, so this is only tried
# after both JBlanked and Finnhub have already failed/come back empty,
# and cached hard to avoid leaning on it. Never raises: any failure just
# means fetch_usd_economic_calendar() returns whatever the primary tiers
# already gave it (possibly an empty list).
TRADINGECONOMICS_CALENDAR_URL = "https://api.tradingeconomics.com/calendar/country/united%20states"
TRADINGECONOMICS_GUEST_CREDENTIALS = "guest:guest"
TRADINGECONOMICS_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours — guest key is heavily rate-limited
_TE_CACHE: list = []
_TE_LAST_FETCH_TIME: float = 0.0

# Venchick Trader is a Gold-only system. Gold (XAUUSD) is priced in USD,
# so only USD economic events are relevant. All other currency events are
# filtered out at the calendar-fetch stage.
WATCHED_CURRENCIES = {"USD"}

# Finnhub's economic-calendar `country` field returns 2-letter ISO-ish
# country/region codes ("US", "EU", "GB", "JP"...), NOT 3-letter currency
# codes — a mismatch that silently dropped almost every event when
# compared directly against WATCHED_CURRENCIES (which IS currency codes).
# This maps the country code Finnhub actually sends to the currency code
# the rest of this module (and fetch_usd_economic_calendar's `== "USD"`
# check) expects. Unmapped codes pass through unchanged rather than being
# dropped, so an unmapped country's events simply won't match
# WATCHED_CURRENCIES instead of raising.
_FINNHUB_COUNTRY_TO_CURRENCY = {
    "US": "USD", "EU": "EUR", "DE": "EUR", "FR": "EUR", "IT": "EUR",
    "ES": "EUR", "NL": "EUR", "GB": "GBP", "UK": "GBP", "JP": "JPY",
    "CH": "CHF", "AU": "AUD", "CA": "CAD", "NZ": "NZD",
}


def _country_code_to_currency(raw_country: str) -> str:
    """Maps a Finnhub 2-letter country code to its currency code, e.g.
    'US' -> 'USD'. Passes unmapped/unknown codes through unchanged."""
    return _FINNHUB_COUNTRY_TO_CURRENCY.get((raw_country or "").upper(), raw_country)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------

@dataclass
class EconomicEvent:
    event_time: datetime          # UTC
    country: str
    event_name: str
    impact: str                   # "low" | "medium" | "high"
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None

    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    def to_dict(self) -> dict:
        """Institutional-context-friendly dict (matches Phase 4 field spec)."""
        return {
            "event_name": self.event_name,
            "date_time_utc": self.event_time.isoformat(),
            "impact": self.impact.upper(),
            "forecast": self.forecast,
            "previous": self.previous,
            "actual": self.actual,
            "country": self.country,
        }


@dataclass
class KillswitchStatus:
    triggered: bool
    reason: Optional[str] = None
    matched_headline: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NewsAlert:
    """A single pending alert surfaced by check_pending_news_alerts().

    kind is either "pre_warning" (event happening within
    PRE_NEWS_WARNING_MINUTES) or "post_release" (actual value just
    published, within POST_RELEASE_WINDOW_MINUTES).
    """
    kind: str
    event: EconomicEvent
    minutes_until: Optional[float] = None  # positive = future, negative = elapsed since release


# --------------------------------------------------------------------------
# Internal cache state
# --------------------------------------------------------------------------

_calendar_cache: dict = {"events": [], "fetched_at": 0.0}
_news_cache: dict = {"headlines": [], "fetched_at": 0.0}

# Isolated cache for _fetch_jblanked_calendar(), kept separate from
# _calendar_cache (Finnhub, CALENDAR_CACHE_TTL_SECONDS) so JBlanked's own
# rate limits/free-tier quota aren't hammered by a 60-second polling loop.
# JBlanked's free tier is 1 request/day (see module docstring) — a 1-hour
# TTL burned the whole day's quota by mid-morning and left JBlanked
# 429ing (and silently falling back to Finnhub) for the rest of the day.
# 23 hours leaves a safety margin so one refresh/day never straddles the
# provider's own daily reset boundary.
JBLANKED_CACHE_TTL_SECONDS = 23 * 60 * 60  # ~23 hours
_JBLANKED_CACHE: list = []
_JBLANKED_LAST_FETCH_TIME: float = 0.0

# Dedup state for check_pending_news_alerts() so the same event doesn't
# trigger repeat Discord alerts every 60-second scan cycle.
_alerted_pre_warnings: set[str] = set()
_alerted_post_releases: set[str] = set()

# Set by check_emergency_killswitch(); main.py can import this flag directly
# for a zero-await "is trading halted right now" check between full scans.
GLOBAL_TRADING_HALTED: bool = False


# --------------------------------------------------------------------------
# 1. Economic calendar
# --------------------------------------------------------------------------

# Keywords used to upgrade an event to "high impact" if the API's own impact
# rating is missing/unreliable (Finnhub's free tier sometimes omits it).
_HIGH_IMPACT_KEYWORDS = (
    "non-farm", "nonfarm", "nfp", "cpi", "consumer price index",
    "interest rate decision", "rate decision", "fomc", "fed funds",
    "ecb press conference", "boe rate", "boj rate", "gdp", "unemployment rate",
    "ppi", "retail sales", "employment change",
)


def _guess_impact(event_name: str, raw_impact) -> str:
    """Fall back to keyword matching when the API impact field is blank."""
    if raw_impact in (3, "3", "high", "High"):
        return "high"
    if raw_impact in (2, "2", "medium", "Medium"):
        return "medium"
    if raw_impact in (1, "1", "low", "Low"):
        return "low"
    name_lower = (event_name or "").lower()
    if any(kw in name_lower for kw in _HIGH_IMPACT_KEYWORDS):
        return "high"
    return "low"


async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict) -> Optional[dict | list]:
    try:
        async with session.get(url, params=params, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status != 200:
                logger.warning("news_engine: %s returned HTTP %s", url, resp.status)
                return None
            return await resp.json(content_type=None)
    except asyncio.TimeoutError:
        logger.warning("news_engine: request to %s timed out", url)
    except aiohttp.ClientError as exc:
        logger.warning("news_engine: request to %s failed: %s", url, exc)
    return None


async def check_economic_calendar(force_refresh: bool = False) -> list[EconomicEvent]:
    """
    Fetch upcoming/recent high-impact economic calendar events (NFP, CPI,
    central bank rate decisions, etc.) from Finnhub's free calendar endpoint.

    Results are cached for CALENDAR_CACHE_TTL_SECONDS to respect free-tier
    rate limits. Returns an empty list (rather than raising) on any failure
    so callers can fail safe.
    """
    now_ts = time.time()
    if not force_refresh and (now_ts - _calendar_cache["fetched_at"]) < CALENDAR_CACHE_TTL_SECONDS:
        return _calendar_cache["events"]

    if not FINNHUB_API_KEY:
        logger.warning("news_engine: FINNHUB_API_KEY not set, skipping calendar fetch")
        return _calendar_cache["events"]

    today = datetime.now(timezone.utc).date()
    params = {
        "from": (today - timedelta(days=1)).isoformat(),
        "to": (today + timedelta(days=2)).isoformat(),
        "token": FINNHUB_API_KEY,
    }

    events: list[EconomicEvent] = []
    async with aiohttp.ClientSession() as session:
        data = await _fetch_json(session, FINNHUB_CALENDAR_URL, params)

    raw_events = (data or {}).get("economicCalendar", []) if isinstance(data, dict) else []

    for raw in raw_events:
        try:
            # Finnhub sends a 2-letter country code here, not a currency
            # code — map it before comparing against WATCHED_CURRENCIES
            # (see _FINNHUB_COUNTRY_TO_CURRENCY above) or literally every
            # event gets filtered out.
            country = _country_code_to_currency(raw.get("country", ""))
            if WATCHED_CURRENCIES and country and country not in WATCHED_CURRENCIES:
                continue

            when_str = raw.get("time") or raw.get("date")
            if not when_str:
                continue
            # Finnhub timestamps are typically "YYYY-MM-DD HH:MM:SS" UTC.
            event_time = datetime.strptime(when_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

            event_name = raw.get("event", "Unknown Event")
            impact = _guess_impact(event_name, raw.get("impact"))

            events.append(EconomicEvent(
                event_time=event_time,
                country=country,
                event_name=event_name,
                impact=impact,
                actual=raw.get("actual"),
                forecast=raw.get("estimate") or raw.get("forecast"),
                previous=raw.get("prev") or raw.get("previous"),
            ))
        except (ValueError, TypeError) as exc:
            logger.debug("news_engine: skipping malformed calendar row: %s (%s)", raw, exc)
            continue

    events.sort(key=lambda e: e.event_time)
    _calendar_cache["events"] = events
    _calendar_cache["fetched_at"] = now_ts
    return events


# --------------------------------------------------------------------------
# 2. News freeze window
# --------------------------------------------------------------------------

async def is_news_freeze_active() -> tuple[bool, Optional[EconomicEvent]]:
    """
    Returns (True, event) if "now" falls within FREEZE_WINDOW_MINUTES of a
    high-impact event (before or after its scheduled time), else (False, None).

    Intended usage in mt5_engine.py / main.py:
        frozen, event = await is_news_freeze_active()
        if frozen:
            await ctx.send(f"⚠️ Trading paused - {event.event_name} window")
            return
    """
    events = await check_economic_calendar()
    now = datetime.now(timezone.utc)
    window = timedelta(minutes=FREEZE_WINDOW_MINUTES)

    for event in events:
        if not event.is_high_impact():
            continue
        if abs(now - event.event_time) <= window:
            return True, event

    return False, None


# --------------------------------------------------------------------------
# 3. Headline sentiment scoring
# --------------------------------------------------------------------------

_BULLISH_WORDS = {
    "beats", "beat", "surge", "surges", "soars", "soar", "rally", "rallies",
    "strong", "stronger", "growth", "expands", "expansion", "upbeat",
    "optimism", "optimistic", "gains", "gain", "recovery", "recovers",
    "outperform", "record high", "hawkish", "tightening", "rate hike",
    "exceeds expectations", "better than expected", "upgrade", "upgraded",
}

_BEARISH_WORDS = {
    "misses", "miss", "plunge", "plunges", "slump", "slumps", "tumbles",
    "tumble", "weak", "weaker", "contraction", "contracts", "downbeat",
    "pessimism", "pessimistic", "losses", "loss", "recession", "slowdown",
    "underperform", "record low", "dovish", "easing", "rate cut",
    "misses expectations", "worse than expected", "downgrade", "downgraded",
}

_VOLATILE_WORDS = {
    "shock", "shocks", "unexpected", "unexpectedly", "surprise", "surprises",
    "surprising", "spike", "spikes", "volatile", "volatility", "chaos",
    "turmoil", "panic", "flash crash", "circuit breaker", "halted", "halt",
    "emergency", "intervention", "black swan", "crash", "crashes", "plummet",
    "plummets", "whipsaw",
}

_WORD_RE = re.compile(r"[a-z']+")


def _try_vader(text: str) -> Optional[str]:
    """Optional secondary signal if vaderSentiment happens to be installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except ImportError:
        return None
    try:
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        compound = scores.get("compound", 0.0)
        if compound >= 0.35:
            return "Bullish"
        if compound <= -0.35:
            return "Bearish"
        return None  # let keyword model decide neutral/volatile cases
    except Exception as exc:
        logger.debug("news_engine: vader scoring failed: %s", exc)
        return None


def analyze_headline_sentiment(headline_text: str) -> str:
    """
    Fast, dependency-free sentiment classification for a single headline.

    Returns one of: "Bullish", "Bearish", "Volatile", "Neutral".

    Priority:
        1. Shock/volatility language always wins (a "surprise rate hike" is
           tradeable-dangerous regardless of directional tone).
        2. Otherwise compare bullish vs bearish keyword hits.
        3. If vaderSentiment is installed, use it to break close ties.
        4. Fall back to Neutral.
    """
    if not headline_text or not headline_text.strip():
        return "Neutral"

    text_lower = headline_text.lower()
    words = set(_WORD_RE.findall(text_lower))

    # Multi-word phrase check (regex tokens strip spaces, so check substrings too)
    def _hits(vocab: set[str]) -> int:
        count = sum(1 for w in vocab if w in words)
        count += sum(1 for phrase in vocab if " " in phrase and phrase in text_lower)
        return count

    volatile_hits = _hits(_VOLATILE_WORDS)
    if volatile_hits > 0:
        return "Volatile"

    bullish_hits = _hits(_BULLISH_WORDS)
    bearish_hits = _hits(_BEARISH_WORDS)

    if bullish_hits == 0 and bearish_hits == 0:
        vader_result = _try_vader(headline_text)
        return vader_result or "Neutral"

    if bullish_hits > bearish_hits:
        return "Bullish"
    if bearish_hits > bullish_hits:
        return "Bearish"

    # Tied keyword counts -> let vader break it, else call it Volatile
    # (mixed strongly-worded headline is itself a risk signal).
    vader_result = _try_vader(headline_text)
    return vader_result or "Volatile"


# --------------------------------------------------------------------------
# 4. Emergency killswitch
# --------------------------------------------------------------------------

_EMERGENCY_PATTERNS = (
    "flash crash", "circuit breaker", "trading halt", "trading halted",
    "market halted", "emergency rate", "emergency meeting", "intervention",
    "pegs abandoned", "depeg", "de-peg", "bank run", "liquidity crisis",
    "capital controls", "market meltdown", "black swan", "war declared",
    "missile strike", "invasion", "coup", "default imminent", "credit downgrade",
    "sovereign default", "circuit-breaker", "suspends trading",
)


async def _fetch_breaking_headlines() -> list[str]:
    now_ts = time.time()
    if (now_ts - _news_cache["fetched_at"]) < NEWS_CACHE_TTL_SECONDS:
        return _news_cache["headlines"]

    headlines: list[str] = []

    async with aiohttp.ClientSession() as session:
        if NEWSAPI_API_KEY:
            params = {
                "category": "business",
                "language": "en",
                "pageSize": 20,
                "apiKey": NEWSAPI_API_KEY,
            }
            data = await _fetch_json(session, NEWSAPI_HEADLINES_URL, params)
            if isinstance(data, dict):
                headlines = [a.get("title", "") for a in data.get("articles", []) if a.get("title")]

        if not headlines and FINNHUB_API_KEY:
            params = {"category": "forex", "token": FINNHUB_API_KEY}
            data = await _fetch_json(session, FINNHUB_NEWS_URL, params)
            if isinstance(data, list):
                headlines = [item.get("headline", "") for item in data if item.get("headline")]

    if not NEWSAPI_API_KEY and not FINNHUB_API_KEY:
        logger.warning("news_engine: no NEWSAPI_API_KEY or FINNHUB_API_KEY set, killswitch scan skipped")

    _news_cache["headlines"] = headlines
    _news_cache["fetched_at"] = now_ts
    return headlines


async def check_emergency_killswitch() -> KillswitchStatus:
    """
    Scans the latest breaking business/forex headlines for shock language
    (flash crash, circuit breaker, intervention, war/invasion, sovereign
    default, etc.) or a headline whose sentiment resolves to "Volatile".

    Sets the module-level GLOBAL_TRADING_HALTED flag so other modules can
    do a cheap synchronous check between full async scans, e.g.:

        import news_engine
        if news_engine.GLOBAL_TRADING_HALTED:
            return  # skip placing new orders this tick

    Returns a KillswitchStatus with the reason and matched headline so
    main.py can post a clear alert to the ops/alerts Discord channel.
    """
    global GLOBAL_TRADING_HALTED

    headlines = await _fetch_breaking_headlines()
    if not headlines:
        GLOBAL_TRADING_HALTED = False
        return KillswitchStatus(triggered=False, reason="No headline data available")

    for headline in headlines:
        headline_lower = headline.lower()

        matched_pattern = next((p for p in _EMERGENCY_PATTERNS if p in headline_lower), None)
        if matched_pattern:
            GLOBAL_TRADING_HALTED = True
            return KillswitchStatus(
                triggered=True,
                reason=f"Emergency keyword match: '{matched_pattern}'",
                matched_headline=headline,
            )

        if analyze_headline_sentiment(headline) == "Volatile":
            GLOBAL_TRADING_HALTED = True
            return KillswitchStatus(
                triggered=True,
                reason="Headline sentiment flagged as Volatile",
                matched_headline=headline,
            )

    GLOBAL_TRADING_HALTED = False
    return KillswitchStatus(triggered=False, reason="No emergency signals in latest headlines")


# --------------------------------------------------------------------------
# 5. Phase 4: USD calendar, pre-news / post-release alerts, AI context
# --------------------------------------------------------------------------

# Realistic desktop browser User-Agents (Chrome/Firefox/Edge/Safari across
# Windows/macOS). Generic hygiene against blanket CDN/WAF bot rules that
# reject default HTTP-client UA strings - NOT an auth mechanism. It will not
# bypass a provider's real API-key check or rate limit (see
# _fetch_jblanked_calendar's docstring).
_BROWSER_USER_AGENTS = [
    # Chrome / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Edge / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Safari / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


def _get_random_headers() -> dict:
    """
    Standard-looking browser headers with a randomized desktop User-Agent.

    This softens generic bot-filtering at the CDN/WAF layer (some providers
    blanket-reject default `python-requests`/`aiohttp` UA strings regardless
    of the caller's legitimacy). It is NOT an authentication mechanism: a
    403 caused by a missing/invalid API key or an exceeded rate limit will
    still be a 403 no matter which User-Agent is attached.
    """
    return {
        "User-Agent": random.choice(_BROWSER_USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }


async def _fetch_jblanked_calendar() -> list[EconomicEvent]:
    """
    Primary calendar source: JBlanked's MQL5-sourced "today" endpoint
    (https://www.jblanked.com/news/api/docs/calendar/).

    Requires JBLANKED_API_KEY (see https://www.jblanked.com/api/key/) sent
    as `Authorization: Api-Key ...` - that header is what actually
    authenticates the request; _get_random_headers() only adds realistic
    browser-style headers alongside it. Never raises: any missing key,
    timeout, non-200 response, rate limit, or malformed payload results in
    an empty list so fetch_usd_economic_calendar() falls back to Finnhub.

    Cached for JBLANKED_CACHE_TTL_SECONDS (1 hour, isolated from the
    Finnhub calendar's own cache) in _JBLANKED_CACHE / _JBLANKED_LAST_FETCH_TIME.
    The cache is updated after every real network attempt - success or
    failure - so a rate limit, an outage, or an exhausted free-tier quota
    doesn't get hit again every 60-second scan cycle; it's retried once the
    hour elapses. A missing JBLANKED_API_KEY does not touch the cache, so
    setting the key mid-run takes effect on the very next call.
    """
    global _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME

    now_ts = time.time()
    if (now_ts - _JBLANKED_LAST_FETCH_TIME) < JBLANKED_CACHE_TTL_SECONDS:
        return _JBLANKED_CACHE

    if not JBLANKED_API_KEY:
        logger.warning("news_engine: JBLANKED_API_KEY not set, skipping JBlanked calendar fetch")
        return []

    headers = _get_random_headers()
    headers["Authorization"] = f"Api-Key {JBLANKED_API_KEY}"
    headers["Content-Type"] = "application/json"

    params = {"currency": "USD", "offset": JBLANKED_UTC_OFFSET_PARAM}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                JBLANKED_CALENDAR_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT
            ) as resp:
                if resp.status == 429:
                    logger.warning("news_engine: JBlanked calendar rate-limited (HTTP 429)")
                    _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
                    return []
                if resp.status in (401, 403):
                    logger.warning(
                        "news_engine: JBlanked calendar auth rejected (HTTP %s) - check JBLANKED_API_KEY",
                        resp.status,
                    )
                    _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
                    return []
                if resp.status != 200:
                    logger.warning("news_engine: JBlanked calendar returned HTTP %s", resp.status)
                    _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
                    return []
                data = await resp.json(content_type=None)
    except asyncio.TimeoutError:
        logger.warning("news_engine: JBlanked calendar request timed out")
        _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
        return []
    except aiohttp.ClientError as exc:
        logger.warning("news_engine: JBlanked calendar request failed: %s", exc)
        _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
        return []
    except Exception:
        logger.exception("news_engine: unexpected error fetching JBlanked calendar")
        _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
        return []

    if not isinstance(data, list):
        logger.warning("news_engine: JBlanked calendar returned an unexpected payload shape")
        _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = [], now_ts
        return []

    events: list[EconomicEvent] = []
    for raw in data:
        try:
            currency = raw.get("Currency", "")
            when_str = raw.get("Date")
            if not when_str:
                continue
            # JBlanked format: "2024.02.08 15:30:00", already UTC via offset=3.
            event_time = datetime.strptime(when_str, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)

            event_name = raw.get("Name", "Unknown Event")
            raw_impact = (raw.get("Impact") or "").strip().lower()
            impact = raw_impact if raw_impact in ("high", "medium", "low") else "low"

            events.append(EconomicEvent(
                event_time=event_time,
                country=currency,
                event_name=event_name,
                impact=impact,
                actual=raw.get("Actual"),
                forecast=raw.get("Forecast"),
                previous=raw.get("Previous"),
            ))
        except (ValueError, TypeError) as exc:
            logger.debug("news_engine: skipping malformed JBlanked calendar row: %s (%s)", raw, exc)
            continue

    events.sort(key=lambda e: e.event_time)
    _JBLANKED_CACHE, _JBLANKED_LAST_FETCH_TIME = events, now_ts
    return events


async def _fetch_tradingeconomics_calendar() -> list[EconomicEvent]:
    """
    Third-tier free fallback: TradingEconomics' public "guest:guest" demo
    calendar, filtered to United States events. No signup/API key needed,
    but the guest credentials are heavily rate-limited and meant for
    evaluation — this is only ever reached after JBlanked and Finnhub have
    both already failed/come back empty (see fetch_usd_economic_calendar).

    Cached for TRADINGECONOMICS_CACHE_TTL_SECONDS. Never raises: any
    missing/malformed response results in an empty list.
    """
    global _TE_CACHE, _TE_LAST_FETCH_TIME

    now_ts = time.time()
    if (now_ts - _TE_LAST_FETCH_TIME) < TRADINGECONOMICS_CACHE_TTL_SECONDS:
        return _TE_CACHE

    params = {"c": TRADINGECONOMICS_GUEST_CREDENTIALS, "f": "json"}

    try:
        async with aiohttp.ClientSession() as session:
            data = await _fetch_json(session, TRADINGECONOMICS_CALENDAR_URL, params)
    except Exception:
        logger.exception("news_engine: unexpected error fetching TradingEconomics calendar")
        _TE_CACHE, _TE_LAST_FETCH_TIME = [], now_ts
        return []

    if not isinstance(data, list):
        logger.warning("news_engine: TradingEconomics calendar returned an unexpected payload shape")
        _TE_CACHE, _TE_LAST_FETCH_TIME = [], now_ts
        return []

    events: list[EconomicEvent] = []
    for raw in data:
        try:
            when_str = raw.get("Date")
            if not when_str:
                continue
            # TradingEconomics timestamps are ISO 8601 UTC, e.g.
            # "2024-02-08T13:30:00".
            event_time = datetime.fromisoformat(when_str.replace("Z", "+00:00"))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            else:
                event_time = event_time.astimezone(timezone.utc)

            event_name = raw.get("Event", "Unknown Event")
            importance = raw.get("Importance")
            impact = {2: "high", 1: "medium", 0: "low"}.get(importance, "low")

            events.append(EconomicEvent(
                event_time=event_time,
                country="USD",  # queried by country=united states, so fixed
                event_name=event_name,
                impact=impact,
                actual=raw.get("Actual"),
                forecast=raw.get("Forecast"),
                previous=raw.get("Previous"),
            ))
        except (ValueError, TypeError) as exc:
            logger.debug("news_engine: skipping malformed TradingEconomics calendar row: %s (%s)", raw, exc)
            continue

    events.sort(key=lambda e: e.event_time)
    _TE_CACHE, _TE_LAST_FETCH_TIME = events, now_ts
    return events


async def fetch_usd_economic_calendar(force_refresh: bool = False) -> list[EconomicEvent]:
    """
    Return today's USD economic calendar events (event_name, date_time_utc,
    impact, forecast, previous, actual - see EconomicEvent.to_dict()).

    Three-tier, same return shape either way:
      1. Try JBlanked (_fetch_jblanked_calendar) first.
      2. If JBlanked raises, 401/403/429s, or comes back empty, log a
         warning and fall back to the Finnhub-backed check_economic_calendar().
      3. If Finnhub also comes back empty, fall back to the free
         TradingEconomics guest calendar (_fetch_tradingeconomics_calendar).
      4. If all three fail, return an empty list rather than raising.
    """
    today = datetime.now(timezone.utc).date()

    try:
        jblanked_events = await _fetch_jblanked_calendar()
    except Exception:
        logger.exception("fetch_usd_economic_calendar: JBlanked primary fetch raised unexpectedly")
        jblanked_events = []

    if jblanked_events:
        matched = [e for e in jblanked_events if e.country == "USD" and e.event_time.date() == today]
        if matched:
            return matched

    logger.warning("news_engine: JBlanked primary returned no usable data, falling back to Finnhub")

    try:
        finnhub_events = await check_economic_calendar(force_refresh=force_refresh)
    except Exception:
        logger.exception("fetch_usd_economic_calendar: Finnhub fallback failed")
        finnhub_events = []

    matched = [e for e in finnhub_events if e.country == "USD" and e.event_time.date() == today]
    if matched:
        return matched

    logger.warning("news_engine: Finnhub fallback returned no usable data, trying TradingEconomics guest calendar")

    try:
        te_events = await _fetch_tradingeconomics_calendar()
    except Exception:
        logger.exception("fetch_usd_economic_calendar: TradingEconomics fallback also failed, returning empty list")
        return []

    return [e for e in te_events if e.country == "USD" and e.event_time.date() == today]


async def get_upcoming_high_impact_usd_events(within_minutes: int = 60) -> list[EconomicEvent]:
    """
    HIGH-impact USD events scheduled between now and `within_minutes` from
    now. Used for pre-news warnings (main.py) and killswitch-adjacent risk
    checks.
    """
    try:
        events = await fetch_usd_economic_calendar()
    except Exception:
        logger.exception("get_upcoming_high_impact_usd_events: calendar fetch failed")
        return []

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(minutes=within_minutes)
    return [e for e in events if e.is_high_impact() and now <= e.event_time <= horizon]


def _event_alert_key(event: EconomicEvent) -> str:
    return f"{event.event_name}|{event.event_time.isoformat()}"


async def check_pending_news_alerts() -> list[NewsAlert]:
    """
    Detect two things among today's HIGH-impact USD events:
      1. "pre_warning"  - event fires within PRE_NEWS_WARNING_MINUTES.
      2. "post_release" - event released within POST_RELEASE_WINDOW_MINUTES
                           and now has an `actual` value.

    A simple in-memory key set (_alerted_pre_warnings / _alerted_post_releases)
    prevents duplicate alerts for the same event across repeated calls (e.g.
    main.py's 60-second background loop). Never raises - a network/parse
    failure just yields an empty list so the caller's loop keeps ticking.
    """
    alerts: list[NewsAlert] = []

    try:
        events = await fetch_usd_economic_calendar()
    except Exception:
        logger.exception("check_pending_news_alerts: unexpected failure, returning no alerts")
        return alerts

    now = datetime.now(timezone.utc)

    for event in events:
        if not event.is_high_impact():
            continue

        key = _event_alert_key(event)
        minutes_until = (event.event_time - now).total_seconds() / 60.0

        if 0 <= minutes_until <= PRE_NEWS_WARNING_MINUTES and key not in _alerted_pre_warnings:
            _alerted_pre_warnings.add(key)
            alerts.append(NewsAlert(kind="pre_warning", event=event, minutes_until=minutes_until))

        minutes_since_release = -minutes_until  # positive once event_time is in the past
        has_actual = event.actual not in (None, "")
        if has_actual and 0 <= minutes_since_release <= POST_RELEASE_WINDOW_MINUTES and key not in _alerted_post_releases:
            _alerted_post_releases.add(key)
            alerts.append(NewsAlert(kind="post_release", event=event, minutes_until=-minutes_since_release))

    # Cheap unbounded-growth guard for long-running processes; today's keys
    # are all that matter, so a periodic full clear is safe.
    if len(_alerted_pre_warnings) > 500:
        _alerted_pre_warnings.clear()
    if len(_alerted_post_releases) > 500:
        _alerted_post_releases.clear()

    return alerts


async def get_institutional_news_context() -> dict:
    """
    Single summary dict for AI-prompt injection (ai_engine.py) or a Discord
    dashboard command. Every sub-check is individually try/excepted so one
    failing data source degrades gracefully instead of blanking the whole
    context.
    """
    try:
        killswitch = await check_emergency_killswitch()
    except Exception:
        logger.exception("get_institutional_news_context: killswitch check failed")
        killswitch = KillswitchStatus(triggered=False, reason="Killswitch check unavailable")

    try:
        freeze_active, freeze_event = await is_news_freeze_active()
    except Exception:
        logger.exception("get_institutional_news_context: freeze check failed")
        freeze_active, freeze_event = False, None

    try:
        usd_events = await fetch_usd_economic_calendar()
    except Exception:
        logger.exception("get_institutional_news_context: calendar fetch failed")
        usd_events = []

    now = datetime.now(timezone.utc)

    upcoming = sorted((e for e in usd_events if e.event_time >= now), key=lambda e: e.event_time)
    upcoming_usd_events_today = [
        {
            "time": e.event_time.isoformat(),
            "event": e.event_name,
            "impact": e.impact.upper(),
            "forecast": e.forecast,
            "previous": e.previous,
        }
        for e in upcoming
    ]

    next_major_event = None
    for e in upcoming:
        if e.is_high_impact():
            next_major_event = {
                "event": e.event_name,
                "time": e.event_time.isoformat(),
                "minutes_remaining": round((e.event_time - now).total_seconds() / 60.0, 1),
                "impact": e.impact.upper(),
                "forecast": e.forecast,
                "previous": e.previous,
            }
            break

    recent_cutoff = now - timedelta(minutes=POST_RELEASE_WINDOW_MINUTES)
    recent_releases = [
        {
            "event": e.event_name,
            "time": e.event_time.isoformat(),
            "impact": e.impact.upper(),
            "actual": e.actual,
            "forecast": e.forecast,
            "previous": e.previous,
        }
        for e in usd_events
        if e.actual not in (None, "") and recent_cutoff <= e.event_time <= now
    ]

    return {
        "active_killswitch": killswitch.triggered,
        "active_news_freeze": freeze_active,
        "upcoming_usd_events_today": upcoming_usd_events_today,
        "next_major_event": next_major_event,
        "recent_releases": recent_releases,
    }


# --------------------------------------------------------------------------
# Manual smoke test
# --------------------------------------------------------------------------

async def _demo():
    logging.basicConfig(level=logging.INFO)

    print("-- Economic calendar --")
    events = await check_economic_calendar()
    for e in events[:5]:
        print(f"  {e.event_time.isoformat()} [{e.impact.upper()}] {e.country} - {e.event_name}")

    frozen, ev = await is_news_freeze_active()
    print(f"\n-- News freeze active: {frozen}", f"({ev.event_name})" if ev else "")

    samples = [
        "US NFP smashes expectations, dollar rallies on strong jobs growth",
        "ECB stuns markets with surprise emergency rate decision",
        "GBP slumps after weak retail sales miss forecasts",
        "Traders await tomorrow's inflation report",
    ]
    print("\n-- Sentiment samples --")
    for s in samples:
        print(f"  [{analyze_headline_sentiment(s)}] {s}")

    print("\n-- Killswitch check --")
    status = await check_emergency_killswitch()
    print(f"  Triggered: {status.triggered} | Reason: {status.reason}")

    print("\n-- Phase 4: USD calendar (dual-engine: JBlanked -> Finnhub) --")
    print(f"  JBLANKED_API_KEY set: {bool(JBLANKED_API_KEY)}")
    print(f"  JBlanked cache age (s): {time.time() - _JBLANKED_LAST_FETCH_TIME:.0f} "
          f"(TTL {JBLANKED_CACHE_TTL_SECONDS}s)")
    usd_events = await fetch_usd_economic_calendar()
    for e in usd_events[:5]:
        print(f"  {e.event_time.isoformat()} [{e.impact.upper()}] {e.event_name} "
              f"(F:{e.forecast} P:{e.previous} A:{e.actual})")

    print("\n-- Phase 4: upcoming high-impact USD events (next 60 min) --")
    upcoming = await get_upcoming_high_impact_usd_events(within_minutes=60)
    for e in upcoming:
        print(f"  {e.event_time.isoformat()} - {e.event_name}")
    if not upcoming:
        print("  (none in the next 60 minutes)")

    print("\n-- Phase 4: pending news alerts --")
    alerts = await check_pending_news_alerts()
    for a in alerts:
        print(f"  [{a.kind}] {a.event.event_name} (minutes_until={a.minutes_until})")
    if not alerts:
        print("  (none pending)")

    print("\n-- Phase 4: institutional news context --")
    context = await get_institutional_news_context()
    print(f"  active_killswitch: {context['active_killswitch']}")
    print(f"  active_news_freeze: {context['active_news_freeze']}")
    print(f"  upcoming_usd_events_today: {len(context['upcoming_usd_events_today'])} event(s)")
    print(f"  next_major_event: {context['next_major_event']}")
    print(f"  recent_releases: {len(context['recent_releases'])} event(s)")


if __name__ == "__main__":
    asyncio.run(_demo())