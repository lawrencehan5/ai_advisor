"""
ai_advisor/price_cache.py

Local parquet cache for yfinance price data.

On the first run, the full 3-year history for every ticker in the approved
universe (plus SPY as the market-tracking benchmark) is downloaded and saved
to  <project_root>/data/price_cache.parquet.

On subsequent runs:
  - If the cache's last date is yesterday or earlier, only the missing trading
    days are fetched and appended — a typical incremental update takes a few
    seconds instead of a full re-download.
  - Any ticker added to the universe after the initial download is detected
    and its full history is fetched on the next run.

Public API
----------
load_prices(tickers)  →  pd.DataFrame   (columns = requested tickers, index = dates)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Paths ────────────────────────────────────────────────────────────────────
# optimizer.py lives at  src/ai_advisor/optimizer.py
# project root is three levels up:  src/ai_advisor/ → src/ → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR    = _PROJECT_ROOT / "data"
_CACHE_FILE   = _CACHE_DIR / "price_cache.parquet"

_YEARS_HISTORY = 3  # how many years of history to fetch on a fresh download


# ── Universe ─────────────────────────────────────────────────────────────────

def _universe_tickers() -> list[str]:
    """Full approved universe + SPY (needed for market-tracking benchmark)."""
    from ai_advisor.stocks import get_all_tickers
    tickers = get_all_tickers()
    if "SPY" not in tickers:
        tickers = tickers + ["SPY"]
    return tickers


# ── Public entry point ───────────────────────────────────────────────────────

def load_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted-close prices for the requested tickers.

    Data is served from the local parquet cache.  The cache is refreshed
    incrementally (missing days only) whenever it is out of date.

    Args:
        tickers: List of ticker symbols to return.

    Returns:
        DataFrame with dates as index and tickers as columns.
        Tickers not available in the cache (bad symbols, zero history) are
        silently dropped from the result.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache = _read_cache()
    universe = _universe_tickers()
    today = date.today()

    if cache is None:
        # ── First run: full download ─────────────────────────────────────
        print(
            f"  [cache] No cache found — downloading {_YEARS_HISTORY}y history "
            f"for {len(universe)} tickers…"
        )
        cache = _fetch_full(universe)
        _write_cache(cache)
        print(f"  [cache] Cache saved → {_CACHE_FILE}  ({len(cache)} rows, {len(cache.columns)} tickers)")

    else:
        # ── Check for new tickers added to the universe ──────────────────
        missing_tickers = [t for t in universe if t not in cache.columns]
        if missing_tickers:
            print(f"  [cache] New universe tickers: {missing_tickers} — fetching full history…")
            new_data = _fetch_full(missing_tickers)
            cache = cache.join(new_data, how="outer")
            _write_cache(cache)

        # ── Incremental update if stale ──────────────────────────────────
        last_cached: date = cache.index[-1].date()

        # Stale = last cached date is older than yesterday.
        # (Today's close is only available after market hours; fetching up to
        # `today` is safe — yfinance returns whatever the latest available
        # close is, which may be yesterday or a prior business day.)
        if last_cached < today - timedelta(days=1):
            fetch_start = last_cached + timedelta(days=1)
            print(
                f"  [cache] Stale (last: {last_cached}) — fetching "
                f"{fetch_start} → {today} for {len(universe)} tickers…"
            )
            new_rows = yf.download(
                universe,
                start=fetch_start,
                end=today,
                auto_adjust=True,
                progress=False,
            )["Close"]
            if isinstance(new_rows, pd.Series):
                new_rows = new_rows.to_frame()

            if not new_rows.empty:
                combined = pd.concat([cache, new_rows])
                combined = combined[~combined.index.duplicated(keep="last")]
                cache = combined.sort_index()
                _write_cache(cache)
                print(
                    f"  [cache] Updated: +{len(new_rows)} row(s), "
                    f"cache now covers {cache.index[0].date()} → {cache.index[-1].date()}"
                )
            else:
                print(f"  [cache] No new rows returned (holiday / weekend / market still open).")
        else:
            print(f"  [cache] Cache is current (last: {last_cached}).")

    # ── Return the requested subset ──────────────────────────────────────────
    available = [t for t in tickers if t in cache.columns]
    dropped   = [t for t in tickers if t not in cache.columns]
    if dropped:
        print(f"  [cache] Tickers not in cache (dropping): {dropped}")

    return cache[available]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_full(tickers: list[str]) -> pd.DataFrame:
    """
    Download up to 20 years of adjusted-close prices for a list of tickers.
    Tickers with less than 3 years of data (recent IPOs / ETF launches) are
    retried on a 3-year window so they still appear in the result.
    Tickers with 3–19 years of history (e.g. TSLA, VOO) are returned as-is
    with NaN before their listing date — the optimizer aligns on the inner
    join of available dates when building the returns matrix.
    """
    end   = date.today()
    start = end - timedelta(days=_YEARS_HISTORY * 365)
    start_3y = end - timedelta(days=3 * 365)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    # Tickers with fewer than 3 years of actual rows get a shorter window retry
    min_rows  = 3 * 252
    sufficient    = [col for col in raw.columns if raw[col].notna().sum() >= min_rows]
    short_history = [col for col in raw.columns if col not in sufficient]

    if short_history:
        print(f"  [cache] {short_history} have < 3y data — retrying on 3y window")
        raw_3y = yf.download(
            short_history, start=start_3y, end=end, auto_adjust=True, progress=False
        )["Close"]
        if isinstance(raw_3y, pd.Series):
            raw_3y = raw_3y.to_frame()
        raw = raw[sufficient].join(raw_3y, how="outer")

    raw = raw.dropna(axis=1, how="all")

    failed = [t for t in tickers if t not in raw.columns]
    if failed:
        print(f"  [cache] Could not fetch data for: {failed} — excluded from cache")

    return raw


def _read_cache() -> pd.DataFrame | None:
    """Load the parquet cache, or return None if it does not exist / is corrupt."""
    if not _CACHE_FILE.exists():
        return None
    try:
        df = pd.read_parquet(_CACHE_FILE)
        if df.empty:
            return None
        return df
    except Exception as exc:
        print(f"  [cache] Failed to read cache ({exc}) — will rebuild from scratch.")
        return None


def _write_cache(df: pd.DataFrame) -> None:
    """Persist the price DataFrame to the parquet cache."""
    df.to_parquet(_CACHE_FILE)
