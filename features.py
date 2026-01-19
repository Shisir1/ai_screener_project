from __future__ import annotations
import pandas as pd
import numpy as np

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_features(long: pd.DataFrame, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    """
    Input long columns: date, ticker, open, high, low, close, volume
    Output: adds feature columns per ticker.
    """
    df = long.copy()
    df = df.sort_values(["ticker", "date"])

    # Per-ticker returns/features
    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ret_5"] = g["close"].pct_change(5)
        g["ret_21"] = g["close"].pct_change(21)
        g["ret_63"] = g["close"].pct_change(63)
        g["ret_126"] = g["close"].pct_change(126)
        g["ret_252"] = g["close"].pct_change(252)

        g["sma_50"] = _sma(g["close"], 50)
        g["sma_200"] = _sma(g["close"], 200)
        g["price_above_sma50"] = (g["close"] / g["sma_50"]) - 1.0
        g["price_above_sma200"] = (g["close"] / g["sma_200"]) - 1.0

        g["rsi_14"] = _rsi(g["close"], 14)
        g["atr_14"] = _atr(g["high"], g["low"], g["close"], 14)
        g["atr_pct"] = g["atr_14"] / g["close"]

        g["rv_20"] = g["close"].pct_change().rolling(20).std() * np.sqrt(252)
        g["dollar_vol_20"] = (g["close"] * g["volume"]).rolling(20).mean()

        # Drawdown over 63d
        roll_max = g["close"].rolling(63).max()
        g["dd_63"] = (g["close"] / roll_max) - 1.0

        return g

    df = (
        df.sort_values(["ticker", "date"])
            .groupby("ticker", group_keys=False)
            .apply(lambda g: per_ticker(g.reset_index(drop=True)))
    )

    # Relative strength vs benchmark (same dates)
    bench = df[df["ticker"] == benchmark_ticker][["date", "close"]].rename(columns={"close":"bench_close"})
    df = df.merge(bench, on="date", how="left")
    df["rel_63"] = (df["close"].pct_change(63) - df["bench_close"].pct_change(63))
    df["rel_126"] = (df["close"].pct_change(126) - df["bench_close"].pct_change(126))
    
    return df