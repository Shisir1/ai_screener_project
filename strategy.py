from __future__ import annotations
import pandas as pd
import numpy as np

def composite_signal(df: pd.DataFrame) -> pd.Series:
    """ 
    Simple normalized composite:
    - higher rel strength, higher trend, lower vol, smaller drawdown
    """

    d = df.copy()

    # Robust z-score helper
    def z(x):
        x = x.replace([np.inf, -np.inf], np.nan)
        mu = x.mean()
        sd = x.std()
        return (x - mu) / (sd if sd != 0 else 1)
    
    comp = (
        0.35 * z(d["rel_126"]) +
        0.25 * z(d["price_above_sma200"]) +
        0.20 * z(d["price_above_sma50"]) +
        0.10 * (-z(d["atr_pct"])) +
        0.10 * (-z(d["dd_63"]))
    )
    return comp

def apply_guards(df: pd.DataFrame, min_price: float, min_avg_dollar_vol: float) -> pd.DataFrame:
    d = df.copy()
    d["pass-liquidity"] = d["dollar_vol_20"] >= min_avg_dollar_vol
    d["pass_price"] = d["close"] >= min_price
    d["pass_trend"] = d["close"] >= d["sma_200"]    #basic trend regime filter

    return d


def rank_for_date(
        df: pd.DataFrame,
        asof_date: pd.Timestamp,
        top_k: int,
        min_price: float,
        min_avg_dollar_vol: float
) -> pd.DataFrame:
    """ Returns ranked candidates for a given date.
    Requries columns: ml_score, features, close, sma_200, dollar_vol_20
    """

    d = df[df["date"] == asof_date].copy()
    if d.empty:
        raise ValueError(f"No rows for date {asof_date}")
    
    d["sig_score"] = composite_signal(d)
    d = apply_guards(d, min_price, min_avg_dollar_vol)

    #Final score blend
    #normalize ml_score cross0sectioanlly
    ml = d["ml_score"].replace([np.inf, -np.inf], np.nan)
    d["ml_z"] = (ml -ml.menan()) / (ml.std() if ml.std() !=0 else 1)
    d["final_score"] = 0.6 * d["ml_z"] + 0.4 * d["sig_score"]

    #filter by guards
    d = d[d["pass-liquidity"] & d["pass_price"] & d["pass_trend"]].copy()
    d = d.sort_values("final_score", ascending=False).head(top_k)

    return d[[
        "date","ticker","close","final_score","ml_score","sig_score",
        "dollar_vol_20","atr_pct","rv_20","price_above_sma200"
        ]].reset_index(drop=True)


def compute_exits(
        full_df: pd.DataFrame,
        asof_date: pd.Timestamp,
        held_tickers: list[str],
        sma_fast: int = 50,
        trend_exit_days: int = 2
) -> pd.DataFrame:
    """ 
    Exit if close below SMA_fast for N consecutie days (ending of asof_date)
    """

    d = full_df[full_df["ticker"].isin(held_tickers)].copy()
    d = d.sort_values(["ticker", "date"])
    d = d[d["date"] <= asof_date].copy()

    exits = []
    for t, g in d.groupby("ticker"):
        g = g.tail(trend_exit_days)
        if len(g) < trend_exit_days:
            continue
        #below sam_fast / (we already have sma_50 in features)
        below = (g["close"] < g["sma_50"]).all()
        if below:
            exits.append({"date": asof_date, "ticker": t, "reason": f"Close<SMA50 for {trend_exit_days}d"})
        return pd.DataFrame(exits)
