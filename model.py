from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

FEATURE_COLS = [
    "ret_5","ret_21","ret_63","ret_126","ret_252",
    "price_above_sma50","price_above_sma200",
    "rsi_14","atr_pct","rv_20","dollar_vol_20","dd_63",
    "rel_63","rel_126",
]

def make_labels(df: pd.DataFrame, benchmark_ticker: str, horizon_days: int) -> pd.DataFrame:
    """
    Label = forward_horizon_return(asset) - forward_horizon_return(benchmark)
    """

    out = df.copy()
    out = out.sort_values(["ticker", "date"])

    #forward return per ticker
    out["fwd_ret"] = out.groupby("ticker")["close"].shift(-horizon_days) / out["close"] - 1.0

    #forward return benchmark by date
    bench = out[out["ticker"]== benchmark_ticker][["date", "fwd_ret"]].rename(columns={"fwd_ret": "bench_fwd_ret"})
    out = out.merge(bench, on="date", how="left")

    out["y"] = out["fwd_ret"] - out["bench_fwd_ret"]
    return out

def train_lgbm(df: pd.DataFrame, random_state: int = 42) -> lgb.LGBMRegressor:
    # BEFORE dropping NA, show shape
    print("train_lgbm: incoming df rows =", len(df), "cols =", len(df.columns))

    # How many rows have each required column non-null?
    required = FEATURE_COLS + ["y", "ticker", "date", "close"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in df: {missing_cols}")

    na_rates = df[FEATURE_COLS + ["y"]].isna().mean().sort_values(ascending=False)
    print("\nTop NaN rates:")
    print(na_rates.head(12).to_string())

    d = df.dropna(subset=FEATURE_COLS + ["y"]).copy()
    print("\nRows after dropna =", len(d))

    if d.empty:
        # Extra diagnosis: is benchmark present? how many tickers?
        print("\nTicker counts (top 10):")
        print(df["ticker"].value_counts().head(10).to_string())

        # Show benchmark rows and bench_fwd_ret availability if present
        if "bench_fwd_ret" in df.columns:
            b = df[df["ticker"] == "SPY"].copy()
            print("\nBenchmark (SPY) rows:", len(b))
            if len(b) > 0:
                print("bench_fwd_ret NaN rate:", b["bench_fwd_ret"].isna().mean())
        else:
            print("\nNo bench_fwd_ret column found in df.")

        # Also: how much history do you have?
        print("\nDate range:", df["date"].min(), "->", df["date"].max())
        raise RuntimeError(
            "Training set is empty after dropna.\n"
            "This means ALL rows have NaN in at least one feature or y.\n"
            "See NaN rates printed above to identify which column is killing rows."
        )

    # Build model inputs
    d["ticker_id"] = d["ticker"].astype("category").cat.codes
    feats = FEATURE_COLS + ["ticker_id"]
    X = d[feats]
    y = d["y"]

    # Keep splits safe
    n_splits = 3
    if len(d) < 300:
        n_splits = 2

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        m = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rmse = mean_squared_error(yte, pred) ** 0.5
        rmses.append(rmse)


    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
    )
    model.fit(X, y)

    model.cv_rmse_mean_ = float(np.mean(rmses))
    model.feature_names_ = feats
    model.ticker_categories_ = d["ticker"].astype("category").cat.categories.tolist()
    return model

def predict_scores(model: lgb.LGBMRegressor, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset= FEATURE_COLS).copy()
    # reproduce tiker_id mapping
    cats = model.ticker_categories_
    out["ticker_id"] = pd.Categorical(out["ticker"], categories=cats).codes
    feats = model.feature_name_
    out["ml_score"] = model.predict(out[feats])
    return out