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
    """
    Simple time-series CV to sanity-check, then train on full data.
    """
    d = df.dropna(subset=FEATURE_COLS + ["y"]).copy()
    #Use numeric ticker encoding as a feature 
    d["ticker_id"] = d["ticker"].astype("category").cat.codes
    feats = FEATURE_COLS + ["ticker_id"]

    X = d[feats]
    y = d["y"]

    tscv = TimeSeriesSplit(n_splits=5)
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
        rmse = mean_squared_error(yte, pred, squared=False)
        rmses.append(rmse)

    #Train final model
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
    model.feature_name_ = feats
    model.ticker_categories_ = d["ticker"].astype("category").cat.categories.tolist()
    return model

def predict_scores(model: lgb.LGBMRegressor, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset= FEATURE_COLS).copy()
    # reproduce tiker_id mapping
    cats = model.ticker_categories_
    out["ticker_id"] = pd.Categorical(out["ticker]"], categories=cats).codes
    feats = model.feature_name_
    out["ml_score"] = model.predict(out[feats])
    return out