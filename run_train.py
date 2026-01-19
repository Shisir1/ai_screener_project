from __future__ import annotations
import joblib
import pandas as pd
from config import CFG
from data import download_ohlcv, to_long
from features import add_features
from model import make_labels, train_lgbm

from dotenv import load_dotenv
load_dotenv()


def main():
    tickers = sorted(set(CFG.tickers + [CFG.benchmark]))
    raw = download_ohlcv(tickers=tickers, start=CFG.start, interval=CFG.interval)
    long = to_long(raw)

    feat = add_features(long, benchmark_ticker=CFG.benchmark)
    labeled = make_labels(feat, benchmark_ticker=CFG.benchmark, horizon_days=CFG.horizon_days)

    model = train_lgbm(labeled, random_state=CFG.random_state)
    print(f"Trained model. CV RMSE (mean): {getattr(model, 'cv_rmse_mean_', None): .6f}")

    joblib.dump(model, "model.joblib")

    #store latest dataset for daily scoring 
    labeled.to_parquet("dataset.parquet", index=False)
    print("Saved model.joblib and dataset.parquet!")

if __name__ == "__main__":
    main()