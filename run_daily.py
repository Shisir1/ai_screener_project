from __future__ import annotations
import os
import joblib
import pandas as pd
from config import CFG
from data import download_ohlcv, to_long
from features import add_features
from strategy import rank_for_date, compute_exits
from alerts import send_email
from model import predict_scores
from dotenv import load_dotenv
load_dotenv()

def format_reco_email(asof_date:pd.Timestamp, buys: pd.DataFrame, sells: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Date: {asof_date.date()}")
    lines.append("")
    lines.append("BUYS / TARGETS (Top Ranked):")

    if buys.empty:
        lines.append("  None")
    else:
        for _, r in buys.iterrows():
            lines.append(f"  {r['ticker']:<6}: Close={r['close']:.2f}, FinalScore={r['final_score']:.3f}")

    
    lines.append("")
    lines.append("SELLS / EXITS (Exit triggers):")

    if sells.empty:
        lines.append("  None")
    else:
        for _, r in sells.iterrows():
            lines.append(f"  {r['ticker']:<6}: Reason={r['reason']}")
    return "\n".join(lines)

def main():
    if not os.path.exists("model.joblib"):
        raise RuntimeError("model.joblib not found. Run: python run_train.py first!")

    model = joblib.load("model.joblib")

    tickers = sorted(set(CFG.tickers + [CFG.benchmark]))
    raw = download_ohlcv(tickers=tickers, start=CFG.start, interval=CFG.interval)
    long = to_long(raw)
    feat = add_features(long, benchmark_ticker=CFG.benchmark)

    scored = predict_scores(model, feat)

    asof_date = scored["date"].max()
    #rank candidates today
    top = rank_for_date(
        scored,
        asof_date=asof_date,
        top_k=CFG.top_k,
        min_price=CFG.min_price,
        min_avg_dollar_vol=CFG.min_avg_dollar_vol
    )

    # ---- "Held positions" mock ----
    # Replace this with your real portfolio state from a DB/broker.
    # Example: last week's top K = current holdings.
    # For MVP: assume you held previous top K.\

    prev_date = scored["date"].sort_values().unique()[-6] # 5 trading days ago
    prev_top = rank_for_date(
        scored,
        asof_date=pd.Timestamp(prev_date),
        top_k=CFG.top_k,
        min_price=CFG.min_price,
        min_avg_dollar_vol=CFG.min_avg_dollar_vol
    )
    held = prev_top["ticker"].tolist()

    #Exits based on tred breaks
    exits = compute_exits(
        full_df=feat,
        asof_date=asof_date,
        held_tickers=held,
        sma_fast=CFG.sma_fast,
        trend_exit_days=CFG.trend_exit_days
    )

    #exit also if it falls out of top-K now
    now_set = set(top["ticker"].tolist())
    out_of_top = [t for t in held if t not in now_set and t != CFG.benchmark]
    out_df = pd.DataFrame([{"date": asof_date, "ticker": t, "reason": "Fell out of top-K"} for t in out_of_top])
    sells = pd.concat([exits, out_df], ignore_index=True).drop_duplicates(subset=["ticker", "reason"])

    #Buy candidates are current top-K minus held tickers
    buy_tickers = [t for t in top["ticker"].tolist() if t not in held and t != CFG.benchmark]
    buys = top[top["ticker"].isin(buy_tickers)].copy()

    body = format_reco_email(asof_date, buys, sells)
    print(body)

    #email alert
    to_email = os.getenv("ALERT_TO_EMAIL")
    if to_email:
        send_email(
            subject=f"Daily Stock Recs for {asof_date.date()}",
            body=body,
            to_email=to_email
        )
        print(f"Sent email alert to {to_email}")
    else:
        print("ALERT_TO_EMAIL not set. Skipping email alert.")


if __name__ == "__main__":
    main()
