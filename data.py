from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import List

def download_ohlcv(tickers: List[str], start: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for given tickers from Yahoo Finance.

    Parameters:
    tickers (List[str]): List of ticker symbols to download.
    start (str): Start date for the data in 'YYYY-MM-DD' format.
    interval (str): Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
    pd.DataFrame: Multi-index DataFrame with OHLCV data.
    """
    df = yf.download(
        tickers = tickers, 
        start=start, 
        interval=interval, 
        group_by='column', 
        auto_adjust=False,
        threads= True,
        progress=False)
    if df.emtpy:
        raise RuntimeError("No data returned from yfinance. Check the tickers and date range, internet or yfinance limits.")
    #Ensure consistent column structure: (Field, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        return df.sort_index()
    else:
        #Single ticker case -> convert to MultiIndex
        df.columns = pd.MultiIndex.from_product([df.columns, ["SINGLE"]])
        return df.sort_index()
    
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MultiIndex OHLCV dataframe to long format:
      date, ticker, open, high, low, close, adjclose?, volume
    """
    fields = [c[0] for c in df.columns]
    unique_fields = sorted(set(fields))
    out = []
    for field in unique_fields:
        if(field,) == ("Adj Close",):
            continue
        slice_ = df[field].copy()
        slice.columns.name = "ticker"
        slice_ = slice_.stack().rename(field.lower().replace(" ", ""))
        out.append(slice_)

        long = pd.concat(out, axis=1).reset_index()
        long = long.rename(columns={"Date": "date"})
        long["date"] = pd.to_datetime(long["date"])
        long = long.sort_values(["ticker", "date"])
    return long