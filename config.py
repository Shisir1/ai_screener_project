from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Config:
    #Universe
    tickers: List[str]
    benchmark: str = "SPY"

    #data
    start: str = "2015-01-01"
    interval: str = "1d"

    #labels
    horizon_days: int = 21  #no. of trading days to look ahead for labeling

    #Portfolio rules
    top_k: int = 10
    min_price: float = 5.0 #minimum price of stock to be considered for portfolio
    min_avg_dollar_vol: float = 20_000_000  #minimum average daily dollar volume

    #Trend filter/exit
    sma_fast: int = 50
    sma_slow: int = 200
    trend_exit_days: int = 2 #no. of days to confirm trend exit


    #Model
    random_state: int = 42

DEFAULT_TICKERS = [ # ETFs (examples)
    "SPY","QQQ","IWM","EFA","EEM","TLT","IEF","GLD","USO",
    # A few large-cap stocks (replace with your universe)
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","JPM","UNH","XOM",
    ]

CFG = Config(tickers=DEFAULT_TICKERS)