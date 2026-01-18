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