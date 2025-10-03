import xgboost as xgb
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta


# TODO: Benchmark

class Benchmark():

    def __init__(self):

        self.df = yf.download("BOVA11.SA", start="2015-01-01", end="2025-09-01")
        self.df = self.df[['Close']].rename(columns={'Close':'AdjClose'})

        self.df['Ret_BH'] = self.df['AdjClose'].pct_change()

        self.df['Equity_BH'] = (1 + self.df['Ret_BH']).cumprod()

        # Métricas
        N = self.df['Ret_BH'].dropna().shape[0]
        years = N/252

    def max_drawdown(self,equity):
        self.roll_max = equity.cummax()
        self.dd = equity/self.roll_max - 1
        return self.dd.min()

    def cagr(self,equity):
        return equity.iloc[-1]**(252/N) - 1

    def vol_annual(self,ret):
        return ret.std()*np.sqrt(252)

    def sharpe(self, ret, rf=0.0):
        return (ret.mean()*252 - rf)/vol_annual(ret)

    # Benchmark
    self.cagr_bh   = cagr(df['Equity_BH'].dropna())
    self.mdd_bh    = max_drawdown(df['Equity_BH'].dropna())
    self.vol_bh    = vol_annual(df['Ret_BH'].dropna())
    self.shp_bh    = sharpe(df['Ret_BH'].dropna(), rf=0.0)

# TODO: Carregando e usando o modelo com o Benchmark

model = xgb.XGBClassifier()
model.load_model('xgboost.json')

