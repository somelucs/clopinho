import numpy as np
import yfinance as yf
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('seaborn-v0_8-darkgrid')

class MLBenchmark():

    def __init__(self, ticker="BOVA11.SA", start_date="2015-01-01", end_date="2025-09-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.results = None

    
    def _cagr(self, equity):

        N = len(equity)
        return equity.iloc[-1]**(252/N) - 1

    def _max_drawdown(self, equity):
        roll_max = equity.cummax()
        dd = equity / roll_max - 1
        return dd.min()

    def _vol_annual(self, ret):

        return ret.std() * np.sqrt(252)

    def _sharpe(self, ret, rf=0.0):

        vol = self._vol_annual(ret)
        if vol == 0:
            return 0
        return (ret.mean() * 252 - rf) / vol


    def _prepare_data_and_train_model(self):
       

        print("1. Baixando dados e preparando features...")
        # Baixa os dados
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Cria as features (indicadores)
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Cria a variável alvo
        self.df['Target'] = np.where(self.df['Close'].shift(-5) > self.df['Close'] * 1.005, 1, 0)
        
        # Limpa NaNs
        self.df.dropna(inplace=True)
        
        # Define X e y
        features = ['SMA_20', 'SMA_50', 'RSI']
        X = self.df[features]
        y = self.df['Target']
        
        print("2. Treinando o modelo XGBoost...")

        model = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            eta=0.06, 
            max_depth=3,
            n_estimators=1000,
            use_label_encoder=False
        )
        model.fit(X, y)
        
        model.save_model("xgboost.json")

        print("3. Gerando sinais de trading com o modelo...")

        self.df['Signal'] = model.predict(X)
        

    def run_backtest(self):
        
        self._prepare_data_and_train_model()
        
        print("4. Calculando performance das estratégias...")

        self.df['Ret_BH'] = self.df['Close'].pct_change()
        self.df['Equity_BH'] = (1 + self.df['Ret_BH']).cumprod()

        self.df['Ret_ML'] = self.df['Ret_BH'] * self.df['Signal'].shift(1)
        self.df['Equity_ML'] = (1 + self.df['Ret_ML']).cumprod()
        
        self.df.dropna(inplace=True)


        metrics = {
            'CAGR': [self._cagr(self.df['Equity_BH']), self._cagr(self.df['Equity_ML'])],
            'Max Drawdown': [self._max_drawdown(self.df['Equity_BH']), self._max_drawdown(self.df['Equity_ML'])],
            'Volatility': [self._vol_annual(self.df['Ret_BH']), self._vol_annual(self.df['Ret_ML'])],
            'Sharpe Ratio': [self._sharpe(self.df['Ret_BH']), self._sharpe(self.df['Ret_ML'])]
        }
        
        self.results = pd.DataFrame(metrics, index=['Buy and Hold', 'XGBoost Model'])
    
    def display_results(self):

        if self.results is None:
            print("Execute o backtest primeiro com o método .run_backtest()")
            return
            
        
        formatted_results = self.results.copy()
        formatted_results['CAGR'] = formatted_results['CAGR'].apply(lambda x: f"{x:.2%}")
        formatted_results['Max Drawdown'] = formatted_results['Max Drawdown'].apply(lambda x: f"{x:.2%}")
        formatted_results['Volatility'] = formatted_results['Volatility'].apply(lambda x: f"{x:.2%}")
        formatted_results['Sharpe Ratio'] = formatted_results['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        
        print(formatted_results)

    def plot_equity_curves(self):

        if self.df is None or 'Equity_BH' not in self.df:
            print("Execute o backtest primeiro com o método .run_backtest()")
            return
            
        plt.figure(figsize=(12, 7))
        self.df['Equity_BH'].plot(label='Buy and Hold', legend=True)
        self.df['Equity_ML'].plot(label='XGBoost Model', legend=True)
        plt.title('Curva de Patrimônio: Modelo XGBoost vs. Buy and Hold')
        plt.xlabel('Data')
        plt.ylabel('Patrimônio (Normalizado)')
        plt.show()

if __name__ == "__main__":
    benchmark = MLBenchmark()
    benchmark.run_backtest()
    benchmark.display_results()
    benchmark.plot_equity_curves()