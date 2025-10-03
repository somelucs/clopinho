import numpy as np
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import date


#TODO: Classe dos dados e indicadores que serão usados no backtest
class Dados():
    def __init__(self):
       
        self.hoje=date.today().strftime("%d-%m-%Y")
	self.X=None
	self.y=None
        self.dataset_historico = yf.download("BOVA11.SA", start="2015-01-01", end="2025-09-01")
        self.dataset_historico = self.dataset_historico[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.dataset_com_data =  self.dataset_historico.reset_index()

        # Criar retornos diários
        self.dataset_historico['Return'] = self.dataset_historico['Close'].pct_change()


        self.dataset_atual= yf.download("BOVA11.SA", start="2015-09-01", end=self.hoje)
        self.dataset_atual = self.dataset_atual[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.dataset_com_data =  self.dataset_atual.reset_index()

    def indicadores_historico(self):
        self.dataset_historico['SMA_20'] = self.dataset_historico['Close'].rolling(20).mean()
        self.dataset_historico['SMA_50'] = self.dataset_historico['Close'].rolling(50).mean()
        self.dataset_historico['RSI'] = 100 - (100 / (1 + self.dataset_historico['Return'].rolling(14).mean() / abs(self.dataset_historico['Return'].rolling(14).mean())))


        # Variável-alvo (classificação)
        self.dataset_historico['Target'] = np.where(self.dataset_historico['Close'].shift(-5) > self.dataset_historico['Close'] * 1.005, 1, 0)  # horizonte de 5 dias e limiar 0,5%

        # Limpar NaN
        self.dataset_historico.dropna(inplace=True)

        # Features e target
        self.X = self.dataset_historico[['SMA_20', 'SMA_50', 'RSI']]
        self.y = self.dataset_historico['Target']
        self.dataset_historico.dropna(inplace=True)
        return self.dataset_historico.head()

#TODO: Classe do backtest
class Backtest():
    def __init__(self):
        self.dados=Dados()
        self.indicadores=self.dados.indicadores_historico()

    def treinamento(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', eta=0.1, max_depth=3)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        model.save_model('xgboost.json') 


# TODO: Rodando o modelo

Backtest().treinamento()
