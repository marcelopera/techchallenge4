import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Adicionando import para otimização de hiperparâmetros
import kerastuner as kt

class StockPredictor:
    def __init__(self, sequence_length=60):
        """
        Inicializa o preditor de ações
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
    
    def quick_test(self, save_path="models/general_model"):
        """
        Realiza um teste rápido de funcionalidade com dados fictícios e salva o modelo
        """
        print("Iniciando teste rápido de funcionalidade...")

        # Gerar dados fictícios (senoide para simular preços)
        np.random.seed(42)
        timesteps = 200
        x = np.linspace(0, 20, timesteps)
        prices = 50 + 10 * np.sin(x) + np.random.normal(0, 1, timesteps)

        # Criar DataFrame com os dados fictícios
        df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=timesteps),
                        'Ticker': ['TEST'] * timesteps,
                        'Close': prices})

        # Preparar os dados
        X, y = self.prepare_data(df)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Construir e treinar o modelo
        print("Treinando modelo com dados fictícios...")
        self.model = self.build_model()
        self.model.fit(X, y, epochs=5, batch_size=16, verbose=1, validation_split=0.2)

        # Salvar o modelo treinado
        self.is_trained = True
        self.last_sequence = X[-1]
        self.save_model(save_path)
        print(f"Modelo salvo em: {save_path}")

        print("Teste rápido concluído com sucesso!")
        
    def prepare_data(self, df):
        """
        Prepara os dados para treinamento de um modelo geral
        """
        if len(df) < self.sequence_length + 5:
            raise ValueError(f"Dados insuficientes. Necessário pelo menos {self.sequence_length + 5} registros.")
        
        df = df.sort_values(['Ticker', 'Date'])
        prices = df['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """
        Constrói o modelo LSTM fixo
        """
        model = Sequential()
        
        # Primeira camada
        model.add(LSTM(units=50, return_sequences=True, 
                      input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))
        
        # Segunda camada
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Terceira camada
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        
        # Camada de saída
        model.add(Dense(units=1))
        
        # Compila o modelo
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        
        return model

    def build_model_hp(self, hp):
        """
        Constrói o modelo LSTM com hiperparâmetros otimizáveis
        """
        model = Sequential()
        
        # Primeira camada
        units_1 = hp.Int('units_1', min_value=30, max_value=100, step=10)
        dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
        model.add(LSTM(units=units_1, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(dropout_1))
        
        # Segunda camada
        units_2 = hp.Int('units_2', min_value=30, max_value=100, step=10)
        dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
        model.add(LSTM(units=units_2, return_sequences=True))
        model.add(Dropout(dropout_2))
        
        # Terceira camada
        units_3 = hp.Int('units_3', min_value=30, max_value=100, step=10)
        dropout_3 = hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)
        model.add(LSTM(units=units_3))
        model.add(Dropout(dropout_3))
        
        # Camada de saída
        model.add(Dense(units=1))
        
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        return model
    
    def optimize_hyperparameters(self, df, ticker, epochs=50, batch_size=32, validation_split=0.2):
        """
        Otimiza os hiperparâmetros utilizando Keras Tuner (Hyperband)
        """
        print(f"Preparando dados para otimização no ticker {ticker}...")
        X, y, ticker_data = self.prepare_data(df, ticker)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        tuner = kt.RandomSearch(
            self.build_model_hp,
            objective='val_loss',
            max_trials=5,
            directory='hyperparam_tuning',
            project_name=f'tuning_{ticker}'
        )
        
        tuner.search(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Melhores hiperparâmetros:", best_hp.values)
        
        self.model = tuner.hypermodel.build(best_hp)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        
        self.is_trained = True
        self.ticker = ticker
        self.last_sequence = X[-1]
        
        return history
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """
        Treina o modelo geral utilizando os dados combinados de todos os tickers
        """
        print("Preparando dados para treinamento geral...")
        X, y = self.prepare_data(df)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print("Construindo modelo...")
        self.model = self.build_model()
        
        print("Treinando modelo geral...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        self.last_sequence = X[-1]
        
        print("Treinamento concluído!")
        
        return history
    
    def predict_next_days(self, days=5):
        """
        Prevê os próximos N dias
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(days):
            pred_scaled = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
            pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled[0][0]
        
        return predictions
    
    def evaluate_model(self, df, ticker):
        """
        Avalia o desempenho do modelo
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        X, y, _ = self.prepare_data(df, ticker)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Faz previsões
        predictions = self.model.predict(X, verbose=0)
        
        # Desnormaliza
        y_actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        y_pred = self.scaler.inverse_transform(predictions)
        
        # Calcula métricas
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        mape = mean_absolute_percentage_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }
    
    def save_model(self, filepath):
        """
        Salva o modelo treinado e o scaler
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salva o modelo e o scaler
        self.model.save(f"{filepath}_model.keras")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        print(f"Modelo salvo em {filepath}")

    def load_model(self, filepath):
        """
        Carrega o modelo treinado e o scaler
        """
        from tensorflow.keras.models import load_model
        
        # Carrega o modelo e o scaler
        self.model = load_model(f"{filepath}_model.keras")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True
        print(f"Modelo carregado de {filepath}")

if __name__ == "__main__":
    predictor = StockPredictor(sequence_length=60)
    predictor.quick_test()