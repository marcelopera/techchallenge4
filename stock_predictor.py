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

class StockPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, df, ticker):
        """
        Prepara os dados para treinamento
        """
        ticker_data = df[df['Ticker'] == ticker].copy()
        
        if len(ticker_data) < self.sequence_length + 5:
            raise ValueError(f"Dados insuficientes para {ticker}. Necessário pelo menos {self.sequence_length + 5} registros.")
        
        ticker_data = ticker_data.sort_values('Date')
        prices = ticker_data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        return np.array(X), np.array(y), ticker_data
    
    def build_model(self):
        """
        Constrói o modelo LSTM
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
    
    def train(self, df, ticker, epochs=50, batch_size=32, validation_split=0.2):
        """
        Treina o modelo para um ticker específico
        """
        print(f"Preparando dados para {ticker}...")
        X, y, ticker_data = self.prepare_data(df, ticker)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"Construindo modelo...")
        self.model = self.build_model()
        
        print(f"Treinando modelo para {ticker}...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        self.ticker = ticker
        self.last_sequence = X[-1]
        
        print(f"Treinamento concluído para {ticker}!")
        
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
    
    def predict_from_manual_data(self, dates, prices, predict_days=5):
        """
        Faz previsões baseadas em dados inseridos manualmente pelo usuário
        
        Args:
            dates (list): Lista de datas no formato 'YYYY-MM-DD' ou objetos datetime
            prices (list): Lista de preços correspondentes às datas
            predict_days (int): Número de dias para prever (padrão: 5)
            
        Returns:
            dict: Dicionário contendo datas previstas e preços previstos
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Validações
        if len(dates) != len(prices):
            raise ValueError("O número de datas deve ser igual ao número de preços!")
        
        if len(prices) < self.sequence_length:
            raise ValueError(f"Necessário pelo menos {self.sequence_length} preços para fazer previsões!")
        
        # Converte datas para datetime se necessário
        if isinstance(dates[0], str):
            dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        
        # Ordena os dados por data
        data_pairs = list(zip(dates, prices))
        data_pairs.sort(key=lambda x: x[0])
        dates, prices = zip(*data_pairs)
        
        # Normaliza os preços usando o scaler do modelo
        prices_array = np.array(prices).reshape(-1, 1)
        
        # Para usar o scaler corretamente, precisamos dos dados na mesma escala
        # Vamos usar apenas os últimos valores para criar a sequência
        last_prices = prices_array[-self.sequence_length:]
        
        # Normaliza usando o scaler já treinado
        # Nota: Isso assume que os novos dados estão em uma faixa similar aos dados de treinamento
        scaled_last_prices = self.scaler.transform(last_prices)
        
        # Cria a sequência inicial
        current_sequence = scaled_last_prices.flatten()
        
        predictions = []
        prediction_dates = []
        
        # Gera as próximas datas
        last_date = dates[-1]
        for i in range(1, predict_days + 1):
            next_date = last_date + timedelta(days=i)
            prediction_dates.append(next_date)
        
        # Faz as previsões
        for _ in range(predict_days):
            # Reshape para o formato esperado pelo modelo
            sequence_reshaped = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Faz a previsão
            pred_scaled = self.model.predict(sequence_reshaped, verbose=0)
            
            # Desnormaliza a previsão
            pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            
            # Atualiza a sequência para a próxima previsão
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled[0][0]
        
        return {
            'input_dates': list(dates),
            'input_prices': list(prices),
            'predicted_dates': prediction_dates,
            'predicted_prices': predictions,
            'summary': {
                'last_real_price': prices[-1],
                'first_predicted_price': predictions[0],
                'last_predicted_price': predictions[-1],
                'predicted_change': predictions[-1] - prices[-1],
                'predicted_change_percent': ((predictions[-1] - prices[-1]) / prices[-1]) * 100
            }
        }
    
    def format_prediction_results(self, results):
        """
        Formata os resultados da previsão de forma legível
        """
        print("="*60)
        print("RESULTADOS DA PREVISÃO")
        print("="*60)
        
        print("\nDados de Entrada:")
        print("-" * 30)
        for date, price in zip(results['input_dates'], results['input_prices']):
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)
            print(f"{date_str}: R$ {price:.2f}")
        
        print(f"\nPrevisões para os próximos {len(results['predicted_prices'])} dias:")
        print("-" * 50)
        for date, price in zip(results['predicted_dates'], results['predicted_prices']):
            date_str = date.strftime('%Y-%m-%d')
            print(f"{date_str}: R$ {price:.2f}")
        
        print(f"\nResumo:")
        print("-" * 20)
        print(f"Último preço real: R$ {results['summary']['last_real_price']:.2f}")
        print(f"Primeira previsão: R$ {results['summary']['first_predicted_price']:.2f}")
        print(f"Última previsão: R$ {results['summary']['last_predicted_price']:.2f}")
        print(f"Mudança prevista: R$ {results['summary']['predicted_change']:.2f}")
        print(f"Mudança prevista (%): {results['summary']['predicted_change_percent']:.2f}%")
        print("="*60)
    
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
        Salva o modelo treinado
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salva o modelo Keras
        self.model.save(f"{filepath}_model.keras")
        
        # Salva o scaler e outros parâmetros
        model_data = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'ticker': self.ticker,
            'last_sequence': self.last_sequence,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f"{filepath}_data.pkl")
        
        print(f"Modelo salvo em {filepath}")
    
    def load_model(self, filepath):
        """
        Carrega um modelo treinado
        """
        from tensorflow.keras.models import load_model
        
        # Carrega o modelo Keras
        self.model = load_model(f"{filepath}_model.keras")
        
        # Carrega dados auxiliares
        model_data = joblib.load(f"{filepath}_data.pkl")
        
        self.scaler = model_data['scaler']
        self.sequence_length = model_data['sequence_length']
        self.ticker = model_data['ticker']
        self.last_sequence = model_data['last_sequence']
        self.is_trained = model_data['is_trained']
        
        print(f"Modelo carregado de {filepath}")