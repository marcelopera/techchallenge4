from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import time
from timeseries_model import StockPredictor
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
metrics = PrometheusMetrics(app)

PREDICTION_REQUEST_COUNT = Counter(
    'stock_prediction_requests_total',
    'Total number of stock prediction requests',
    ['ticker']
)

PREDICTION_LATENCY = Histogram(
    'stock_prediction_latency_seconds',
    'Time spent processing prediction requests',
    ['ticker']
)

MODEL_LOAD_TIME = Histogram(
    'model_load_time_seconds',
    'Time spent loading models',
    ['ticker']
)

ENDPOINT_REQUESTS = Counter(
    'endpoint_requests_total',
    'Total number of requests per endpoint',
    ['endpoint', 'status_code']
)

ENDPOINT_LATENCY = Histogram(
    'endpoint_latency_seconds',
    'Latency per endpoint request',
    ['endpoint']
)


loaded_models = {}

class StockPredictionAPI:
    def __init__(self, models_dir="models"):
        """
        Inicializa a API de previsão de ações
        """
        self.models_dir = models_dir
        self.ensure_models_dir()
    
    def ensure_models_dir(self):
        """Garante que o diretório de modelos existe"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def load_model_for_ticker(self, ticker):
        """
        Carrega o modelo para um ticker específico
        """
        model_path = os.path.join(self.models_dir, f"{ticker.lower()}_model")
        
        if not os.path.exists(f"{model_path}_model.keras"):
            raise FileNotFoundError(f"Modelo não encontrado para o ticker {ticker}")
        
        if ticker not in loaded_models:
            predictor = StockPredictor()
            predictor.load_model(model_path)
            loaded_models[ticker] = predictor
            logger.info(f"Modelo carregado para {ticker}")
        
        return loaded_models[ticker]
    
    def get_prediction(self, ticker, days=5):
        """
        Obtém previsão para um ticker
        """
        try:
            predictor = self.load_model_for_ticker(ticker)
            predictions = predictor.predict_next_days(days)
            
            base_date = datetime.now()
            future_dates = []
            
            for i in range(1, days + 1):
                future_date = base_date + timedelta(days=i)
                while future_date.weekday() >= 5:
                    future_date += timedelta(days=1)
                future_dates.append(future_date.strftime('%Y-%m-%d'))
            
            result = {
                'ticker': ticker,
                'predictions': [
                    {
                        'date': date,
                        'predicted_close': round(float(price), 2)
                    }
                    for date, price in zip(future_dates, predictions)
                ],
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_info': {
                    'sequence_length': predictor.sequence_length,
                    'trained_ticker': predictor.ticker
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao gerar previsão para {ticker}: {str(e)}")
            raise e
    

prediction_api = StockPredictionAPI()

@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def record_metrics(response):
    resp_time = time.time() - request.start_time if hasattr(request, 'start_time') else 0
    endpoint = request.endpoint if request.endpoint else request.path
    ENDPOINT_LATENCY.labels(endpoint=endpoint).observe(resp_time)
    ENDPOINT_REQUESTS.labels(endpoint=endpoint, status_code=response.status_code).inc()
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'loaded_models': list(loaded_models.keys())
    })

@app.route('/models', methods=['GET'])
def list_models():
    """Lista todos os modelos disponíveis"""
    try:
        models = []
        for filename in os.listdir(prediction_api.models_dir):
            if filename.endswith('_model.keras'):
                ticker = filename.replace('_model.keras', '').upper()
                models.append({
                    'ticker': ticker,
                    'model_file': filename,
                    'loaded': ticker in loaded_models
                })
        
        return jsonify({
            'available_models': models,
            'total_models': len(models)
        })
        
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        return jsonify({
            'error': 'Erro ao listar modelos',
            'message': str(e)
        }), 500

@app.route('/forecast', methods=['POST'])
def forecast_from_history():
    """
    Endpoint para receber dados históricos e retornar uma previsão.
    
    Exemplo de corpo da requisição:
    {
        "data": [
            {
                "date": "2025-07-28",
                "value": "200"
            },
            {
                "date": "2025-07-29",
                "value": "201"
            },
            {
                "date": "2025-07-30",
                "value": "202"
            }
        ]
    }
    
    Query Parameters:
        days (int): Número de dias para prever (padrão: 5)
    """
    try:
        payload = request.get_json()
        if not payload or 'data' not in payload:
            return jsonify({"error": "Dados inválidos"}), 400

        history = payload['data']
        if not isinstance(history, list) or not history:
            return jsonify({"error": "Dados históricos vazios"}), 400

        # Ordena os dados pela data
        history_sorted = sorted(history, key=lambda x: x['date'])
        values = [float(item['value']) for item in history_sorted if 'value' in item]

        if not values:
            return jsonify({"error": "Valores históricos não encontrados"}), 400

        # Cálculo simples do incremento médio entre os valores
        differences = [j - i for i, j in zip(values[:-1], values[1:])]
        avg_change = sum(differences) / len(differences) if differences else 0

        days = int(request.args.get('days', 5))
        predictions = []
        last_value = values[-1]

        for i in range(1, days + 1):
            last_value += avg_change
            predictions.append(round(last_value, 2))

        # Geração de datas futuras válidas (desconsiderando fins de semana)
        last_date_str = history_sorted[-1]['date']
        base_date = datetime.strptime(last_date_str, '%Y-%m-%d')
        future_dates = []
        for i in range(1, days + 1):
            future_date = base_date + timedelta(days=i)
            while future_date.weekday() >= 5:
                future_date += timedelta(days=1)
            future_dates.append(future_date.strftime('%Y-%m-%d'))

        result = {
            "predictions": [
                {
                    "date": date,
                    "predicted_value": pred
                }
                for date, pred in zip(future_dates, predictions)
            ],
            "prediction_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error("Erro ao gerar previsão a partir de dados históricos: " + str(e))
        return jsonify({
            "error": "Erro ao gerar previsão a partir de dados históricos",
            "message": str(e)
        }), 500
    
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint não encontrado'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Erro interno do servidor'
    }), 500