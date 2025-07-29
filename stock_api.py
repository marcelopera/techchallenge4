from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from stock_predictor import StockPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

request_count = Counter('flask_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('flask_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_requests = Gauge('flask_active_requests', 'Active requests')
memory_usage = Gauge('flask_memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('flask_cpu_usage_percent', 'CPU usage percentage')
loaded_models_count = Gauge('flask_loaded_models_total', 'Number of loaded models')
prediction_errors = Counter('flask_prediction_errors_total', 'Total prediction errors', ['ticker', 'type'])

loaded_models = {}

class StockPredictionAPI:
    def __init__(self, models_dir="models"):
        """Inicializa a API de previsão de ações"""
        self.models_dir = models_dir
        self.ensure_models_dir()
        self.update_system_metrics()
    
    def update_system_metrics(self):
        """Atualiza métricas do sistema"""
        memory_usage.set(psutil.virtual_memory().used)
        cpu_usage.set(psutil.cpu_percent())
        loaded_models_count.set(len(loaded_models))
    
    def ensure_models_dir(self):
        """Garante que o diretório de modelos existe"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def load_model_for_ticker(self, ticker):
        """Carrega o modelo para um ticker específico"""
        model_path = os.path.join(self.models_dir, f"{ticker.lower()}_model")
        
        if not os.path.exists(f"{model_path}_model.keras"):
            prediction_errors.labels(ticker=ticker, type='model_not_found').inc()
            raise FileNotFoundError(f"Modelo não encontrado para o ticker {ticker}")
        
        if ticker not in loaded_models:
            predictor = StockPredictor()
            predictor.load_model(model_path)
            loaded_models[ticker] = predictor
            loaded_models_count.set(len(loaded_models))
            logger.info(f"Modelo carregado para {ticker}")
        
        return loaded_models[ticker]
    
    def get_prediction(self, ticker, days=5):
        """Obtém previsão para um ticker"""
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
            prediction_errors.labels(ticker=ticker, type='prediction_error').inc()
            logger.error(f"Erro ao gerar previsão para {ticker}: {str(e)}")
            raise e
    
    def get_manual_prediction(self, ticker, dates, prices, predict_days=5):
        """Obtém previsão baseada em dados inseridos manualmente"""
        try:
            predictor = self.load_model_for_ticker(ticker)
            
            result = predictor.predict_from_manual_data(dates, prices, predict_days)
            
            api_result = {
                'ticker': ticker,
                'input_data': [
                    {
                        'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date,
                        'price': round(float(price), 2)
                    }
                    for date, price in zip(result['input_dates'], result['input_prices'])
                ],
                'predictions': [
                    {
                        'date': date.strftime('%Y-%m-%d'),
                        'predicted_close': round(float(price), 2)
                    }
                    for date, price in zip(result['predicted_dates'], result['predicted_prices'])
                ],
                'summary': {
                    'last_real_price': round(float(result['summary']['last_real_price']), 2),
                    'first_predicted_price': round(float(result['summary']['first_predicted_price']), 2),
                    'last_predicted_price': round(float(result['summary']['last_predicted_price']), 2),
                    'predicted_change': round(float(result['summary']['predicted_change']), 2),
                    'predicted_change_percent': round(float(result['summary']['predicted_change_percent']), 2)
                },
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_info': {
                    'sequence_length': predictor.sequence_length,
                    'trained_ticker': predictor.ticker
                }
            }
            
            return api_result
            
        except Exception as e:
            prediction_errors.labels(ticker=ticker, type='manual_prediction_error').inc()
            logger.error(f"Erro ao gerar previsão manual para {ticker}: {str(e)}")
            raise e
    

prediction_api = StockPredictionAPI()

@app.before_request
def before_request():
    request.start_time = time.time()
    active_requests.inc()
    prediction_api.update_system_metrics()

@app.after_request
def after_request(response):
    request_duration.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown'
    ).observe(time.time() - request.start_time)
    
    request_count.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()
    
    active_requests.dec()
    return response

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint para métricas do Prometheus"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'loaded_models': list(loaded_models.keys())
    })

@app.route('/predict/<ticker>', methods=['GET'])
def predict_stock(ticker):
    """Endpoint para obter previsões de um ticker"""
    try:
        ticker = ticker.upper()
        days = int(request.args.get('days', 5))
        
        if days < 1 or days > 30:
            return jsonify({
                'error': 'Número de dias deve estar entre 1 e 30'
            }), 400
        
        result = prediction_api.get_prediction(ticker, days)
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({
            'error': f'Modelo não encontrado para o ticker {ticker}',
            'message': 'Você precisa treinar um modelo para este ticker primeiro'
        }), 404
        
    except Exception as e:
        logger.error(f"Erro no endpoint predict: {str(e)}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500

@app.route('/predict-manual/<ticker>', methods=['POST'])
def predict_stock_manual(ticker):
    """Endpoint para obter previsões baseadas em dados inseridos manualmente
        {
            "dates": ["2024-01-15", "2024-01-16", ...],
            "prices": [185.50, 187.20, ...],
            "predict_days": 5  // opadrão: 5
        }
    """
    try:
        ticker = ticker.upper()

        if not request.is_json:
            return jsonify({
                'error': 'Content-Type deve ser application/json'
            }), 400
        
        data = request.get_json()

        if 'dates' not in data or 'prices' not in data:
            return jsonify({
                'error': 'Campos obrigatórios: dates, prices',
                'example': {
                    'dates': ['2024-01-15', '2024-01-16', '2024-01-17'],
                    'prices': [185.50, 187.20, 189.10],
                    'predict_days': 5
                }
            }), 400
        
        dates = data['dates']
        prices = data['prices']
        predict_days = data.get('predict_days', 5)
        
        if not isinstance(dates, list) or not isinstance(prices, list):
            return jsonify({
                'error': 'dates e prices devem ser listas'
            }), 400
        
        if len(dates) != len(prices):
            return jsonify({
                'error': 'Número de datas deve ser igual ao número de preços'
            }), 400
        
        if len(dates) < 10:
            return jsonify({
                'error': 'Necessário pelo menos 10 pontos de dados para fazer previsões'
            }), 400
        
        if predict_days < 1 or predict_days > 30:
            return jsonify({
                'error': 'predict_days deve estar entre 1 e 30'
            }), 400
        
        try:
            for date_str in dates:
                datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({
                'error': 'Formato de data inválido. Use YYYY-MM-DD'
            }), 400
        
        try:
            prices = [float(price) for price in prices]
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Todos os preços devem ser números válidos'
            }), 400

        result = prediction_api.get_manual_prediction(ticker, dates, prices, predict_days)
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({
            'error': f'Modelo não encontrado para o ticker {ticker}',
            'message': 'Você precisa treinar um modelo para este ticker primeiro'
        }), 404
        
    except ValueError as e:
        return jsonify({
            'error': 'Erro de validação',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Erro no endpoint predict-manual: {str(e)}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500

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
