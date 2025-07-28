from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from stock_predictor import StockPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
    """
    Endpoint para obter previsões de um ticker
    
    Args:
        ticker (str): Símbolo da ação
        
    Query Parameters:
        days (int): Número de dias para prever (padrão: 5)
    """
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

