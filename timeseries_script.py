import pandas as pd
import numpy as np
from timeseries_model import StockPredictor

def train_general_model(df, epochs=100):
    """
    Treina um modelo geral para todos os tickers
    """
    print("\nTreinando modelo geral para todos os tickers...")
    predictor = StockPredictor(sequence_length=60)
    history = predictor.train(df, epochs=epochs, batch_size=32)
    
    model_path = "models/general_model"
    predictor.save_model(model_path)
    print(f"Modelo geral salvo em: {model_path}")

def main():
    """Função principal"""
    print("Iniciando treinamento de modelo geral")
    print("=" * 60)
    
    try:
        print("Carregando dados do arquivo parquet...")
        df = pd.read_parquet('dados')
        
        print(f"Dados carregados com sucesso!")
        print(f"Total de registros: {len(df):,}")
        print(f"Tickers únicos: {sorted(df['Ticker'].unique())}")
        print(f"Período: {df['Date'].min()} até {df['Date'].max()}")
        print(f"Colunas disponíveis: {list(df.columns)}")
        
        required_columns = ['Date', 'Ticker', 'Close', 'Open', 'High', 'Low', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Colunas obrigatórias ausentes: {missing_columns}")
            return
        
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            print("Convertendo coluna Date para datetime...")
            df['Date'] = pd.to_datetime(df['Date'])
        
        epochs_input = input(f"\nNúmero de epochs para treinamento (padrão: 100): ").strip()
        epochs = 100
        if epochs_input:
            try:
                epochs = int(epochs_input)
                if epochs <= 0:
                    epochs = 100
                    print("Número inválido, usando padrão: 100")
            except ValueError:
                print("Número inválido, usando padrão: 100")
        
        train_general_model(df, epochs=epochs)
    
    except FileNotFoundError:
        print("Arquivo 'dados' não encontrado!")
        print("Certifique-se de que o arquivo parquet existe no diretório atual.")
    
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()