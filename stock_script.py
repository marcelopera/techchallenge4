import pandas as pd
import numpy as np
from stock_predictor import StockPredictor
import warnings
warnings.filterwarnings('ignore')

def train_models_for_tickers(df, tickers, epochs=100):
    """
    Treina modelos para múltiplos tickers
    """
    results = {}
    
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Treinando modelo para {ticker}")
        print(f"{'='*50}")
        
        try:
            ticker_data = df[df['Ticker'] == ticker]
            print(f"Dados disponíveis para {ticker}: {len(ticker_data)} registros")
            
            ticker_df = df
            
            predictor = StockPredictor(sequence_length=60)
            history = predictor.train(ticker_df, ticker, epochs=epochs, batch_size=32)
            
            metrics = predictor.evaluate_model(ticker_df, ticker)
            print(f"Métricas para {ticker}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            model_path = f"models/{ticker.lower()}_model"
            predictor.save_model(model_path)

            predictions = predictor.predict_next_days(5)
            print(f"Previsões para os próximos 5 dias de {ticker}:")
            for i, pred in enumerate(predictions, 1):
                print(f"  Dia {i}: ${pred:.2f}")
            
            results[ticker] = {
                'status': 'success',
                'metrics': metrics,
                'model_path': model_path,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Erro ao treinar {ticker}: {str(e)}")
            results[ticker] = {
                'status': 'error',
                'error': str(e)
            }
    
    return results

def main():
    """Função principal"""
    print("Iniciando treinamento de modelos de previsão de ações")
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
        
        print(f"\nEstatísticas dos dados:")
        print(f"Data mais antiga: {df['Date'].min()}")
        print(f"Data mais recente: {df['Date'].max()}")
        
        ticker_counts = df['Ticker'].value_counts()
        print(f"\nRegistros por ticker:")
        for ticker in ticker_counts.head(10).index:
            print(f"   {ticker}: {ticker_counts[ticker]:,} registros")
        
        if len(ticker_counts) > 10:
            print(f"   ... e mais {len(ticker_counts) - 10} tickers")
        
        tickers_to_train = ticker_counts.head(10).index.tolist()
        
        print(f"\nTickers selecionados para treinamento: {tickers_to_train}")

        user_input = input("\nDeseja treinar todos esses tickers? (s/n) ou digite tickers específicos separados por vírgula: ").strip()
        
        if user_input.lower() == 'n':
            print("Treinamento cancelado pelo usuário.")
            return
        elif user_input.lower() != 's' and user_input:
            custom_tickers = [t.strip().upper() for t in user_input.split(',')]
            available_tickers = set(df['Ticker'].unique())
            valid_tickers = [t for t in custom_tickers if t in available_tickers]
            invalid_tickers = [t for t in custom_tickers if t not in available_tickers]
            
            if invalid_tickers:
                print(f"Tickers não encontrados nos dados: {invalid_tickers}")
            
            if valid_tickers:
                tickers_to_train = valid_tickers
                print(f"Usando tickers especificados: {tickers_to_train}")
            else:
                print("Nenhum ticker válido especificado.")
                return
        
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
        
        print(f"Iniciando treinamento com {epochs} epochs...")
        
        print("\nIniciando treinamento dos modelos...")
        results = train_models_for_tickers(df, tickers_to_train, epochs=epochs)
        
        print("\n" + "="*60)
        print("RESUMO DO TREINAMENTO")
        print("="*60)
        
        successful_models = 0
        failed_models = 0
        
        for ticker, result in results.items():
            if result['status'] == 'success':
                print(f"   {ticker}: Modelo treinado com sucesso")
                print(f"   RMSE: {result['metrics']['RMSE']:.4f}")
                print(f"   MAE: {result['metrics']['MAE']:.4f}")
                print(f"   Modelo salvo em: {result['model_path']}")
                successful_models += 1
            else:
                print(f"{ticker}: Falha no treinamento - {result['error']}")
                failed_models += 1
        
        print(f"\nResultados finais:")
        print(f"   Modelos treinados com sucesso: {successful_models}")
        print(f"   Modelos com falha: {failed_models}")
        print(f"   Modelos salvos na pasta: ./models/")
        
        if successful_models > 0:           
            print(f"\nTeste rápido - Previsão para {tickers_to_train[0]}:")
            try:
                first_ticker = tickers_to_train[0]
                if first_ticker in results and results[first_ticker]['status'] == 'success':
                    predictions = results[first_ticker]['predictions']
                    for i, pred in enumerate(predictions[:3], 1):
                        print(f"   Dia {i}: ${pred:.2f}")
                    if len(predictions) > 3:
                        print(f"   ... e mais {len(predictions)-3} previsões")
            except Exception as e:
                print(f"   Erro no teste: {e}")
        else:
            print(f"\nNenhum modelo foi treinado com sucesso.")
            print(f"Verifique os erros acima e tente novamente.")
    
    except FileNotFoundError:
        print("Arquivo 'dados' não encontrado!")
        print("Certifique-se de que o arquivo parquet existe no diretório atual.")
        print("Ou ajuste o caminho do arquivo na linha: df = pd.read_parquet('dados')")
    
    except Exception as e:
        print(f"Erro inesperado: {e}")
        print("Verifique se:")
        print("   - O arquivo 'dados' é um parquet válido")
        print("   - As colunas necessárias existem")
        print("   - Há dados suficientes para treinamento")

if __name__ == "__main__":
    main()