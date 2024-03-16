import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return r2, mae, mse, rmse, mape

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)

def print_metrics( conf, r2, mae, mse, rmse, mape):
    print(f'\nConfigurazione {config["features"]}:')
    print(f'R2: {r2:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')


# Configurazioni di input
input_configs = [
    {'features': ['pm2p5_x'], 'output': 'pm2p5_y'},
    {'features': ['pm2p5_x', 'relative_humidity', 'temperature'], 'output': 'pm2p5_y'},
    {'features': ['pm1', 'pm2p5_x', 'pm4', 'pm10', 'relative_humidity', 'temperature', 'pressure', 'wind_speed', 'cloud_coverage'], 'output': 'pm2p5_y'}
]

# Creazione dei DataFrame
try:
    df_train_corrected = pd.read_csv('../MitH/dataset/corrected_data/train.csv')
    df_test_corrected = pd.read_csv('../MitH/dataset/corrected_data/test.csv')

    df_train_preprocessed = pd.read_csv('../MitH/dataset/preprocessed_data/train.csv')
    df_test_preprocessed = pd.read_csv('../MitH/dataset/preprocessed_data/test.csv')

    df_train_raw = pd.read_csv('../MitH/dataset/raw_data/train.csv')
    df_test_raw = pd.read_csv('../MitH/dataset/raw_data/test.csv')

    print("Lettura dei dati completata con successo.")
except Exception as e:
    print(f"Errore durante la lettura dei dati: {e}")

#corrected data
for config in input_configs:
    X_train = df_train_corrected[config['features']]
    Y_train = df_train_corrected[config['output']]
    X_test = df_test_corrected[config['features']]
    Y_test = df_test_corrected[config['output']]
    # Istanziare il modello LGBMRegressor
    model = LGBMRegressor()
    # Addestramento e valutazione del modello
    r2, mae, mse, rmse, mape = train_and_evaluate_model(X_train, Y_train, X_test, Y_test, model)
    # Stampa delle metriche di valutazione
    print('Corrected data:')
    print_metrics(config,r2,mae,mse,rmse,mape)


#pre-processed data
for config in input_configs:
    X_train = df_train_preprocessed[config['features']]
    Y_train = df_train_preprocessed[config['output']]
    X_test = df_test_preprocessed[config['features']]
    Y_test = df_test_preprocessed[config['output']]
    # Istanziare il modello LGBMRegressor
    model = LGBMRegressor()
    # Addestramento e valutazione del modello
    r2, mae, mse, rmse, mape = train_and_evaluate_model(X_train, Y_train, X_test, Y_test, model)
    print('Pre-processed data:')
    # Stampa delle metriche di valutazione
    print_metrics(config,r2,mae,mse,rmse,mape)


#raw data
for config in input_configs:
    X_train = df_train_raw[config['features']]
    Y_train = df_train_raw[config['output']]
    X_test = df_test_raw[config['features']]
    Y_test = df_test_raw[config['output']]
    # Istanziare il modello LGBMRegressor
    model = LGBMRegressor()
    # Addestramento e valutazione del modello
    r2, mae, mse, rmse, mape = train_and_evaluate_model(X_train, Y_train, X_test, Y_test, model)
    print('Raw data:')
    # Stampa delle metriche di valutazione
    print_metrics(config,r2,mae,mse,rmse,mape)



