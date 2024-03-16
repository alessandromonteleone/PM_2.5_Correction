import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import os
from itertools import product


# Funzione per valutare un modello
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return r2, mae, mse, rmse, mape

def train_and_evaluate_arima(X_train, Y_train, X_test, Y_test, order):
    arima_model = ARIMA(Y_train, exog= X_train, order=order)
    arima_result = arima_model.fit()

    # Effettua previsioni
    y_pred = arima_result.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)

    # Calcola le metriche di valutazione
    r2 = r2_score(Y_test, y_pred)
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100

    return r2, mae, mse, rmse, mape

def find_best_arima_order(X_train, y_train):
    best_aic = float("inf")
    best_order = None

    # Definisci i range possibili per p, d, q
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 2)

    # Genera tutte le possibili combinazioni di p, d, q
    orders = list(product(p_values, d_values, q_values))

    for order in orders:
        try:
            # Crea e addestra il modello ARIMA
            model = sm.tsa.ARIMA(y_train, exog=X_train, order=order)
            results = model.fit()

            # Verifica se l'AIC è migliore del migliore finora
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        except:
            continue

    return best_order

# Funzione per addestrare e valutare i modelli
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)

def print_results_model(model_name, result):
    print(f"Modello: {model_name}")
    print(f"R-squared: {result[0]:.4f}")
    print(f"MAE: {result[1]:.4f}")
    print(f"MSE: {result[2]:.4f}")
    print(f"RMSE: {result[3]:.4f}")
    print(f"MAPE: {result[4]:.4f}")
    print("--------------------")


#-----------------------------------------------------------------------------------------------------------------------
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

#print(df_train_corrected.head())

# Modelli da provare
models_to_try = [
    LinearRegression(),
    ElasticNet(),
    Ridge(),
    Lasso(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    SVR(),
    xgb.XGBRegressor(objective="reg:squarederror"),
    #ARIMA
]

#------------------------------corrected_data---------------------------------------------------------------------------

for config in input_configs:
    X_train = df_train_corrected[config['features']]
    Y_train = df_train_corrected[config['output']]
    X_test = df_test_corrected[config['features']]
    Y_test = df_test_corrected[config['output']]
    results_models = {}

    print(f'Corrected_data - Configurazione {config}')

    for model in models_to_try:
        result = train_and_evaluate_model(X_train, Y_train, X_test, Y_test, model)
        model_name = type(model).__name__
        results_models[model_name] = result

    # Trova la migliore combinazione di ordini ARIMA
    best_arima_order = find_best_arima_order(X_train, Y_train)
    # Addestra ARIMA
    result_arima = train_and_evaluate_arima(X_train, Y_train, X_test, Y_test, best_arima_order)
    results_models['ARIMA'] = result_arima

    for model_name, result in results_models.items():
        print_results_model(model_name, result)

#------------------------------preprocessed_data------------------------------------------------------------------------

for config in input_configs:
    X_train = df_train_preprocessed[config['features']]
    Y_train = df_train_preprocessed[config['output']]
    X_test = df_test_preprocessed[config['features']]
    Y_test = df_test_preprocessed[config['output']]
    results_models = {}

    print(f'Preprocessed_data - Configurazione {config}')

    # Loop attraverso i modelli da provare
    for model in models_to_try:
        result = train_and_evaluate_model(X_train, Y_train, X_test, Y_test, model)
        model_name = type(model).__name__
        results_models[model_name] = result

    # Trova la migliore combinazione di ordini ARIMA
    best_arima_order = find_best_arima_order(X_train, Y_train)
    # Addestra ARIMA
    result_arima = train_and_evaluate_arima(X_train, Y_train, X_test, Y_test, best_arima_order)
    results_models['ARIMA'] = result_arima

    for model_name, result in results_models.items():
        print_results_model(model_name, result)

#------------------------------raw_data---------------------------------------------------------------------------------

# Loop attraverso le configurazioni
for config in input_configs:
    # Seleziona le colonne di input e output in base alla configurazione
    X_train = df_train_raw[config['features']]
    Y_train = df_train_raw[config['output']]
    X_test = df_test_raw[config['features']]
    Y_test = df_test_raw[config['output']]
    results_models = {}

    print(f'Raw_data - Configurazione {config}')

    # Loop attraverso i modelli da provare
    for model in models_to_try:
        result = train_and_evaluate_model(X_train, Y_train, X_test, Y_test, model)
        model_name = type(model).__name__
        results_models[model_name] = result

    # Trova la migliore combinazione di ordini ARIMA
    best_arima_order = find_best_arima_order(X_train, Y_train)
    # Addestra ARIMA
    result_arima = train_and_evaluate_arima(X_train, Y_train, X_test, Y_test, best_arima_order)
    results_models['ARIMA'] = result_arima

    for model_name, result in results_models.items():
        print_results_model(model_name, result)

