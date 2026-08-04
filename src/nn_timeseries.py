"""
nn_timeseries.py
----------------
Segunda aplicación de redes neuronales inspirada en el video: usar una
red para intentar PREDECIR el siguiente precio de una serie temporal
generada por un random walk (GBM), y comparar su desempeño contra la
línea base "ingenua" que dicta la Hipótesis de Mercados Eficientes (EMH):

    "la mejor predicción del precio de mañana es el precio de hoy"

Si el mercado es realmente eficiente (como postulaba Bachelier y como
se explica en el video), ninguna red neuronal debería superar de forma
consistente a esta línea base ingenua quitando el ruido de la muestra.
Esto replica, en miniatura, el mismo test que Bradford Cornell aplicó
al Medallion Fund de Renaissance Technologies.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .simulate import simulate_single_walk


def build_lag_features(prices, n_lags=10):
    """
    Construye un dataset de ventanas deslizantes:
    X_t = [precio_{t-n_lags}, ..., precio_{t-1}]  ->  y_t = precio_t
    """
    X, y = [], []
    for i in range(n_lags, len(prices)):
        X.append(prices[i - n_lags:i])
        y.append(prices[i])
    return np.array(X), np.array(y)


def naive_random_walk_baseline(y_test, X_test):
    """Predicción ingenua: el precio de mañana = último precio conocido."""
    return X_test[:, -1]


def run_emh_experiment(S0=100.0, mu=0.0, sigma=0.25, T=1.0, n_steps=500,
                        n_lags=10, seed=7, hidden_layer_sizes=(32, 16)):
    """
    Ejecuta el experimento completo:
      1. Simula una trayectoria de precios (random walk / GBM).
      2. Entrena una red neuronal para predecir el precio siguiente
         a partir de los últimos `n_lags` precios.
      3. Compara su error (RMSE) contra la línea base ingenua (EMH).

    Retorna un diccionario con series y métricas listas para graficar
    en el dashboard.
    """
    t, prices = simulate_single_walk(S0, mu, sigma, T, n_steps, seed=seed)

    X, y = build_lag_features(prices, n_lags=n_lags)
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=1000,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=25,
    )
    model.fit(X_train_s, y_train)
    y_pred_nn = model.predict(X_test_s)

    y_pred_naive = naive_random_walk_baseline(y_test, X_test)

    rmse_nn = float(np.sqrt(mean_squared_error(y_test, y_pred_nn)))
    rmse_naive = float(np.sqrt(mean_squared_error(y_test, y_pred_naive)))

    return {
        "t": t,
        "prices": prices,
        "y_test": y_test,
        "y_pred_nn": y_pred_nn,
        "y_pred_naive": y_pred_naive,
        "rmse_nn": rmse_nn,
        "rmse_naive": rmse_naive,
        "beats_market": rmse_nn < rmse_naive,
        "split_index": split + n_lags,
    }
