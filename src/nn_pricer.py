"""
nn_pricer.py
------------
Red neuronal (Perceptrón Multicapa) que APRENDE a poner precio a una
opción europea a partir de ejemplos, en lugar de usar la fórmula
cerrada de Black-Scholes-Merton.

Esta es la idea central que conecta el video con las redes neuronales:
así como Jim Simons y Renaissance Technologies reemplazaron modelos
analíticos por modelos estadísticos/de aprendizaje automático que
"descubren" patrones directamente de los datos, aquí entrenamos una
red neuronal para redescubrir (aproximar) la superficie de precios
de Black-Scholes únicamente a partir de ejemplos (S, K, T, r, sigma) -> precio.

Se usa scikit-learn (MLPRegressor) para mantener el proyecto ligero
y 100% reproducible sin dependencias pesadas como TensorFlow/PyTorch,
pero el pipeline es el mismo que se usaría con cualquier framework de
deep learning.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from .black_scholes import bs_price


def generate_training_data(n_samples=20000, seed=42):
    """
    Genera un dataset sintético muestreando parámetros de mercado en
    rangos realistas y calculando el precio 'verdadero' con Black-Scholes.
    La red neuronal solo verá (S, K, T, r, sigma) -> price, nunca la fórmula.
    """
    rng = np.random.default_rng(seed)

    S = rng.uniform(50, 150, n_samples)
    K = rng.uniform(50, 150, n_samples)
    T = rng.uniform(0.05, 2.0, n_samples)
    r = rng.uniform(0.0, 0.08, n_samples)
    sigma = rng.uniform(0.05, 0.6, n_samples)

    price = bs_price(S, K, T, r, sigma, option_type="call")

    X = np.column_stack([S, K, T, r, sigma])
    y = price
    return X, y


def train_pricer(n_samples=20000, hidden_layer_sizes=(64, 64, 32),
                  seed=42, test_size=0.2):
    """
    Entrena el MLP y devuelve el modelo, el escalador de features y
    métricas de desempeño sobre un conjunto de prueba.
    """
    X, y = generate_training_data(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=800,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    return model, scaler, metrics, (X_test, y_test, y_pred)


def predict_price(model, scaler, S, K, T, r, sigma):
    """Predice el precio de la opción con la red ya entrenada."""
    X = np.array([[S, K, T, r, sigma]], dtype=float)
    X_s = scaler.transform(X)
    return float(model.predict(X_s)[0])
