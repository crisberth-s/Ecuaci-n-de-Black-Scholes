"""
simulate.py
-----------
Simula los dos fenómenos centrales del video:

1. La "máquina de Galton" (Galton board): miles de bolas que caen de
   forma aleatoria (50/50 en cada clavija) y que, en conjunto, forman
   una distribución normal. Bachelier usó esta misma idea para modelar
   el precio de una acción.

2. El Movimiento Browniano Geométrico (GBM): el modelo estándar de
   "paseo aleatorio" (random walk) para el precio de un activo,
   equivalente matemático del movimiento de partículas de polen
   que Einstein explicó en 1905.
"""

import numpy as np


def galton_board(n_rows=20, n_balls=6000, seed=None):
    """
    Simula una máquina de Galton: cada bola pasa por n_rows clavijas,
    yendo a la izquierda (0) o derecha (1) con probabilidad 0.5.
    La posición final es la suma de sus pasos (~ Binomial(n_rows, 0.5)),
    que converge a una Normal por el Teorema Central del Límite.
    """
    rng = np.random.default_rng(seed)
    steps = rng.integers(0, 2, size=(n_balls, n_rows))  # 0 = izquierda, 1 = derecha
    final_positions = steps.sum(axis=1)  # entre 0 y n_rows
    return final_positions


def simulate_gbm_paths(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=252,
                        n_paths=200, seed=None):
    """
    Simula trayectorias de precio bajo Movimiento Browniano Geométrico:

        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Esta es la versión "mejorada" del modelo de Bachelier que usaron
    Thorp, Black, Scholes y Merton, ya que evita precios negativos y
    incorpora una tendencia (drift) además del componente puramente
    aleatorio.

    Retorna
    -------
    t : vector de tiempos (n_steps+1,)
    paths : matriz (n_paths, n_steps+1) con las trayectorias simuladas
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    Z = rng.standard_normal(size=(n_paths, n_steps))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    paths = S0 * np.exp(log_paths)
    return t, paths


def simulate_single_walk(S0=100.0, mu=0.0, sigma=0.2, T=1.0, n_steps=252, seed=None):
    """Devuelve una única trayectoria GBM (útil para la demo de series de tiempo)."""
    t, paths = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths=1, seed=seed)
    return t, paths[0]
