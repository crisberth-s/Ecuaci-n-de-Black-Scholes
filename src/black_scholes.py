"""
black_scholes.py
-----------------
Implementación analítica del modelo de Black-Scholes-Merton (1973),
la ecuación que —según el video "La Ecuación del Billón de Dólares"—
transformó el pricing de opciones y dio origen a la industria moderna
de derivados.

También se incluyen las 'Griegas' (Delta, Gamma, Vega) necesarias
para el hedging dinámico (idea introducida por Ed Thorp y formalizada
por Black, Scholes y Merton).

Referencias conceptuales del video:
  - Bachelier (1900): random walk / paseo aleatorio del precio.
  - Thorp (1967): delta hedging dinámico.
  - Black-Scholes-Merton (1973): fórmula cerrada de precio de opciones.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """Calcula d1 y d2, los parámetros centrales de la fórmula BSM."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-8)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Precio analítico de una opción europea bajo Black-Scholes-Merton.

    Parámetros
    ----------
    S : precio actual del subyacente
    K : precio de ejercicio (strike)
    T : tiempo a vencimiento, en años
    r : tasa libre de riesgo (anual, continua)
    sigma : volatilidad anualizada del subyacente
    option_type : "call" o "put"
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type debe ser 'call' o 'put'")

    return np.maximum(price, 0.0)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Delta: sensibilidad del precio de la opción ante cambios en S.
    Es la cantidad de acciones que Thorp/Black-Scholes usan para el
    'dynamic hedging' (portafolio libre de riesgo)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    """Gamma: variación del Delta ante cambios en S."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = np.maximum(np.asarray(T, dtype=float), 1e-8)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """Vega: sensibilidad del precio ante cambios en la volatilidad."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = np.maximum(np.asarray(T, dtype=float), 1e-8)
    return S * norm.pdf(d1) * np.sqrt(T)


def payoff_diagram(S_range, K, premium, option_type="call", position="long"):
    """
    Calcula el diagrama de ganancias/pérdidas (P&L) de una opción al
    vencimiento, tal como se muestra en el video (minuto ~05:14).
    """
    S_range = np.asarray(S_range, dtype=float)
    if option_type == "call":
        intrinsic = np.maximum(S_range - K, 0.0)
    else:
        intrinsic = np.maximum(K - S_range, 0.0)

    pnl = intrinsic - premium
    if position == "short":
        pnl = -pnl
    return pnl
