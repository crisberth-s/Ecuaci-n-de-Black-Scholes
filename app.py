"""
app.py
------
Dashboard interactivo (Streamlit) que combina el contenido del video
"La Ecuación del Billón de Dólares" con redes neuronales.

Ejecutar con:
    streamlit run app.py

Secciones:
  1. Teoría          -> resumen del video (Bachelier, Einstein, Thorp, BSM, Simons)
  2. Paseo Aleatorio  -> máquina de Galton + simulación GBM
  3. Black-Scholes    -> calculadora analítica + diagrama de P&L
  4. Red Neuronal (Pricing)   -> MLP que aprende a poner precio a opciones
  5. Red Neuronal (Series)    -> test de la Hipótesis de Mercados Eficientes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.black_scholes import bs_price, bs_delta, bs_gamma, bs_vega, payoff_diagram
from src.simulate import galton_board, simulate_gbm_paths
from src.nn_pricer import train_pricer, predict_price
from src.nn_timeseries import run_emh_experiment

st.set_page_config(
    page_title="Opciones + Redes Neuronales",
    page_icon="📈",
    layout="wide",
)

st.title("📈 De Bachelier a las Redes Neuronales: Pricing de Opciones")
st.caption(
    "Proyecto basado en el video *La Ecuación del Billón de Dólares* — "
    "Universidad Nacional del Altiplano, Puno 2026"
)

# ----------------------------------------------------------------------
# Sidebar / navegación
# ----------------------------------------------------------------------
seccion = st.sidebar.radio(
    "Navegación",
    [
        "1. Teoría del video",
        "2. Paseo aleatorio (Bachelier / Einstein)",
        "3. Black-Scholes-Merton",
        "4. Red neuronal: Pricing de opciones",
        "5. Red neuronal: ¿Se puede vencer al mercado?",
    ],
)

# ----------------------------------------------------------------------
# 1. TEORÍA
# ----------------------------------------------------------------------
if seccion.startswith("1"):
    st.header("Resumen del video")
    st.markdown(
        """
El video reconstruye la historia de la ecuación que dio origen a la industria
moderna de derivados (opciones, futuros, swaps), de tamaño estimado en
**cientos de billones de dólares**.

| Personaje | Aporte | Año |
|---|---|---|
| Isaac Newton | Perdió un tercio de su fortuna en el South Sea Bubble: *"puedo calcular el movimiento de los astros, pero no la locura de la gente"* | 1720 |
| Louis Bachelier | Modeló el precio de una acción como un **paseo aleatorio** (random walk), origen de la Hipótesis de Mercados Eficientes | 1900 |
| Albert Einstein | Explicó el Movimiento Browniano con la misma matemática, probando la existencia de los átomos | 1905 |
| Ed Thorp | Contó cartas en Blackjack; luego inventó el **delta hedging dinámico** para opciones | 1960s |
| Fischer Black, Myron Scholes, Robert Merton | Fórmula cerrada **Black-Scholes-Merton** para el precio de opciones | 1973 |
| Jim Simons (Renaissance Technologies) | Reemplazó los modelos analíticos por **modelos estadísticos / machine learning** (Medallion Fund, ~66% anual) | 1988+ |

**La idea que conecta todo con este proyecto:** así como Simons usó
aprendizaje automático para encontrar patrones que los modelos analíticos
no capturaban, aquí usamos **redes neuronales** para (a) aprender a poner
precio a una opción solo a partir de ejemplos, y (b) poner a prueba si es
posible "vencer" al paseo aleatorio, tal como sugiere la Hipótesis de
Mercados Eficientes.
        """
    )

# ----------------------------------------------------------------------
# 2. PASEO ALEATORIO
# ----------------------------------------------------------------------
elif seccion.startswith("2"):
    st.header("Máquina de Galton y Movimiento Browniano Geométrico")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Máquina de Galton")
        n_rows = st.slider("Número de filas de clavijas", 5, 40, 20)
        n_balls = st.slider("Número de bolas", 500, 20000, 6000, step=500)
        positions = galton_board(n_rows=n_rows, n_balls=n_balls, seed=1)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(positions, bins=n_rows + 1, color="#3366cc", edgecolor="white")
        ax.set_title("Distribución final de las bolas ≈ Normal")
        ax.set_xlabel("Posición final")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
        st.caption(
            "Cada bola sigue un camino impredecible, pero el conjunto siempre "
            "converge a una distribución normal (Teorema Central del Límite). "
            "Bachelier propuso que el precio de una acción se comporta igual."
        )

    with col2:
        st.subheader("Simulación de precios (GBM)")
        S0 = st.number_input("Precio inicial S₀", value=100.0)
        mu = st.slider("Tendencia anual (drift, μ)", -0.3, 0.3, 0.05)
        sigma = st.slider("Volatilidad anual (σ)", 0.05, 1.0, 0.2)
        n_paths = st.slider("Número de trayectorias simuladas", 5, 300, 60)

        t, paths = simulate_gbm_paths(S0=S0, mu=mu, sigma=sigma, T=1.0,
                                       n_steps=252, n_paths=n_paths, seed=2)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for i in range(min(n_paths, 60)):
            ax2.plot(t, paths[i], linewidth=0.7, alpha=0.6)
        ax2.set_title("Trayectorias simuladas de precio (paseo aleatorio)")
        ax2.set_xlabel("Tiempo (años)")
        ax2.set_ylabel("Precio")
        st.pyplot(fig2)
        st.caption(
            "Cada trayectoria representa un futuro posible del precio del "
            "activo, exactamente como cada bola en la máquina de Galton."
        )

# ----------------------------------------------------------------------
# 3. BLACK-SCHOLES
# ----------------------------------------------------------------------
elif seccion.startswith("3"):
    st.header("Calculadora Black-Scholes-Merton")

    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Precio actual (S)", value=100.0)
    K = c2.number_input("Strike (K)", value=100.0)
    T = c3.number_input("Vencimiento en años (T)", value=1.0, min_value=0.01)
    r = c4.number_input("Tasa libre de riesgo (r)", value=0.05)
    sigma = c5.number_input("Volatilidad (σ)", value=0.2, min_value=0.01)

    option_type = st.radio("Tipo de opción", ["call", "put"], horizontal=True)

    price = bs_price(S, K, T, r, sigma, option_type)
    delta = bs_delta(S, K, T, r, sigma, option_type)
    gamma = bs_gamma(S, K, T, r, sigma)
    vega = bs_vega(S, K, T, r, sigma)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio (prima)", f"${price:.2f}")
    m2.metric("Delta", f"{delta:.3f}")
    m3.metric("Gamma", f"{gamma:.4f}")
    m4.metric("Vega", f"{vega:.3f}")

    S_range = np.linspace(0.5 * K, 1.5 * K, 200)
    pnl = payoff_diagram(S_range, K, price, option_type=option_type, position="long")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.plot(S_range, pnl, color="#cc3333", linewidth=2)
    ax3.axvline(K, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_title(f"Diagrama de Ganancias/Pérdidas ({option_type})")
    ax3.set_xlabel("Precio del subyacente al vencimiento")
    ax3.set_ylabel("Ganancia / Pérdida")
    st.pyplot(fig3)

# ----------------------------------------------------------------------
# 4. RED NEURONAL: PRICING
# ----------------------------------------------------------------------
elif seccion.startswith("4"):
    st.header("Red neuronal como motor de pricing (sustituto de Black-Scholes)")
    st.markdown(
        "Entrenamos un **Perceptrón Multicapa (MLP)** con ejemplos "
        "`(S, K, T, r, σ) → precio`, generados con Black-Scholes, "
        "sin darle jamás la fórmula. La red debe *redescubrir* la superficie "
        "de precios únicamente a partir de los datos, igual que Renaissance "
        "Technologies buscaba patrones sin conocer las causas subyacentes."
    )

    n_samples = st.slider("Tamaño del dataset de entrenamiento", 2000, 40000, 15000, step=1000)
    capas = st.selectbox("Arquitectura oculta", ["(32,)", "(64, 32)", "(64, 64, 32)", "(128, 64, 32)"], index=2)
    hidden = eval(capas)

    if st.button("🚀 Entrenar red neuronal"):
        with st.spinner("Entrenando..."):
            model, scaler, metrics, (X_test, y_test, y_pred) = train_pricer(
                n_samples=n_samples, hidden_layer_sizes=hidden
            )
        st.session_state["nn_pricer"] = (model, scaler)

        c1, c2 = st.columns(2)
        c1.metric("Error absoluto medio (MAE)", f"${metrics['MAE']:.4f}")
        c2.metric("R² (test)", f"{metrics['R2']:.5f}")

        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.scatter(y_test, y_pred, s=6, alpha=0.4, color="#3366cc")
        lims = [0, max(y_test.max(), y_pred.max())]
        ax4.plot(lims, lims, "r--", linewidth=1)
        ax4.set_xlabel("Precio real (Black-Scholes)")
        ax4.set_ylabel("Precio predicho (Red Neuronal)")
        ax4.set_title("Precio real vs. predicho por la red")
        st.pyplot(fig4)

    st.divider()
    st.subheader("Probar la red entrenada")
    if "nn_pricer" in st.session_state:
        model, scaler = st.session_state["nn_pricer"]
        c1, c2, c3, c4, c5 = st.columns(5)
        S = c1.number_input("S", value=100.0, key="nnS")
        K = c2.number_input("K", value=100.0, key="nnK")
        T = c3.number_input("T", value=1.0, key="nnT")
        r = c4.number_input("r", value=0.05, key="nnr")
        sigma = c5.number_input("σ", value=0.2, key="nnsigma")

        nn_price = predict_price(model, scaler, S, K, T, r, sigma)
        bs_ref = bs_price(S, K, T, r, sigma, "call")

        c1, c2 = st.columns(2)
        c1.metric("Precio de la Red Neuronal", f"${nn_price:.3f}")
        c2.metric("Precio real Black-Scholes", f"${bs_ref:.3f}",
                   delta=f"{nn_price - bs_ref:+.3f}")
    else:
        st.info("Entrena la red primero para poder probarla aquí.")

# ----------------------------------------------------------------------
# 5. RED NEURONAL: EMH
# ----------------------------------------------------------------------
elif seccion.startswith("5"):
    st.header("¿Puede una red neuronal vencer al mercado?")
    st.markdown(
        """
Según la **Hipótesis de Mercados Eficientes** (Bachelier, y confirmada
matemáticamente por Einstein), el mejor pronóstico del precio de mañana
es simplemente el precio de hoy. Aquí entrenamos una red neuronal para
predecir el "siguiente precio" de una serie generada por un random walk,
y la comparamos contra esa línea base ingenua.
        """
    )

    c1, c2, c3 = st.columns(3)
    sigma = c1.slider("Volatilidad (σ)", 0.05, 0.6, 0.25)
    n_lags = c2.slider("Ventana de precios pasados (lags)", 3, 30, 10)
    seed = c3.number_input("Semilla aleatoria", value=7, step=1)

    if st.button("🧠 Entrenar y evaluar"):
        with st.spinner("Simulando serie y entrenando red..."):
            res = run_emh_experiment(sigma=sigma, n_lags=n_lags, seed=int(seed))

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE Red Neuronal", f"{res['rmse_nn']:.4f}")
        c2.metric("RMSE Línea base ingenua (EMH)", f"{res['rmse_naive']:.4f}")
        c3.metric("¿La red venció al mercado?",
                   "Sí ✅" if res["beats_market"] else "No ❌")

        fig5, ax5 = plt.subplots(figsize=(9, 4))
        idx = np.arange(len(res["y_test"]))
        ax5.plot(idx, res["y_test"], label="Precio real", color="black", linewidth=1.2)
        ax5.plot(idx, res["y_pred_nn"], label="Predicción Red Neuronal", color="#3366cc", alpha=0.8)
        ax5.plot(idx, res["y_pred_naive"], label="Predicción ingenua (EMH)", color="#cc3333",
                 linestyle="--", alpha=0.8)
        ax5.legend()
        ax5.set_title("Predicción de precios: Red Neuronal vs. Hipótesis de Mercados Eficientes")
        ax5.set_xlabel("Paso de tiempo (conjunto de prueba)")
        ax5.set_ylabel("Precio")
        st.pyplot(fig5)

        st.markdown(
            "**Conclusión esperada:** en un mercado simulado como random walk "
            "puro, ninguna red debería superar de forma consistente a la línea "
            "base ingenua — tal como concluye el video: si se descubren y "
            "explotan los patrones, estos tienden a desaparecer, acercando al "
            "mercado a la eficiencia perfecta."
        )

st.sidebar.divider()
st.sidebar.caption("Universidad Nacional del Altiplano — Puno, 2026")
