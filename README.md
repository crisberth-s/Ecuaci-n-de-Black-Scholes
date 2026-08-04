# Opciones Financieras + Redes Neuronales

Proyecto que combina la historia contada en el video **"La Ecuación del
Billón de Dólares"** (Bachelier, Einstein, Thorp, Black-Scholes-Merton,
Jim Simons/Renaissance Technologies) con **redes neuronales** aplicadas
al pricing de opciones y a la predicción de series de tiempo financieras.

## ¿Qué hace este proyecto?

1. **Teoría interactiva**: resumen navegable de las ideas del video.
2. **Paseo aleatorio**: simulación de la máquina de Galton y de
   trayectorias de precio con Movimiento Browniano Geométrico (GBM).
3. **Black-Scholes-Merton**: calculadora analítica de precio de
   opciones (call/put) con Delta, Gamma, Vega y diagrama de P&L.
4. **Red neuronal — Pricing**: un Perceptrón Multicapa (MLP) que
   aprende a predecir el precio de una opción europea a partir de
   ejemplos `(S, K, T, r, σ) → precio`, sin usar la fórmula cerrada,
   y se compara contra Black-Scholes.
5. **Red neuronal — Test de Hipótesis de Mercados Eficientes**: se
   entrena una red para predecir el "siguiente precio" de una serie
   simulada como random walk, y se compara contra la línea base
   ingenua `precio(t+1) = precio(t)`, replicando la idea del video de
   que en un mercado eficiente ningún modelo debería vencer de forma
   consistente al paseo aleatorio.

## Estructura del proyecto

```
proyecto_opciones_nn/
├── app.py                 # Dashboard interactivo (Streamlit)
├── test_core.py            # Script de verificación rápida sin Streamlit
├── requirements.txt
├── README.md
└── src/
    ├── black_scholes.py    # Fórmula BSM + Griegas + diagrama de P&L
    ├── simulate.py          # Galton board + simulación GBM
    ├── nn_pricer.py         # Red neuronal para pricing de opciones
    └── nn_timeseries.py     # Red neuronal para el test de la EMH
```

## Instalación y ejecución

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar que la lógica corre correctamente (opcional)
python test_core.py

# 4. Lanzar el dashboard
streamlit run app.py
```

Luego abre el navegador en `http://localhost:8501`.

## Fundamento teórico (resumen)

| Concepto | Módulo | Idea del video |
|---|---|---|
| Random walk / Galton board | `simulate.py` | Bachelier modeló el precio como una suma de pasos aleatorios |
| Movimiento Browniano Geométrico | `simulate.py` | Einstein probó que la difusión de partículas sigue la misma matemática |
| Black-Scholes-Merton | `black_scholes.py` | Fórmula cerrada que originó la industria moderna de derivados |
| Delta hedging | `black_scholes.py` | Idea de Ed Thorp: replicar una opción con acciones + bonos |
| Red neuronal de pricing | `nn_pricer.py` | Análogo a los modelos estadísticos de Renaissance Technologies |
| Test de la EMH | `nn_timeseries.py` | ¿Puede un modelo "vencer" al mercado si este es eficiente? |

## Repositorio de GitHub

> _Reemplazar con el enlace real del repositorio del proyecto._
> `https://github.com/<usuario>/<repositorio>`

## Autores

Universidad Nacional del Altiplano — Puno, 2026
Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas
Escuela Profesional de Ingeniería de Sistemas
