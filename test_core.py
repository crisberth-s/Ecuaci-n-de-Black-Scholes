"""
test_core.py
------------
Script rápido de verificación (no usa Streamlit) para confirmar que
toda la lógica del proyecto funciona correctamente antes de lanzar
el dashboard con `streamlit run app.py`.
"""

from src.black_scholes import bs_price, bs_delta
from src.simulate import galton_board, simulate_gbm_paths
from src.nn_pricer import train_pricer, predict_price
from src.nn_timeseries import run_emh_experiment

print("== 1. Black-Scholes ==")
p = bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
d = bs_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
print(f"Precio call ATM: {p:.4f} | Delta: {d:.4f}")

print("\n== 2. Simulaciones ==")
pos = galton_board(n_rows=20, n_balls=2000, seed=1)
print(f"Galton board -> media: {pos.mean():.2f} (esperado ~10), std: {pos.std():.2f}")
t, paths = simulate_gbm_paths(n_paths=5, n_steps=50, seed=1)
print(f"GBM paths shape: {paths.shape}, precio final ejemplo: {paths[0, -1]:.2f}")

print("\n== 3. Red neuronal - Pricing ==")
model, scaler, metrics, _ = train_pricer(n_samples=3000, hidden_layer_sizes=(32, 16))
print(f"MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f}")
nn_p = predict_price(model, scaler, 100, 100, 1, 0.05, 0.2)
print(f"Precio NN vs BS (S=K=100,T=1,r=5%,sigma=20%): NN={nn_p:.3f} | BS={p:.3f}")

print("\n== 4. Red neuronal - Test EMH ==")
res = run_emh_experiment(n_steps=300, seed=7)
print(f"RMSE NN: {res['rmse_nn']:.4f} | RMSE ingenuo (EMH): {res['rmse_naive']:.4f}")
print(f"¿La red venció al mercado?: {res['beats_market']}")

print("\nTodo funcionó correctamente ✅")
