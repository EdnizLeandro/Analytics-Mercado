# ============================================================
#  EPIDEMIOLOGIA COVID-19 – MODELOS SIR/SEIR + MACHINE LEARNING
#  Autor: (Seu Nome)
#  Data: 2025
# ============================================================

# ----------------------
# IMPORTAÇÕES
# ----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------
# CONFIGURAÇÕES DE GRÁFICO
# ----------------------
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

# ----------------------
# 1. LEITURA DO ARQUIVO PARQUET
# ----------------------
file_path = "dados_tratados.parquet"   # altere aqui

print("📂 Carregando dados...")
df = pd.read_parquet(file_path)

print("Colunas encontradas:")
print(df.columns)

# ----------------------
# 2. PRÉ–PROCESSAMENTO E FILTRAGEM (2020–2024)
# ----------------------
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data")

df_periodo = df[(df["data"].dt.year >= 2020) & (df["data"].dt.year <= 2024)]

# Exemplo: analisar o Brasil inteiro (soma por dia)
df_br = df_periodo.groupby("data").agg({
    "casosnovos": "sum",
    "obitosnovos": "sum",
    "populacaotcu2019": "sum"
}).reset_index()

df_br["infectados"] = df_br["casosnovos"].cumsum()
df_br["recuperados"] = df_br["infectados"] * 0.92
df_br["susceptiveis"] = df_br["populacaotcu2019"] - df_br["infectados"]

# ----------------------
# 3. MODELO SIR
# ----------------------

def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Valores iniciais
N = df_br["populacaotcu2019"].iloc[0]
I0 = df_br["infectados"].iloc[0] + 1
R0 = 0
S0 = N - I0
beta = 0.22
gamma = 0.085

t = np.arange(len(df_br))

res_sir = odeint(sir_model, [S0, I0, R0], t, args=(beta, gamma, N))
S, I, R = res_sir.T

# ----------------------
# 4. MODELO SEIR
# ----------------------

def seir_model(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

sigma = 1/5.2   # período médio de incubação
E0 = 100
y0_seir = [S0, E0, I0, 0]

res_seir = odeint(seir_model, y0_seir, t, args=(beta, sigma, gamma, N))
S2, E2, I2, R2 = res_seir.T

# ----------------------
# 5. MACHINE LEARNING – PREVISÃO DE CASOS
# ----------------------

df_ml = df_br.copy()
df_ml["dia"] = np.arange(len(df_ml))

X = df_ml[["dia"]]
y = df_ml["casosnovos"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

modelo = RandomForestRegressor(n_estimators=300, random_state=42)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n📊 RESULTADOS ML:")
print(f"MAE = {mae:.2f}")
print(f"R²  = {r2:.3f}")

# Previsão dos próximos 60 dias
dias_futuros = np.arange(len(df_ml), len(df_ml) + 60)
pred_futuro = modelo.predict(pd.DataFrame({"dia": dias_futuros}))

# ----------------------
# 6. GRÁFICOS
# ----------------------

# ----- SIR -----
plt.plot(df_br["data"], df_br["infectados"], label="Infectados Reais")
plt.plot(df_br["data"], I, label="Modelo SIR – Infectados")
plt.title("Modelo SIR – Brasil (2020–2024)")
plt.xlabel("Data")
plt.ylabel("Casos")
plt.legend()
plt.grid()
plt.show()

# ----- SEIR -----
plt.plot(df_br["data"], I2, label="Modelo SEIR – Infectados")
plt.plot(df_br["data"], E2, label="Expostos (SEIR)")
plt.title("Modelo SEIR – Brasil (2020–2024)")
plt.xlabel("Data")
plt.ylabel("Casos")
plt.legend()
plt.grid()
plt.show()

# ----- ML -----
plt.plot(df_br["data"].iloc[len(X_train):], y_test, label="Real")
plt.plot(df_br["data"].iloc[len(X_train):], pred, label="Previsão")
plt.title("Machine Learning – Random Forest")
plt.xlabel("Data")
plt.ylabel("Casos Novos")
plt.legend()
plt.grid()
plt.show()

# ----- Previsão Futura -----
plt.plot(dias_futuros, pred_futuro, label="Previsão 60 dias")
plt.title("Previsão de Casos – 60 dias")
plt.xlabel("Dias Futuros")
plt.ylabel("Casos Previstos")
plt.legend()
plt.grid()
plt.show()
