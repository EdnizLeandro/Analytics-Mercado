# ==========================================================
# 📊 Aplicativo Streamlit - Previsão do Mercado de Trabalho
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pmdarima as pm
import joblib
from tqdm import tqdm
import plotly.express as px

# ==============================
# Configurações iniciais
# ==============================
st.set_page_config(page_title="Análise do Mercado de Trabalho", layout="wide")
st.title("📈 Análise e Previsão do Mercado de Trabalho no Brasil")

st.sidebar.header("⚙️ Configurações do Modelo")

# ==============================
# Carregamento de dados
# ==============================
@st.cache_data
def carregar_dados():
    df = pd.read_parquet("dados.parquet")
    cbo = pd.read_excel("CBO.xlsx")
    return df, cbo

try:
    df, cbo = carregar_dados()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# ==============================
# Seleção de profissão
# ==============================
if "Descricao" not in cbo.columns:
    st.error("O arquivo CBO.xlsx precisa conter a coluna 'Descricao'.")
    st.stop()

profissoes = sorted(cbo["Descricao"].dropna().unique().tolist())
profissao_escolhida = st.sidebar.selectbox("Selecione uma profissão", profissoes)

# ==============================
# Filtrar dados
# ==============================
if "profissao" not in df.columns:
    st.error("O arquivo dados.parquet precisa conter a coluna 'profissao'.")
    st.stop()

filtro = df[df["profissao"] == profissao_escolhida]

if filtro.empty:
    st.warning("Nenhum dado encontrado para a profissão selecionada.")
    st.stop()

# ==============================
# Exibir dados
# ==============================
st.subheader(f"📊 Dados Históricos — {profissao_escolhida}")

if "data" not in filtro.columns or "valor" not in filtro.columns:
    st.error("O dataset deve conter as colunas 'data' e 'valor'.")
    st.stop()

filtro["data"] = pd.to_datetime(filtro["data"])
filtro = filtro.sort_values("data")

st.dataframe(filtro.head())

# ==============================
# Gráfico Histórico
# ==============================
st.subheader("📅 Evolução Histórica")
fig = px.line(filtro, x="data", y="valor", title=f"Evolução da variável para {profissao_escolhida}")
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Modelagem - Prophet
# ==============================
st.subheader("🔮 Previsão com Facebook Prophet")

df_prophet = filtro[["data", "valor"]].rename(columns={"data": "ds", "valor": "y"})

try:
    modelo_prophet = Prophet()
    modelo_prophet.fit(df_prophet)

    futuro = modelo_prophet.make_future_dataframe(periods=12, freq="M")
    previsao = modelo_prophet.predict(futuro)

    fig1 = modelo_prophet.plot(previsao)
    st.pyplot(fig1)

except Exception as e:
    st.error(f"Erro ao executar Prophet: {e}")

# ==============================
# Modelagem - XGBoost Regressor
# ==============================
st.subheader("🤖 Previsão com XGBoost")

filtro["mes"] = filtro["data"].dt.month
filtro["ano"] = filtro["data"].dt.year
X = filtro[["ano", "mes"]]
y = filtro["valor"]

modelo_xgb = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    max_depth=5
)
modelo_xgb.fit(X, y)

# Criar previsões para 12 meses à frente
ult_ano, ult_mes = filtro["ano"].max(), filtro["mes"].max()
previsoes = []
for i in range(12):
    ult_mes += 1
    if ult_mes > 12:
        ult_mes = 1
        ult_ano += 1
    previsoes.append({"ano": ult_ano, "mes": ult_mes})

previsoes_df = pd.DataFrame(previsoes)
previsoes_df["valor_previsto"] = modelo_xgb.predict(previsoes_df)

# Converter para datas e exibir gráfico
previsoes_df["data"] = pd.to_datetime(previsoes_df["ano"].astype(str) + "-" + previsoes_df["mes"].astype(str) + "-01")

fig2 = px.line(previsoes_df, x="data", y="valor_previsto", title="Previsão com XGBoost (12 meses futuros)")
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Download dos resultados
# ==============================
st.subheader("📥 Download dos Resultados")

previsoes_csv = previsoes_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Baixar previsões (CSV)",
    data=previsoes_csv,
    file_name=f"previsoes_{profissao_escolhida}.csv",
    mime="text/csv"
)

st.success("✅ Previsões geradas com sucesso!")
