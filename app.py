import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# CONFIGURAÇÃO GERAL
# ======================================================
st.set_page_config(page_title="Plataforma Jovem Futuro", layout="wide")

st.title("📊 Plataforma Jovem Futuro — Inteligência de Mercado e Profissões")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

# ======================================================
# 1) CARREGAMENTO DE ARQUIVOS (ROBUSTO + CACHEADO)
# ======================================================
@st.cache_data(show_spinner=True)
def load_dataset():
    # Valida parquet
    if not os.path.exists(PARQUET_FILE):
        st.error(f"❌ Arquivo não encontrado: **{PARQUET_FILE}**")
        st.stop()

    # Valida CBO
    if not os.path.exists(CBO_FILE):
        st.error(f"❌ Arquivo não encontrado: **{CBO_FILE}**")
        st.stop()

    df = pd.read_parquet(PARQUET_FILE)

    df_cbo = pd.read_excel(CBO_FILE)
    df_cbo.columns = ["codigo", "descricao"]

    return df, df_cbo


df, df_cbo = load_dataset()
st.success("✅ Dados carregados com sucesso!")


# ======================================================
# 2) VALIDAÇÃO DE COLUNAS OBRIGATÓRIAS
# ======================================================
REQUIRED_COLUMNS = [
    "cbo2002ocupacao",
    "competenciadec",
    "salario",
    "saldomovimentacao",
]

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing_cols:
    st.error(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
    st.stop()


# Normalização
df["competenciadec"] = pd.to_datetime(df["competenciadec"], errors="coerce")

st.write("### 🔍 Colunas detectadas no dataset:")
st.json(list(df.columns))


# ======================================================
# 3) BUSCA POR PROFISSÃO (CBO)
# ======================================================
st.header("🔎 Buscar profissão (por nome ou código CBO)")

query = st.text_input("Digite nome ou código da profissão:")

if query:
    mask = (
        df_cbo["descricao"].str.contains(query, case=False, na=False)
        | df_cbo["codigo"].astype(str).str.contains(query, na=False)
    )

    resultados = df_cbo[mask]

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        st.write("### Resultados encontrados:")
        st.dataframe(resultados, use_container_width=True)

        cbo_selected = st.selectbox(
            "Selecione um código CBO para análise:",
            resultados["codigo"].astype(str).unique(),
        )

        if cbo_selected:

            st.info(f"📌 Mostrando análise completa para CBO **{cbo_selected}**")

            df_job = df[df["cbo2002ocupacao"].astype(str) == cbo_selected]

            if df_job.empty:
                st.warning("⚠️ Não há registros para esse CBO.")
                st.stop()

            # ======================================================
            # 4) ANÁLISE EXPLORATÓRIA
            # ======================================================
            st.subheader("📊 Estatísticas Gerais")

            col1, col2, col3 = st.columns(3)
            col1.metric("Média Salarial", f"R$ {df_job['salario'].mean():,.2f}")
            col2.metric("Mediana Salarial", f"R$ {df_job['salario'].median():,.2f}")
            col3.metric("Salário Máximo", f"R$ {df_job['salario'].max():,.2f}")

            st.write("### Distribuição Salarial (Boxplot)")
            fig_box = px.box(df_job, y="salario", color="cbo2002ocupacao",
                             title="Distribuição Salarial")
            st.plotly_chart(fig_box, use_container_width=True)

            st.write("### Evolução do Saldo de Contratações")
            fig_line = px.line(df_job, x="competenciadec", y="saldomovimentacao",
                               title="Evolução Mensal")
            st.plotly_chart(fig_line, use_container_width=True)

            # ======================================================
            # 5) PREVISÃO — ML (PROPHET OU LSTM)
            # ======================================================
            st.subheader("🤖 Previsão de demanda futura")

            model_type = st.radio(
                "Escolha o modelo de previsão:",
                ["Prophet (Recomendado)", "LSTM Neural Network"]
            )

            df_ml = df_job[["competenciadec", "saldomovimentacao"]].dropna()

            df_ml = df_ml.rename(columns={"competenciadec": "ds", "saldomovimentacao": "y"})

            if len(df_ml) < 12:
                st.warning("⚠️ Dados insuficientes para previsão (mínimo 12 registros).")
                st.stop()

            # ======================================================
            # PROPHET
            # ======================================================
            if model_type == "Prophet (Recomendado)":
                model = Prophet()
                model.fit(df_ml)

                future = model.make_future_dataframe(periods=12, freq="M")
                forecast = model.predict(future)

                st.write("### 📈 Previsão (Prophet)")
                fig_forecast = model.plot(forecast)
                st.pyplot(fig_forecast)

                st.write("### 🔢 Tabela com previsão dos próximos 12 meses")
                st.dataframe(
                    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12),
                    use_container_width=True
                )

            # ======================================================
            # LSTM (MODELO NEURAL AVANÇADO)
            # ======================================================
            else:
                st.write("🔧 Preparando dados para o modelo LSTM...")

                df_lstm = df_ml.copy()
                df_lstm["ds"] = pd.to_datetime(df_lstm["ds"])
                df_lstm = df_lstm.set_index("ds")

                scaler = MinMaxScaler()
                scaled_values = scaler.fit_transform(df_lstm[["y"]])

                X, y = [], []
                window = 6  # usa 6 meses para prever 1

                for i in range(window, len(scaled_values)):
                    X.append(scaled_values[i-window:i])
                    y.append(scaled_values[i])

                X, y = np.array(X), np.array(y)

                X = X.reshape((X.shape[0], X.shape[1], 1))

                model = Sequential([
                    LSTM(50, return_sequences=True),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])

                model.compile(optimizer="adam", loss="mse")

                st.write("⏳ Treinando modelo LSTM...")
                model.fit(X, y, epochs=40, batch_size=8, verbose=0)

                # Previsões futuras
                last_window = scaled_values[-window:]
                preds = []

                cur = last_window

                for _ in range(12):
                    pred = model.predict(cur.reshape(1, window, 1), verbose=0)
                    preds.append(pred[0][0])
                    cur = np.append(cur[1:], pred, axis=0)

                preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

                st.write("### 📈 Previsão (LSTM)")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(
                    x=df_lstm.index, y=df_lstm["y"], mode="lines", name="Histórico"
                ))
                future_dates = pd.date_range(df_lstm.index[-1], periods=13, freq="M")[1:]
                fig_lstm.add_trace(go.Scatter(
                    x=future_dates, y=preds.flatten(), mode="lines+markers",
                    name="Previsão LSTM"
                ))
                st.plotly_chart(fig_lstm, use_container_width=True)

                st.write("### 🔢 Valores previstos:")
                st.dataframe(pd.DataFrame({"data": future_dates, "previsao": preds.flatten()}))
