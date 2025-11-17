import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Previsão de Salários por Profissão")

# --- CARREGAR DADOS ---
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Converter 'competênciamov' para datetime
    df["competênciamov"] = df["competênciamov"].astype(str)
    df["data"] = pd.to_datetime(df["competênciamov"], format="%Y%m")

    st.write("Primeiras linhas do dataset:")
    st.dataframe(df.head())

    # --- SELEÇÃO DE PROFISSÃO ---
    profissao = st.selectbox("Selecione a profissão:", df["profissao"].unique())
    df_prof = df[df["profissao"] == profissao].copy()
    
    st.write(f"Dados filtrados para a profissão: **{profissao}**")

    # --- PREPARAR DADOS ---
    # Transformar datetime em número para os modelos
    df_prof["data_num"] = df_prof["data"].map(pd.Timestamp.toordinal)
    X = df_prof[["data_num"]]
    y = df_prof["salario"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # --- TREINAR VÁRIOS MODELOS ---
    modelos = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression()
    }

    resultados = {}
    for nome, model in modelos.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        resultados[nome] = {"modelo": model, "rmse": rmse}

    # --- ESCOLHER MELHOR MODELO ---
    melhor_nome = min(resultados, key=lambda x: resultados[x]["rmse"])
    melhor_modelo = resultados[melhor_nome]["modelo"]
    melhor_rmse = resultados[melhor_nome]["rmse"]

    st.write(f"✅ Melhor modelo: **{melhor_nome}** com RMSE = **{melhor_rmse:.2f}**")

    # --- PREDIÇÃO E TENDÊNCIA ---
    df_prof["predicao"] = melhor_modelo.predict(X)

    # Previsão futura (ex.: 12 meses)
    ult_data = df_prof["data"].max()
    datas_fut = pd.date_range(start=ult_data + pd.DateOffset(months=1), periods=12, freq='M')
    datas_fut_num = datas_fut.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    pred_fut = melhor_modelo.predict(datas_fut_num)

    # --- GRÁFICO ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prof["data"], df_prof["salario"], label="Salário Real", marker='o')
    ax.plot(df_prof["data"], df_prof["predicao"], label="Predição Modelo", linestyle="--")
    ax.plot(datas_fut, pred_fut, label="Projeção Futura", linestyle=":", color="red")
    ax.set_title(f"Tendência do Salário para {profissao}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Salário")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("""
    **Explicação do gráfico:**  
    - Linha sólida: valores reais de salário.  
    - Linha tracejada: valores previstos pelo modelo treinado.  
    - Linha pontilhada vermelha: projeção futura para os próximos 12 meses.  
    - RMSE indica o erro médio das previsões; quanto menor, melhor o modelo.
    """)
