import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

from prophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --------------------------------------------
# FUNÇÃO DE NORMALIZAÇÃO DE TEXTO
# --------------------------------------------
def normalizar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# --------------------------------------------
# CARREGAR CBO
# --------------------------------------------
@st.cache_data
def carregar_dados_cbo():
    df = pd.read_excel("cbo.xlsx")
    df.columns = ["Código", "Descrição"]

    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)
    return df

# --------------------------------------------
# CARREGAR HISTÓRICO
# --------------------------------------------
@st.cache_data
def carregar_historico():
    df = pd.read_parquet("dados.parquet")

    cols_norm = {}
    for col in df.columns:
        col_norm = "".join(
            c for c in unicodedata.normalize("NFD", col.lower())
            if unicodedata.category(c) != "Mn"
        )
        cols_norm[col] = col_norm
    df.columns = cols_norm.values()

    col_cbo = next((c for c in df.columns if "cbo" in c), None)
    col_sal = next((c for c in df.columns if "sal" in c), None)

    df[col_cbo] = df[col_cbo].astype(str).str.strip()
    df[col_sal] = pd.to_numeric(df[col_sal], errors="coerce").fillna(0)

    return df, col_cbo, col_sal

# --------------------------------------------
# BUSCA PROFISSÃO
# --------------------------------------------
def buscar_profissoes(df_cbo, texto):
    tnorm = normalizar(texto)
    if texto.isdigit():
        return df_cbo[df_cbo["Código"] == texto]
    return df_cbo[df_cbo["Descrição_norm"].str.contains(tnorm, na=False)]

# ================================================================
# MODELOS DE PREVISÃO
# ================================================================

def treinar_e_escolher_melhor_modelo(df):
    df = df.sort_values("data").dropna()

    if len(df) < 24:
        return None  # dados insuficientes

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    valid = df.iloc[split:]

    y_train = train["y"].values
    y_valid = valid["y"].values

    results = {}

    # ------------------------------
    # PROPHET
    # ------------------------------
    try:
        prophet_df = train.rename(columns={"data": "ds", "y": "y"})
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)

        future = valid.rename(columns={"data": "ds"})
        forecast = prophet_model.predict(future)
        prophet_pred = forecast["yhat"].values

        rmse_prophet = np.sqrt(mean_squared_error(y_valid, prophet_pred))
        results["prophet"] = (rmse_prophet, prophet_model)
    except:
        pass

    # ------------------------------
    # SARIMA
    # ------------------------------
    try:
        sarima_model = auto_arima(train["y"], seasonal=True, m=12)
        sarima_pred = sarima_model.predict(n_periods=len(valid))

        rmse_sarima = np.sqrt(mean_squared_error(y_valid, sarima_pred))
        results["sarima"] = (rmse_sarima, sarima_model)
    except:
        pass

    # ------------------------------
    # XGBOOST
    # ------------------------------
    try:
        df_ml = df.copy()
        df_ml["mes"] = df_ml["data"].dt.month
        df_ml["ano"] = df_ml["data"].dt.year

        train_ml = df_ml.iloc[:split]
        valid_ml = df_ml.iloc[split:]

        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05)
        xgb.fit(train_ml[["mes", "ano"]], train_ml["y"])

        xgb_pred = xgb.predict(valid_ml[["mes", "ano"]])
        rmse_xgb = np.sqrt(mean_squared_error(valid_ml["y"], xgb_pred))
        results["xgboost"] = (rmse_xgb, xgb)
    except:
        pass

    if not results:
        return None

    melhor_modelo_nome = min(results, key=lambda m: results[m][0])
    melhor_rmse, melhor_modelo = results[melhor_modelo_nome]

    return {
        "melhor_modelo": melhor_modelo,
        "modelo_nome": melhor_modelo_nome,
        "rmse": melhor_rmse
    }

# --------------------------------------------
# PREVISÃO COM O MELHOR MODELO
# --------------------------------------------
def prever(melhor_modelo, modelo_nome, df, anos=20):
    if modelo_nome == "prophet":
        future = melhor_modelo.make_future_dataframe(periods=anos * 12, freq="M")
        forecast = melhor_modelo.predict(future)
        return forecast[["ds", "yhat"]].rename(columns={"ds": "data", "yhat": "y"})

    elif modelo_nome == "sarima":
        pred = melhor_modelo.predict(n_periods=anos * 12)
        datas = pd.date_range(start=df["data"].max(), periods=anos*12 + 1, freq="M")
        return pd.DataFrame({"data": datas[1:], "y": pred})

    elif modelo_nome == "xgboost":
        datas = pd.date_range(start=df["data"].max(), periods=anos*12 + 1, freq="M")
        temp = pd.DataFrame({"data": datas[1:]})
        temp["mes"] = temp["data"].dt.month
        temp["ano"] = temp["data"].dt.year
        temp["y"] = melhor_modelo.predict(temp[["mes", "ano"]])
        return temp

# ================================================================
# INTERFACE STREAMLIT
# ================================================================
st.set_page_config(page_title="Mercado de Trabalho - Previsões Inteligentes", layout="wide")
st.title("Previsão Inteligente do Mercado de Trabalho (CAGED + IA)")

df_cbo = carregar_dados_cbo()
df_hist, COL_CBO, COL_SALARIO = carregar_historico()

entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    resultado = buscar_profissoes(df_cbo, entrada)
    if resultado.empty:
        st.warning("Nenhuma profissão encontrada.")
        st.stop()
    lista_profissoes = (resultado["Descrição"] + " (" + resultado["Código"] + ")").tolist()
else:
    lista_profissoes = []

escolha = st.selectbox("Selecione a profissão:", [""] + lista_profissoes)

# ------------------------------
# MOSTRAR RESULTADOS
# ------------------------------
if escolha:
    cbo_codigo = escolha.split("(")[-1].replace(")", "").strip()
    descricao = escolha.split("(")[0].strip()
    st.header(f"Profissão: {descricao}")

    dados_prof = df_hist[df_hist[COL_CBO] == cbo_codigo]

    if dados_prof.empty:
        st.error("Sem dados para esta profissão.")
        st.stop()

    # Preparar série para modelagem
    df_sal = pd.DataFrame({
        "data": pd.to_datetime(dados_prof.index),  # se seu parquet já tiver data, ajuste aqui
        "y": dados_prof[COL_SALARIO].values
    })

    st.subheader("Treinando modelos...")

    modelo = treinar_e_escolher_melhor_modelo(df_sal)

    if modelo is None:
        st.error("Sem dados suficientes para treinar modelos.")
        st.stop()

    st.success(f"Modelo escolhido: **{modelo['modelo_nome']}** (RMSE: {modelo['rmse']:.2f})")

    previsao = prever(modelo["melhor_modelo"], modelo["modelo_nome"], df_sal)

    st.subheader("Previsão de até 20 anos")
    st.line_chart(previsao.set_index("data")["y"])
