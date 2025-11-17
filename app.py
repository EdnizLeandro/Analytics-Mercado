import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------
# FUNÇÃO PARA NORMALIZAR TEXTO
# ----------------------------------------------------------
def normalizar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# ----------------------------------------------------------
# CARREGAR CBO
# ----------------------------------------------------------
@st.cache_data
def carregar_dados_cbo():
    df = pd.read_excel("cbo.xlsx")
    df.columns = ["Código", "Descrição"]

    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)
    return df

# ----------------------------------------------------------
# CARREGAR HISTÓRICO
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# BUSCA PROFISSÕES
# ----------------------------------------------------------
def buscar_profissoes(df_cbo, texto):
    tnorm = normalizar(texto)
    if texto.isdigit():
        return df_cbo[df_cbo["Código"] == texto]
    return df_cbo[df_cbo["Descrição_norm"].str.contains(tnorm, na=False)]

# ----------------------------------------------------------
# CRIAR COLUNA DE DATA DE FORMA INTELIGENTE
# ----------------------------------------------------------
def criar_datas_seguras(df):
    df_sal = df.copy()
    df_sal["y"] = df_sal.iloc[:, 0]

    # 1) Se existir ano/mes no parquet
    col_ano = next((c for c in df_sal.columns if "ano" in c), None)
    col_mes = next((c for c in df_sal.columns if "mes" in c), None)

    if col_ano and col_mes:
        df_sal["data"] = pd.to_datetime(
            df_sal[col_ano].astype(str) + "-" + df_sal[col_mes].astype(str) + "-01"
        )
        return df_sal[["data", "y"]]

    # 2) Se existir 'competencia' tipo 202001
    if "competencia" in df_sal.columns:
        df_sal["data"] = pd.to_datetime(df_sal["competencia"].astype(str), format="%Y%m")
        return df_sal[["data", "y"]]

    # 3) Gerar datas artificiais seguras
    start_year = 2010
    max_year = 2100
    n = len(df_sal)

    end_year = min(start_year + n // 12, max_year)

    datas = pd.date_range(
        start=f"{start_year}-01-01", 
        end=f"{end_year}-12-01", 
        freq="M"
    )
    datas = list(datas)[:n]

    df_sal["data"] = datas
    return df_sal[["data", "y"]]

# ----------------------------------------------------------
# TREINAMENTO DO MELHOR MODELO
# ----------------------------------------------------------
def treinar_e_escolher_melhor_modelo(df):
    df = df.sort_values("data").dropna()

    if len(df) < 24:
        return None

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    valid = df.iloc[split:]

    results = {}

    # PROPHET
    try:
        prophet_df = train.rename(columns={"data": "ds", "y": "y"})
        model_prophet = Prophet()
        model_prophet.fit(prophet_df)

        future = valid.rename(columns={"data": "ds"})
        pred = model_prophet.predict(future)["yhat"].values

        rmse = np.sqrt(mean_squared_error(valid["y"].values, pred))
        results["prophet"] = (rmse, model_prophet)
    except:
        pass

    # XGBOOST
    try:
        df_ml = df.copy()
        df_ml["mes"] = df_ml["data"].dt.month
        df_ml["ano"] = df_ml["data"].dt.year

        train_ml = df_ml.iloc[:split]
        valid_ml = df_ml.iloc[split:]

        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05)
        xgb.fit(train_ml[["mes", "ano"]], train_ml["y"])

        pred = xgb.predict(valid_ml[["mes", "ano"]])
        rmse = np.sqrt(mean_squared_error(valid_ml["y"], pred))
        results["xgboost"] = (rmse, xgb)
    except:
        pass

    if not results:
        return None

    best_name = min(results, key=lambda m: results[m][0])
    rmse, model = results[best_name]

    return {"modelo_nome": best_name, "melhor_modelo": model, "rmse": rmse}

# ----------------------------------------------------------
# PREVISÃO
# ----------------------------------------------------------
def prever(modelo, modelo_nome, df, anos=20):
    MAX_YEAR = 2100
    start = df["data"].max()

    if start.year >= MAX_YEAR:
        start = pd.Timestamp(f"{MAX_YEAR}-01-01")

    n_periods = anos * 12 + 1
    datas = pd.date_range(start=start, periods=n_periods, freq="M")

    if modelo_nome == "prophet":
        future = modelo.make_future_dataframe(periods=anos * 12, freq="M")
        fc = modelo.predict(future)
        return fc[["ds", "yhat"]].rename(columns={"ds": "data", "yhat": "y"})

    if modelo_nome == "xgboost":
        temp = pd.DataFrame({"data": datas[1:]})
        temp["mes"] = temp["data"].dt.month
        temp["ano"] = temp["data"].dt.year
        temp["y"] = modelo.predict(temp[["mes", "ano"]])
        return temp

# ----------------------------------------------------------
# INTERFACE STREAMLIT
# ----------------------------------------------------------
st.set_page_config(page_title="Mercado de Trabalho - IA", layout="wide")
st.title("Previsão Inteligente do Mercado de Trabalho (CAGED + IA)")

df_cbo = carregar_dados_cbo()
df_hist, COL_CBO, COL_SALARIO = carregar_historico()

entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    res = buscar_profissoes(df_cbo, entrada)
    if res.empty:
        st.warning("Nenhuma profissão encontrada.")
        st.stop()
    lista = (res["Descrição"] + " (" + res["Código"] + ")").tolist()
else:
    lista = []

escolha = st.selectbox("Selecione a profissão:", [""] + lista)

if escolha:
    cbo = escolha.split("(")[-1].replace(")", "").strip()
    desc = escolha.split("(")[0].strip()

    st.header(f"Profissão: {desc}")

    dados = df_hist[df_hist[COL_CBO] == cbo]

    if dados.empty:
        st.error("Sem dados.")
        st.stop()

    # CRIAR SÉRIE COM DATAS SEGURAS
    df_sal = criar_datas_seguras(dados[[COL_SALARIO]])

    st.subheader("Treinando modelos...")

    modelo = treinar_e_escolher_melhor_modelo(df_sal)

    if modelo is None:
        st.error("Dados insuficientes.")
        st.stop()

    st.success(f"Modelo escolhido: **{modelo['modelo_nome']}** (RMSE: {modelo['rmse']:.2f})")

    previsao = prever(modelo["melhor_modelo"], modelo["modelo_nome"], df_sal)

    st.subheader("Previsão de até 20 anos")
    st.line_chart(previsao.set_index("data")["y"])
