# ==========================================================
# 📊 Aplicativo Streamlit - Mercado de Trabalho (Versão Estável)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import openpyxl

# ==============================
# Configurações gerais
# ==============================
st.set_page_config(page_title="Mercado de Trabalho", layout="wide")
st.title("📈 Análise e Previsão do Mercado de Trabalho no Brasil")

st.sidebar.header("⚙️ Configurações")

# ==============================
# Carregar dados
# ==============================
@st.cache_data(show_spinner=True)
def carregar_dados():
    df = pd.read_parquet("dados.parquet")
    cbo = pd.read_excel("CBO.xlsx")
    return df, cbo

try:
    df, cbo = carregar_dados()
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

# ==============================
# Validação das colunas
# ==============================
colunas_necessarias = {"profissao", "data", "valor"}
if not colunas_necessarias.issubset(df.columns):
    st.error("O arquivo 'dados.parquet' deve conter as colunas: 'profissao', 'data' e 'valor'.")
    st.stop()

if "Descricao" not in cbo.columns:
    st.error("O arquivo 'CBO.xlsx' deve conter a coluna 'Descricao'.")
    st.stop()

# ==============================
# Filtro de profissão
# ==============================
profissoes = sorted(cbo["Descricao"].dropna().unique().tolist())
prof = st.sidebar.selectbox("Selecione uma profissão:", profissoes)

dados_prof = df[df["profissao"] == prof].copy()
if dados_prof.empty:
    st.warning("Nenhum dado encontrado para essa profissão.")
    st.stop()

# ==============================
# Tratamento e exibição dos dados
# ==============================
dados_prof["data"] = pd.to_datetime(dados_prof["data"])
dados_prof = dados_prof.sort_values("data")

st.subheader(f"📊 Histórico — {prof}")
st.dataframe(dados_prof.tail())

fig_hist = px.line(
    dados_prof, x="data", y="valor",
    title=f"Evolução histórica — {prof}",
    markers=True,
    template="plotly_white"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ==============================
# Modelo XGBoost
# ==============================
st.subheader("🤖 Previsão com XGBoost (12 meses)")

# Criar variáveis explicativas
dados_prof["ano"] = dados_prof["data"].dt.year
dados_prof["mes"] = dados_prof["data"].dt.month

X = dados_prof[["ano", "mes"]]
y = dados_prof["valor"]

# Treinamento
modelo = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
modelo.fit(X, y)

# Gerar previsões futuras
ultimo_ano, ultimo_mes = dados_prof["ano"].max(), dados_prof["mes"].max()
futuro = []
for _ in range(12):
    ultimo_mes += 1
    if ultimo_mes > 12:
        ultimo_mes = 1
        ultimo_ano += 1
    futuro.append({"ano": ultimo_ano, "mes": ultimo_mes})

futuro_df = pd.DataFrame(futuro)
futuro_df["valor_previsto"] = modelo.predict(futuro_df)
futuro_df["data"] = pd.to_datetime(futuro_df["ano"].astype(str) + "-" + futuro_df["mes"].astype(str) + "-01")

# Exibir gráfico de previsão
fig_prev = px.line(
    futuro_df, x="data", y="valor_previsto",
    title=f"Previsão — {prof} (Próximos 12 meses)",
    markers=True,
    template="plotly_white"
)
st.plotly_chart(fig_prev, use_container_width=True)

# ==============================
# Avaliação do modelo
# ==============================
y_pred = modelo.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write(f"📏 **MAE (erro médio absoluto):** {mae:,.2f}")
st.write(f"📈 **R² (coeficiente de determinação):** {r2:.3f}")

# ==============================
# Download dos resultados
# ==============================
csv = futuro_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Baixar previsões (CSV)",
    data=csv,
    file_name=f"previsoes_{prof}.csv",
    mime="text/csv"
)

st.success("✅ Previsões geradas com sucesso!")
