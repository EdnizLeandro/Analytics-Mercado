# app.py
# ==========================================
# 📊 Jobin – Analytics & Mercado Dashboard
# ==========================================
# Autor: [Seu Nome]
# Descrição: Dashboard interativo em Streamlit
# para análise e previsão do mercado de trabalho por profissão (CBO)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================
# Funções utilitárias
# ==========================
def formatar_moeda(valor):
    """Formata valor para padrão brasileiro"""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ==========================
# Classe de processamento
# ==========================
class MercadoTrabalho:
    def __init__(self, df):
        self.df = df
        self._identificar_colunas()

    def _identificar_colunas(self):
        self.coluna_cbo = next((c for c in self.df.columns if "cbo" in c.lower() and "ocupa" in c.lower()), None)
        self.coluna_data = next((c for c in self.df.columns if "competencia" in c.lower()), None)
        self.coluna_salario = next((c for c in self.df.columns if "salario" in c.lower() and "fixo" in c.lower()), None)

    def filtrar_cbo(self, cbo_codigo):
        return self.df[self.df[self.coluna_cbo].astype(str) == str(cbo_codigo)].copy()

    def prever_salario(self, df_cbo, anos_futuros=[5, 10, 15, 20]):
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        df_cbo["tempo_meses"] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                 df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby("tempo_meses")[self.coluna_salario].mean().reset_index()
        salario_atual = df_cbo[self.coluna_salario].mean()

        if len(df_mensal) < 2:
            return pd.DataFrame({
                "Anos Futuro": anos_futuros,
                "Salário Previsto": [salario_atual]*len(anos_futuros)
            })

        X = df_mensal[["tempo_meses"]]
        y = df_mensal[self.coluna_salario]
        model = LinearRegression()
        model.fit(X, y)
        ult_mes = df_mensal["tempo_meses"].max()

        previsoes = []
        for anos in anos_futuros:
            mes_futuro = ult_mes + anos * 12
            pred = model.predict(np.array([[mes_futuro]]))[0]
            previsoes.append(pred)

        return pd.DataFrame({
            "Anos Futuro": anos_futuros,
            "Salário Previsto": previsoes
        })

# ==========================
# Configuração do App
# ==========================
st.set_page_config(
    page_title="Jobin – Analytics & Mercado",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Jobin – Analytics & Mercado")
st.markdown("""
Plataforma de **inteligência de mercado** para análise e previsão do mercado de trabalho jovem em Recife.  
Selecione os filtros abaixo para explorar dados, tendências e previsões por profissão (CBO).
""")

# ==========================
# Carregamento de dados
# ==========================
try:
    base_path = os.path.dirname(__file__)
    dados_path = os.path.join(base_path, "dados.parquet")
    df = pd.read_parquet(dados_path)
    st.success("✅ Dados carregados com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao carregar os dados: {e}")
    st.stop()

mercado = MercadoTrabalho(df)

# ==========================
# Sidebar - filtros
# ==========================
st.sidebar.header("🔍 Filtros")
profissoes_disponiveis = sorted(df[mercado.coluna_cbo].astype(str).unique())
cbo_codigo = st.sidebar.selectbox("Selecione o código CBO da profissão:", profissoes_disponiveis)
anos_futuros = st.sidebar.multiselect(
    "Períodos de previsão (anos):", options=[5,10,15,20], default=[5,10]
)

# ==========================
# Filtrar dados
# ==========================
df_cbo = mercado.filtrar_cbo(cbo_codigo)

if df_cbo.empty:
    st.warning("Nenhum registro encontrado para essa profissão.")
    st.stop()

# ==========================
# Indicadores gerais
# ==========================
st.subheader(f"📌 Profissão selecionada: {cbo_codigo}")

salario_medio = df_cbo[mercado.coluna_salario].mean()
saldo_total = df_cbo["saldomovimentacao"].sum() if "saldomovimentacao" in df_cbo.columns else np.nan

col1, col2, col3 = st.columns(3)
col1.metric("Salário médio", formatar_moeda(salario_medio))
col2.metric("Saldo de movimentação", f"{saldo_total:+,.0f}" if not np.isnan(saldo_total) else "N/A")
col3.metric("Registros analisados", f"{len(df_cbo):,}")

# ==========================
# Gráficos e análises
# ==========================
st.markdown("### 📊 Análises e Visualizações")
tab1, tab2, tab3 = st.tabs(["💰 Salário", "📅 Tendência de Vagas", "🌎 Distribuição Geográfica"])

with tab1:
    df_prev = mercado.prever_salario(df_cbo, anos_futuros)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prev["Anos Futuro"],
        y=df_prev["Salário Previsto"],
        mode="lines+markers",
        name="Previsão Salarial",
        line=dict(color="#4CAF50", width=3)
    ))
    fig.update_layout(
        title="Previsão de Salário Médio por Ano",
        xaxis_title="Anos no Futuro",
        yaxis_title="Salário Previsto (R$)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "saldomovimentacao" in df_cbo.columns:
        df_cbo["ano"] = pd.to_datetime(df_cbo[mercado.coluna_data], errors='coerce').dt.year
        df_saldo = df_cbo.groupby("ano")["saldomovimentacao"].sum().reset_index()
        fig2 = px.bar(df_saldo, x="ano", y="saldomovimentacao",
                      title="Saldo de Vagas por Ano",
                      labels={"saldomovimentacao":"Saldo de Vagas","ano":"Ano"},
                      color="saldomovimentacao", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Coluna de movimentação não disponível.")

with tab3:
    if "uf" in df_cbo.columns:
        df_geo = df_cbo["uf"].value_counts().reset_index()
        df_geo.columns = ["UF","Quantidade"]
        fig3 = px.choropleth(
            df_geo,
            locations="UF",
            locationmode="ISO-3",
            color="Quantidade",
            title="Distribuição Geográfica por UF"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Dados geográficos não disponíveis.")
