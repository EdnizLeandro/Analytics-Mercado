import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================
# CONFIGURAÇÕES INICIAIS DO APP
# ==============================================================

st.set_page_config(
    page_title="Jobin – Analytics & Mercado",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Jobin – Analytics & Mercado")
st.markdown(
    """
    **Iniciativa que transforma a vida de jovens em Recife por meio de dados e inteligência de mercado.**  
    Conectamos talentos a oportunidades reais de trabalho, educação e renda, promovendo inclusão e impacto social.
    """
)

# ==============================================================
# FUNÇÃO DE CARREGAMENTO DE DADOS
# ==============================================================

@st.cache_data
def carregar_dados():
    try:
        base_path = os.path.dirname(__file__)
        dados_path = os.path.join(base_path, "dados.parquet")
        codigos_path = os.path.join(base_path, "dados.parquet")

        df = pd.read_parquet(dados_path)
        df_codigos = pd.read_excel(codigos_path)
        df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        df_codigos["cbo_codigo"] = df_codigos["cbo_codigo"].astype(str)
        st.success("✅ Dados carregados com sucesso!")
        return df, df_codigos
    except Exception as e:
        st.error(f"❌ Erro ao carregar os dados: {e}")
        return None, None


df, df_codigos = carregar_dados()

if df is None or df_codigos is None:
    st.stop()

# ==============================================================
# SEÇÃO DE FILTROS
# ==============================================================

st.sidebar.header("🔍 Filtros")

profissao = st.sidebar.text_input("Digite o nome ou código CBO da profissão:")
anos_previsao = st.sidebar.multiselect(
    "Selecione os horizontes de previsão (anos):",
    [5, 10, 15, 20],
    default=[5, 10]
)

# Identificar colunas
colunas_lower = [c.lower().replace(" ", "") for c in df.columns]
coluna_cbo = next((c for c in df.columns if "cbo" in c.lower() and "ocupa" in c.lower()), None)
coluna_data = next((c for c in df.columns if "competencia" in c.lower()), None)
coluna_salario = next((c for c in df.columns if "salario" in c.lower() and "fixo" in c.lower()), None)

if not all([coluna_cbo, coluna_data, coluna_salario]):
    st.error("❌ Colunas essenciais (CBO, Data, Salário) não encontradas no dataset.")
    st.stop()

# ==============================================================
# BUSCA E SELEÇÃO DE PROFISSÃO
# ==============================================================

if profissao:
    resultados = df_codigos[df_codigos["cbo_descricao"].str.contains(profissao, case=False, na=False) | 
                             df_codigos["cbo_codigo"].str.contains(profissao, case=False, na=False)]

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
        st.stop()
    else:
        cbo_codigo = st.sidebar.selectbox(
            "Selecione o código CBO:",
            resultados["cbo_codigo"],
            format_func=lambda x: f"[{x}] " + resultados.loc[resultados["cbo_codigo"] == x, "cbo_descricao"].values[0]
        )

        cbo_nome = resultados.loc[resultados["cbo_codigo"] == cbo_codigo, "cbo_descricao"].values[0]
        st.subheader(f"📌 Profissão selecionada: **{cbo_nome}**")

        df_cbo = df[df[coluna_cbo].astype(str) == cbo_codigo].copy()

        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para esta profissão.")
            st.stop()

        # ==============================================================
        # ANÁLISES EXPLORATÓRIAS
        # ==============================================================

        st.markdown("### 📈 Análise do Mercado de Trabalho")

        # SALDO DE MOVIMENTAÇÃO
        if "saldomovimentacao" in df_cbo.columns:
            saldo_total = df_cbo["saldomovimentacao"].sum()
            st.metric("Saldo total de movimentação", f"{saldo_total:+,.0f}")
            fig_saldo = px.histogram(df_cbo, x="saldomovimentacao", nbins=50, title="Distribuição do Saldo de Movimentação")
            st.plotly_chart(fig_saldo, use_container_width=True)

        # PERFIL DEMOGRÁFICO
        col1, col2 = st.columns(2)
        if "idade" in df_cbo.columns:
            col1.metric("Idade média", f"{df_cbo['idade'].mean():.1f} anos")
        if "sexo" in df_cbo.columns:
            dist = df_cbo["sexo"].value_counts(normalize=True) * 100
            fig_sexo = px.pie(
                values=dist.values,
                names=["Masculino" if str(i) == "1.0" else "Feminino" for i in dist.index],
                title="Distribuição por Sexo"
            )
            col2.plotly_chart(fig_sexo, use_container_width=True)

        # ==============================================================
        # PREVISÃO SALARIAL
        # ==============================================================

        st.markdown("### 💰 Previsão Salarial")

        df_cbo[coluna_data] = pd.to_datetime(df_cbo[coluna_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[coluna_data])
        df_cbo["tempo_meses"] = ((df_cbo[coluna_data].dt.year - 2020) * 12 + df_cbo[coluna_data].dt.month)

        df_mensal = df_cbo.groupby("tempo_meses")[coluna_salario].mean().reset_index()

        if len(df_mensal) >= 2:
            X = df_mensal[["tempo_meses"]]
            y = df_mensal[coluna_salario]
            model = LinearRegression()
            model.fit(X, y)

            ult_mes = df_mensal["tempo_meses"].max()
            salario_atual = y.mean()

            previsoes = []
            for anos in anos_previsao:
                mes_futuro = ult_mes + anos * 12
                pred = model.predict(np.array([[mes_futuro]]))[0]
                variacao = ((pred - salario_atual) / salario_atual) * 100
                previsoes.append((anos, pred, variacao))

            df_prev = pd.DataFrame(previsoes, columns=["Anos", "Salário Previsto", "Variação (%)"])

            st.dataframe(df_prev.style.format({"Salário Previsto": "R$ {:,.2f}", "Variação (%)": "{:+.1f}%"}))

            fig_prev = px.line(
                df_prev,
                x="Anos",
                y="Salário Previsto",
                markers=True,
                title="Projeção Salarial Futura"
            )
            st.plotly_chart(fig_prev, use_container_width=True)
        else:
            st.warning("Dados insuficientes para previsão salarial.")

        # ==============================================================
        # TENDÊNCIA DO MERCADO
        # ==============================================================

        st.markdown("### 📊 Tendência do Mercado")

        if "saldomovimentacao" in df_cbo.columns:
            df_saldo = df_cbo.groupby("tempo_meses")["saldomovimentacao"].sum().reset_index()
            if len(df_saldo) >= 2:
                X = df_saldo[["tempo_meses"]]
                y = df_saldo["saldomovimentacao"]
                model_saldo = LinearRegression().fit(X, y)
                ult_mes = df_saldo["tempo_meses"].max()

                previsoes_saldo = []
                for anos in anos_previsao:
                    mes_futuro = ult_mes + anos * 12
                    pred = model_saldo.predict(np.array([[mes_futuro]]))[0]
                    previsoes_saldo.append((anos, pred))

                df_tend = pd.DataFrame(previsoes_saldo, columns=["Anos", "Saldo Previsto"])

                fig_tend = px.bar(
                    df_tend,
                    x="Anos",
                    y="Saldo Previsto",
                    title="Tendência de Saldo de Vagas (Projeção)"
                )
                st.plotly_chart(fig_tend, use_container_width=True)
            else:
                st.info("Dados insuficientes para previsão de tendência.")
else:
    st.info("Digite uma profissão na barra lateral para começar a análise.")
