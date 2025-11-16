import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

# ---------------------------------------
# Função para remover acentos (sem unidecode)
# ---------------------------------------
def normalizar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# ---------------------------------------
# Carregar CBO
# ---------------------------------------
@st.cache_data
def carregar_dados_cbo(cbo_path="cbo.xlsx"):
    df = pd.read_excel(cbo_path)
    df.columns = ["Código", "Descrição"]
    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)
    return df

# ---------------------------------------
# Carregar histórico
# ---------------------------------------
@st.cache_data
def carregar_historico(path="dados.parquet"):
    df = pd.read_parquet(path)
    df["cbo2002ocupação"] = df["cbo2002ocupação"].astype(str).str.strip()
    df["salário"] = pd.to_numeric(df["salário"], errors="coerce").fillna(0)
    return df

# ---------------------------------------
# Busca profissional
# ---------------------------------------
def buscar_profissao(df_cbo, entrada):
    entrada_norm = normalizar(entrada)

    if entrada.isdigit():
        return df_cbo[df_cbo["Código"] == entrada]

    return df_cbo[df_cbo["Descrição_norm"].str.contains(entrada_norm)]

# ---------------------------------------
# Previsão salarial (simples)
# ---------------------------------------
def prever_salario(salario_atual):
    anos = [5, 10, 15, 20]
    taxa = 0.02  # 2% ao ano
    return {ano: salario_atual * ((1 + taxa) ** ano) for ano in anos}

# ---------------------------------------
# Tendência de mercado
# ---------------------------------------
def tendencia_mercado(df, cbo):
    df_cbo = df[df["cbo2002ocupação"] == cbo]
    if df_cbo.empty:
        return "Sem dados suficientes", {5: 0, 10: 0, 15: 0, 20: 0}

    saldo_medio = df_cbo["saldomovimentação"].mean()

    if saldo_medio > 10:
        status = "CRESCIMENTO ACELERADO"
    elif saldo_medio > 0:
        status = "CRESCIMENTO LEVE"
    elif saldo_medio < -10:
        status = "QUEDA ACELERADA"
    elif saldo_medio < 0:
        status = "QUEDA LEVE"
    else:
        status = "ESTÁVEL"

    previsao = {ano: int(saldo_medio) for ano in [5,10,15,20]}

    return status, previsao

# ==================================================
#                   STREAMLIT APP
# ==================================================
st.set_page_config(page_title="Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

df_cbo = carregar_dados_cbo()
df_hist = carregar_historico()

entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    resultados = buscar_profissao(df_cbo, entrada)

    if resultados.empty:
        st.error("Profissão não encontrada. Digite outro nome ou código.")
        st.stop()

    if len(resultados) > 1:
        st.warning("Foram encontradas várias profissões. Selecione uma:")
        escolha = st.selectbox(
            "Selecione a profissão:",
            resultados["Descrição"] + " (" + resultados["Código"] + ")"
        )
        codigo_escolhido = escolha.split("(")[-1].replace(")","").strip()
    else:
        codigo_escolhido = resultados.iloc[0]["Código"]

    desc = resultados[resultados["Código"]==codigo_escolhido]["Descrição"].values[0]

    st.subheader(f"Profissão: {desc}")

    df_cbo_hist = df_hist[df_hist["cbo2002ocupação"] == codigo_escolhido]

    if df_cbo_hist.empty:
        st.error("Sem dados históricos para calcular salário.")
        st.stop()

    salario_atual = df_cbo_hist["salário"].mean()
    st.write(f"Salário médio atual: **R$ {salario_atual:,.2f}**")

    # PREVISÃO SALARIAL
    previsoes = prever_salario(salario_atual)

    st.markdown("### 📈 Previsão salarial futura:")

    for ano, valor in previsoes.items():
        st.write(f"**{ano} anos → R$ {valor:,.2f}**")

    st.write("*Tendência de crescimento do salário no longo prazo.*")

    # TENDÊNCIA DE MERCADO
    st.markdown("---")
    st.markdown("## 🧭 TENDÊNCIA DE MERCADO PARA A PROFISSÃO")

    status, vagas = tendencia_mercado(df_hist, codigo_escolhido)

    st.write(f"Situação histórica recente: **{status}**")
    st.write("### Projeção de saldo de vagas:")

    for ano, val in vagas.items():
        seta = "↑" if val > 0 else "↓" if val < 0 else "→"
        st.write(f"**{ano} anos: {val} ({seta})**")
