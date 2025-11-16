import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import streamlit as st

class MercadoTrabalhoPredictor:
    def __init__(self, parquet_file: str, codigos_filepath: str):
        self.parquet_file = parquet_file
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    def carregar_dados(self):
        # Carrega dados principais
        self.df = pd.read_parquet(self.parquet_file)
        # Garantir CBO como string
        self.df['cbo2002ocupacao'] = self.df['cbo2002ocupacao'].astype(str)

        # Carrega tabela CBO
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        self.df_codigos["cbo_codigo"] = self.df_codigos["cbo_codigo"].astype(str)

        # Preenche salário ausente com mediana
        if "salario" in self.df.columns:
            self.df["salario"] = pd.to_numeric(
                self.df["salario"].astype(str).str.replace(",", "."),
                errors="coerce"
            )
            mediana = self.df["salario"].median()
            self.df["salario"] = self.df["salario"].fillna(mediana)

        # Garantir datetime
        if "competenciamov" in self.df.columns:
            self.df["competenciamov"] = pd.to_datetime(self.df["competenciamov"], errors="coerce")

        self.cleaned = True

    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()

        entrada = entrada.strip()

        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]

        mask = self.df_codigos["cbo_descricao"].str.contains(
            entrada, case=False, na=False
        )
        return self.df_codigos[mask]

    def escolher_melhor_modelo(self, X, y):
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        r2_lr = r2_score(y, y_pred_lr)
        mae_lr = mean_absolute_error(y, y_pred_lr)

        # XGBoost
        xgb = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        xgb.fit(X, y)
        y_pred_xgb = xgb.predict(X)
        r2_xgb = r2_score(y, y_pred_xgb)
        mae_xgb = mean_absolute_error(y, y_pred_xgb)

        if r2_xgb >= r2_lr:
            return "XGBoost", xgb, r2_xgb, mae_xgb
        else:
            return "Linear Regression", lr, r2_lr, mae_lr

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):
        df = self.df
        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        # Nome da profissão
        prof_info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        titulo = prof_info.iloc[0]["cbo_descricao"] if not prof_info.empty else f"CBO {cbo_codigo}"

        st.header(f"Profissão: {titulo}")

        # Filtra registros
        df_cbo = df[df[col_cbo] == cbo_codigo].copy()

        if df_cbo.empty:
            st.warning("Nenhum dado disponível para esta profissão.")
            return

        # Perfil demográfico
        with st.expander("Perfil Demográfico"):
            if "idade" in df_cbo.columns:
                media = pd.to_numeric(df_cbo["idade"], errors="coerce").mean()
                st.write(f"Idade média: **{media:.1f} anos**")

            if "sexo" in df_cbo.columns:
                sexo_map = {"1": "Masculino", "3": "Feminino"}
                contagem = df_cbo["sexo"].astype(str).value_counts()
                txt = ", ".join(
                    f"{sexo_map.get(k,k)}: {(v/len(df_cbo))*100:.1f}%"
                    for k, v in contagem.items()
                )
                st.write("Distribuição por sexo:", txt)

        # Previsão salarial
        st.subheader("Salário médio atual")
        df_cbo = df_cbo.dropna(subset=[col_data, col_salario])
        if df_cbo.empty:
            st.info("Sem dados suficientes para fazer previsões salariais.")
            return

        df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month
        salario_atual = df_cbo[col_salario].mean()
        st.write(f"Salário médio atual: R$ {self.formatar_moeda(salario_atual)}")

        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()
        if len(df_mensal) < 2:
            st.info("Sem dados suficientes para fazer previsões salariais.")
            return

        X = df_mensal[["tempo_meses"]]
        y = df_mensal[col_salario]
        modelo_nome, modelo, r2, mae = self.escolher_melhor_modelo(X, y)

        ult_mes = df_mensal["tempo_meses"].max()
        previsoes = []
        for anos in anos_futuros:
            futuro = ult_mes + anos * 12
            pred = modelo.predict([[futuro]])[0]
            variacao = ((pred - salario_atual) / salario_atual) * 100
            previsoes.append(f"{anos} anos → R$ {self.formatar_moeda(pred)}")

        st.subheader(f"Modelo vencedor: {modelo_nome} (R²={r2*100:.2f}%, MAE={mae:.2f})")
        st.write("Previsão salarial futura do melhor modelo:")
        for linha in previsoes:
            st.write("  ", linha)
        st.write("* Tendência de crescimento do salário no longo prazo.")

        # Previsão de vagas
        st.subheader("Tendência de Mercado (Projeção de demanda)")
        if col_saldo not in df_cbo.columns:
            st.info("Sem dados de movimentação para projeção de vagas.")
            return

        df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()
        if len(df_saldo) < 2:
            st.info("Dados insuficientes para prever vagas.")
            return

        Xs = df_saldo[["tempo_meses"]]
        ys = df_saldo[col_saldo]
        mod_saldo = LinearRegression().fit(Xs, ys)
        ult_mes_s = df_saldo["tempo_meses"].max()

        st.write("="*70)
        st.write("Situação histórica recente:", end=" ")
        historico = ys.iloc[-1]
        if historico > 50:
            st.write("ALTA DEMANDA")
        elif historico > 10:
            st.write("CRESCIMENTO LEVE")
        elif historico > 0:
            st.write("CRESCIMENTO MODERADO")
        else:
            st.write("RETRAÇÃO LEVE")

        st.write("\nProjeção de saldo de vagas (admissões - desligamentos):")
        for anos in anos_futuros:
            futuro = ult_mes_s + anos * 12
            pred = mod_saldo.predict([[futuro]])[0]
            seta = "→"
            st.write(f"  {anos} anos: {int(pred)} ({seta})")


# ---------------------- STREAMLIT ----------------------
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("Previsão do Mercado de Trabalho ( Novo CAGED )")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

# Carregar dados
with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(PARQUET_FILE, CBO_FILE)
    app.carregar_dados()

busca = st.text_input("Digite nome ou código da profissão:")

if busca:
    resultados = app.buscar_profissao(busca)

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        lista = resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]
        escolha = st.selectbox("Selecione o CBO:", lista)
        cbo_codigo = escolha.split(" - ")[0]

        if st.button("Gerar Relatório Completo"):
            app.relatorio_previsao(cbo_codigo)
