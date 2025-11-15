import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import streamlit as st

class MercadoTrabalhoPredictor:
    def __init__(self, csv_files: list, codigos_filepath: str):
        self.csv_files = csv_files
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(valor)

    def carregar_dados(self):
        dfs = []
        for path in self.csv_files:
            df_temp = pd.read_csv(path, encoding='utf-8', sep=';', on_bad_lines='skip')
            dfs.append(df_temp)

        self.df = pd.concat(dfs, ignore_index=True)

        # Carrega tabela CBO
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
        self.df_codigos['cbo_codigo'] = self.df_codigos['cbo_codigo'].astype(str)

        self.cleaned = True

    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        if not self.cleaned:
            return pd.DataFrame()

        if entrada.isdigit():
            return self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]

        mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5,10,15,20]):
        df = self.df
        col_cbo = "cbo2002ocupação"
        col_data = "competênciamov"
        col_salario = "salário"

        # Nome da profissão
        prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
        st.subheader(f"Profissão: {prof_info.iloc[0]['cbo_descricao']}" if not prof_info.empty else f"CBO: {cbo_codigo}")

        # Filtragem
        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para essa profissão.")
            return

        st.write(f"**Registros encontrados:** {len(df_cbo):,}")

        # -------- DEMOGRAFIA --------
        with st.expander("Perfil Demográfico"):
            # idade
            if 'idade' in df_cbo.columns:
                idade = pd.to_numeric(df_cbo['idade'], errors='coerce')
                st.write(f"Idade média: {idade.mean():.1f} anos")

            # sexo
            if 'sexo' in df_cbo.columns:
                sexo_dist = df_cbo['sexo'].value_counts()
                sexo_map = {'1': 'Masculino', '3': 'Feminino'}
                sexo_lista = [
                    f"{sexo_map.get(str(k), str(k))}: {(v/len(df_cbo))*100:.1f}%"
                    for k, v in sexo_dist.items()
                ]
                st.write("Distribuição por sexo: " + ", ".join(sexo_lista))

            # escolaridade
            if 'graudeinstrucao' in df_cbo.columns:
                esc_map = {
                    '1': 'Analfabeto','2': 'Até 5ª inc. Fund.','3': '5ª comp. Fund.',
                    '4': '6ª a 9ª Fund.','5': 'Fund. completo','6': 'Médio incompleto',
                    '7': 'Médio completo','8': 'Superior incompleto','9': 'Superior completo',
                    '10': 'Mestrado','11': 'Doutorado','80': 'Pós-graduação'
                }
                esc = df_cbo['graudeinstrucao'].value_counts().head(3)
                esc_txt = [
                    f"{esc_map.get(str(int(float(k))), k)}: {(v/len(df_cbo))*100:.1f}%"
                    for k, v in esc.items()
                ]
                st.write("Principais níveis: " + ", ".join(esc_txt))

        # -------- MERCADO ATUAL --------
        st.subheader("Situação do Mercado de Trabalho")

        saldo_col = "saldomovimentação"
        if saldo_col in df_cbo.columns:
            saldo_total = pd.to_numeric(df_cbo[saldo_col], errors='coerce').sum()

            if saldo_total > 0:
                status = "EXPANSÃO"
            elif saldo_total < 0:
                status = "RETRAÇÃO"
            else:
                status = "ESTABILIDADE"

            st.write(f"Saldo total: {saldo_total:+,.0f}  → **{status}**")

        # -------- PREVISÃO SALARIAL --------
        st.subheader("Previsão Salarial (5, 10, 15 e 20 anos)")

        # limpa salário
        df_cbo[col_salario] = (
            df_cbo[col_salario]
            .astype(str)
            .str.replace(",", ".")
            .str.replace(" ", "")
        )

        df_cbo[col_salario] = pd.to_numeric(df_cbo[col_salario], errors="coerce")

        # Preenche faltantes com mediana
        mediana_salario = df_cbo[col_salario].median()
        df_cbo[col_salario].fillna(mediana_salario, inplace=True)

        # Datas
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[col_data])

        df_cbo['tempo_meses'] = (
            (df_cbo[col_data].dt.year - 2020) * 12 +
            df_cbo[col_data].dt.month
        )

        # Série mensal
        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()

        salario_atual = df_mensal[col_salario].iloc[-1]
        st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")

        # ---- RMSE ----
        if len(df_mensal) >= 3:
            X = df_mensal[['tempo_meses']]
            y = df_mensal[col_salario]

            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            rmse = math.sqrt(mean_squared_error(y, y_pred))
            st.write(f"**RMSE do modelo (qualidade): R$ {self.formatar_moeda(rmse)}**")
        else:
            st.info("Sem dados suficientes para calcular RMSE.")
            return

        # Previsão futura
        ult_mes = df_mensal["tempo_meses"].max()
        previsoes = []

        for anos in anos_futuros:
            mes_futuro = ult_mes + anos * 12
            pred = model.predict(np.array([[mes_futuro]]))[0]
            variacao = ((pred - salario_atual) / salario_atual) * 100

            previsoes.append([
                anos,
                f"R$ {self.formatar_moeda(pred)}",
                f"{variacao:+.1f}%"
            ])

        st.table(pd.DataFrame(
            previsoes,
            columns=["Anos", "Salário Previsto", "Variação (%)"]
        ))

        # -------- PREVISÃO DE VAGAS --------
        st.subheader("Tendência de Vagas")

        if saldo_col in df_cbo.columns:
            df_saldo = (
                df_cbo.groupby("tempo_meses")[saldo_col]
                .sum()
                .reset_index()
            )

            if len(df_saldo) >= 3:
                Xs = df_saldo[['tempo_meses']]
                ys = df_saldo[saldo_col]

                mod = LinearRegression().fit(Xs, ys)

                ult_mes_saldo = df_saldo["tempo_meses"].max()
                tendencias = []

                for anos in anos_futuros:
                    mes_fut = ult_mes_saldo + anos * 12
                    pred_saldo = mod.predict(np.array([[mes_fut]]))[0]

                    if pred_saldo > 100: status = "ALTA DEMANDA"
                    elif pred_saldo > 50: status = "CRESCIMENTO MODERADO"
                    elif pred_saldo > 0: status = "CRESCIMENTO LEVE"
                    elif pred_saldo > -50: status = "RETRAÇÃO LEVE"
                    elif pred_saldo > -100: status = "RETRAÇÃO MODERADA"
                    else: status = "RETRAÇÃO FORTE"

                    tendencias.append([
                        anos,
                        f"{pred_saldo:+,.0f}",
                        status
                    ])

                st.table(pd.DataFrame(
                    tendencias,
                    columns=["Anos", "Vagas Previstas/mês", "Tendência"]
                ))
            else:
                st.info("Histórico insuficiente para prever vagas.")

# ------------------- STREAMLIT APP -------------------
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsões do Mercado de Trabalho — Jovem Futuro")

csv_files = [
    "2020_PE1.csv","2021_PE1.csv","2022_PE1.csv",
    "2023_PE1.csv","2024_PE1.csv","2025_PE1.csv"
]

codigos_filepath = "cbo.xlsx"

with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(csv_files, codigos_filepath)
    app.carregar_dados()

st.success("Dados carregados com sucesso!")

busca = st.text_input("Digite o nome ou código da profissão:")

if busca:
    resultados = app.buscar_profissao(busca)
    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        cbo_opcao = st.selectbox(
            "Selecione o CBO:",
            resultados['cbo_codigo'] + " - " + resultados['cbo_descricao']
        )
        cbo_codigo = cbo_opcao.split(" - ")[0]

        if st.button("Gerar Análise Completa"):
            app.relatorio_previsao(cbo_codigo, anos_futuros=[5,10,15,20])
