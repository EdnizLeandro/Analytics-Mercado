import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
#                    CLASSE PRINCIPAL
# ============================================================
class MercadoTrabalhoPredictor:
    def __init__(self, parquet_file: str, codigos_filepath: str):
        self.parquet_file = parquet_file
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    # ------------------------------------------------------------
    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    # ------------------------------------------------------------
    def carregar_dados(self):
        self.df = pd.read_parquet(self.parquet_file)

        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        self.df_codigos["cbo_codigo"] = self.df_codigos["cbo_codigo"].astype(str)

        # Salário
        if "salario" in self.df.columns:
            self.df["salario"] = (
                self.df["salario"].astype(str).str.replace(",", ".")
            )
            self.df["salario"] = pd.to_numeric(self.df["salario"], errors="coerce")
            self.df["salario"] = self.df["salario"].fillna(self.df["salario"].median())

        self.cleaned = True

    # ------------------------------------------------------------
    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()

        entrada = entrada.strip()

        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]

        mask = self.df_codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    # ------------------------------------------------------------
    def treinar_modelos(self, X, y):
        modelos = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42),
            "XGBoost": XGBRegressor(
                n_estimators=400,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.3,
                objective="reg:squarederror"
            )
        }

        resultados = {}

        for nome, modelo in modelos.items():
            modelo.fit(X, y)
            pred = modelo.predict(X)

            r2 = r2_score(y, pred)
            mae = mean_absolute_error(y, pred)

            resultados[nome] = {
                "modelo": modelo,
                "r2": r2,
                "mae": mae
            }

        # Escolher o melhor (maior R² / menor MAE)
        vencedor = max(resultados.keys(), key=lambda m: (resultados[m]["r2"], -resultados[m]["mae"]))
        return vencedor, resultados[vencedor], resultados

    # ------------------------------------------------------------
    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):
        df = self.df

        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        prof_info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        nome = prof_info.iloc[0]["cbo_descricao"] if not prof_info.empty else f"CBO {cbo_codigo}"

        texto = f"## **Profissão:** {nome}\n\n"

        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.markdown("⚠️ Nenhum dado encontrado para esta profissão.")
            return

        # ===================== SALÁRIOS =============================
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[col_data])

        df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month

        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()

        salario_atual = df_mensal[col_salario].iloc[-1]
        texto += f"**Salário médio atual:** R$ {self.formatar_moeda(salario_atual)}\n\n"

        # ========= TREINAR MODELOS ==========
        X = df_mensal[["tempo_meses"]]
        y = df_mensal[col_salario]

        vencedor_nome, vencedor, _ = self.treinar_modelos(X, y)

        modelo = vencedor["modelo"]
        r2 = vencedor["r2"] * 100
        mae = vencedor["mae"]

        texto += f"**Modelo vencedor:** {vencedor_nome} (R²={r2:.2f}%, MAE={mae:.2f})\n\n"
        texto += "### **Previsão salarial futura do melhor modelo:**\n"

        ult_mes = df_mensal["tempo_meses"].max()

        for anos in anos_futuros:
            futuro = ult_mes + anos * 12
            pred = modelo.predict([[futuro]])[0]
            texto += f"- **{anos} anos → R$ {self.formatar_moeda(pred)}**\n"

        # ===================== TENDÊNCIA DE MERCADO ==========================
        texto += "\n---\n## 📈 **TENDÊNCIA DE MERCADO**\n"

        if col_saldo not in df_cbo.columns:
            texto += "Sem dados de movimentação.\n"
            st.markdown(texto)
            return

        df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()

        if len(df_saldo) < 2:
            texto += "Dados insuficientes para prever vagas.\n"
            st.markdown(texto)
            return

        Xs = df_saldo[["tempo_meses"]]
        ys = df_saldo[col_saldo]

        mod_saldo = LinearRegression().fit(Xs, ys)

        saldo_atual = df_saldo[col_saldo].iloc[-1]
        if saldo_atual > 0:
            status_hist = "CRESCIMENTO LEVE"
        elif saldo_atual == 0:
            status_hist = "ESTABILIDADE"
        else:
            status_hist = "RETRAÇÃO"

        texto += f"**Situação histórica recente:** {status_hist}\n\n"
        texto += "### **Projeção de saldo de vagas:**\n"

        ult_mes_s = df_saldo["tempo_meses"].max()

        for anos in anos_futuros:
            futuro = ult_mes_s + anos * 12
            pred = mod_saldo.predict([[futuro]])[0]
            seta = "→" if abs(pred) < 5 else ("↑" if pred > 0 else "↓")
            texto += f"- **{anos} anos:** {int(pred)} ({seta})\n"

        # ---- Mostrar tudo de uma vez (EVITA O ERRO) ----
        st.markdown(texto)


# ============================================================
#                   STREAMLIT — INTERFACE
# ============================================================
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

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
        escolha = st.selectbox("Selecione a profissão:", lista)

        cbo_codigo = escolha.split(" - ")[0]

        if st.button("Gerar Relatório Completo"):
            app.relatorio_previsao(cbo_codigo)
