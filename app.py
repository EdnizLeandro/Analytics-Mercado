# ========================================================================
# SISTEMA DE PREVISÃO DO MERCADO DE TRABALHO - VERSÃO ROBUSTA E OTIMIZADA
# ========================================================================

import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ========================================================================
#  🔧 CLASSE PRINCIPAL
# ========================================================================

class MercadoTrabalhoPredictor:
    """
    Sistema de previsão de mercado de trabalho e salário baseado em dados 
    CAGED + CBO, integrado ao Streamlit com análises automáticas.
    """

    def __init__(self, parquet_path: str, codigos_path: str):
        self.parquet_path = parquet_path
        self.codigos_path = codigos_path
        self.df = None
        self.codigos = None
        self.cleaned = False

        # Mapeamentos úteis
        self.column_map = {
            "cbo": ["cbo2002ocupacao", "cbo2002ocupação"],
            "data": ["competenciamov", "competênciamov"],
            "salario": ["salario", "salário"],
            "saldo": ["saldomovimentacao", "saldomovimentação"]
        }

    # ====================================================================
    # 🔍 Funções auxiliares
    # ====================================================================

    def _find_column(self, df: pd.DataFrame, keys: list):
        """Encontra automaticamente a coluna correta, mesmo com acentos."""
        for col in keys:
            if col in df.columns:
                return col
        raise KeyError(f"Colunas esperadas não encontradas: {keys}")

    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    def interpretacao_score(self, score):
        if score > 0.9: return "🟢 Excelente"
        if score > 0.7: return "🟡 Bom"
        if score > 0.5: return "🟠 Moderado"
        return "🔴 Baixo"

    # ====================================================================
    # 📥 Carregamento de dados
    # ====================================================================

    def carregar_dados(self):
        """Carrega parquet + planilha CBO com tratamento robusto."""
        missing = [f for f in [self.parquet_path, self.codigos_path] if not os.path.exists(f)]

        if missing:
            st.error(f"Arquivos ausentes: {missing}")
            return False

        try:
            self.df = pd.read_parquet(self.parquet_path)
            self.codigos = pd.read_excel(self.codigos_path)

            # Padronizar colunas
            self.codigos.columns = ["cbo_codigo", "cbo_descricao"]
            self.codigos["cbo_codigo"] = self.codigos["cbo_codigo"].astype(str)

            # Verificação básica
            assert len(self.df) > 0, "Arquivo parquet vazio!"

            self.cleaned = True
            st.success("Dados carregados com sucesso!")
            return True

        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return False

    # ====================================================================
    # 🔎 Busca da profissão
    # ====================================================================

    def buscar_profissao(self, entrada: str):
        """Busca nome ou código CBO."""
        if not self.cleaned:
            return pd.DataFrame()

        entrada = entrada.strip()

        if entrada.isdigit():
            return self.codigos[self.codigos["cbo_codigo"] == entrada]

        return self.codigos[self.codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)]

    # ====================================================================
    # 📈 PREVISÃO DO MERCADO DE TRABALHO
    # ====================================================================

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):
        """
        Gera relatório completo + previsões de saldo e salário.
        Inclui análises demográficas e tendências futuras.
        """

        # ------------------------
        # Identificação das colunas
        # ------------------------
        col_cbo = self._find_column(self.df, self.column_map["cbo"])
        col_data = self._find_column(self.df, self.column_map["data"])
        col_sal = self._find_column(self.df, self.column_map["salario"])
        col_saldo = self._find_column(self.df, self.column_map["saldo"])

        df = self.df.copy()
        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo]

        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para a profissão selecionada.")
            return

        # ------------------------
        # Cabeçalho
        # ------------------------
        nome_prof = self.codigos.loc[self.codigos["cbo_codigo"] == cbo_codigo, "cbo_descricao"].values
        nome_prof = nome_prof[0] if len(nome_prof) else cbo_codigo
        
        st.markdown(f"""
        ## 👨‍💼 Profissão: **{nome_prof}**
        Código CBO: `{cbo_codigo}`
        """)

        st.markdown(f"### 🔍 Registros encontrados: **{len(df_cbo):,}**")

        # ====================================================================
        # 👥 Perfil Demográfico
        # ====================================================================

        with st.expander("👥 Perfil demográfico completo"):
            if "idade" in df_cbo.columns:
                idade_media = pd.to_numeric(df_cbo["idade"], errors="coerce").mean()
                st.write(f"Idade média: **{idade_media:.1f} anos**")

            if "sexo" in df_cbo.columns:
                mapa = {"1": "Masculino", "3": "Feminino", "1.0": "Masculino", "3.0": "Feminino"}
                s = df_cbo["sexo"].astype(str).map(mapa).value_counts()
                total = s.sum()
                st.write(f"Homens: **{s.get('Masculino', 0)}** ({s.get('Masculino',0)/total*100:.1f}%)")
                st.write(f"Mulheres: **{s.get('Feminino', 0)}** ({s.get('Feminino',0)/total*100:.1f}%)")

        # ====================================================================
        # 📊 Mercado de Trabalho – Histórico + Previsão
        # ====================================================================

        st.subheader("📊 Situação do Mercado (Saldo de vagas)")

        try:
            df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
            df_cbo[col_saldo] = pd.to_numeric(df_cbo[col_saldo], errors="coerce")
            df_cbo = df_cbo.dropna(subset=[col_data])

            df_cbo["ano"] = df_cbo[col_data].dt.year
            df_cbo = df_cbo[df_cbo["ano"] >= 2020]

            saldo_ano = df_cbo.groupby("ano")[col_saldo].sum().reset_index()

            # Mostrar histórico textual
            for _, linha in saldo_ano.iterrows():
                valor = linha[col_saldo]
                status = "Expansão" if valor > 0 else "Retração" if valor < 0 else "Estável"
                st.write(f"- {linha['ano']}: {valor:+,} ({status})")

            # --------------------------
            # Regressão Linear
            # --------------------------
            X = saldo_ano[["ano"]]
            y = saldo_ano[col_saldo]

            if len(X) > 1:
                modelo = LinearRegression().fit(X, y)

                st.write("### 🔮 Previsões futuras:")

                for anos in anos_futuros:
                    ano_futuro = saldo_ano["ano"].max() + anos
                    pred = int(modelo.predict([[ano_futuro]])[0])

                    st.write(f"- Ano {ano_futuro}: **{pred:+,} vagas**")

                score = r2_score(y, modelo.predict(X))
                st.info(f"Score do modelo (R²): {score:.2f} – {self.interpretacao_score(score)}")

        except Exception as e:
            st.error(f"Erro na previsão de saldo: {e}")

        # ====================================================================
        # 💰 PREVISÕES SALARIAIS
        # ====================================================================

        st.subheader("💰 Previsão Salarial")

        try:
            # Limpeza
            df_cbo[col_sal] = pd.to_numeric(
                df_cbo[col_sal].astype(str).str.replace(",", ".").str.replace(" ", ""), 
                errors="coerce"
            )
            df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
            df_cbo = df_cbo.dropna(subset=[col_sal, col_data])

            df_cbo["ano"] = df_cbo[col_data].dt.year

            salario_atual = df_cbo[col_sal].mean()
            st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")

            # Série temporal mensal
            df_cbo["t"] = (df_cbo["ano"] - 2020) * 12 + df_cbo[col_data].dt.month
            df_mensal = df_cbo.groupby("t")[col_sal].mean().reset_index()

            if len(df_mensal) > 1:
                X = df_mensal[["t"]]
                y = df_mensal[col_sal]

                modelo = LinearRegression().fit(X, y)
                t_final = df_mensal["t"].max()

                st.write("### 🔮 Previsão salarial:")

                for anos in anos_futuros:
                    futuro = t_final + anos * 12
                    pred = modelo.predict([[futuro]])[0]
                    variacao = (pred - salario_atual) / salario_atual * 100

                    st.write(f"- **{2020 + futuro // 12}**: R$ {self.formatar_moeda(pred)} (**{variacao:+.1f}%**)")

                score = r2_score(y, modelo.predict(X))
                st.info(f"Score do modelo (R²): {score:.2f} – {self.interpretacao_score(score)}")

        except Exception as e:
            st.error(f"Erro na previsão salarial: {e}")


# ========================================================================
# STREAMLIT APP
# ========================================================================

st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Plataforma de Previsão do Mercado de Trabalho (CAGED / CBO)")

# Arquivos
parquet_path = "dados.parquet"
cbo_path = "cbo.xlsx"

# Carregar
predictor = MercadoTrabalhoPredictor(parquet_path, cbo_path)
if not predictor.carregar_dados():
    st.stop()

# Busca
entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    resultados = predictor.buscar_profissao(entrada)

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        cbo_selecionado = st.selectbox(
            "Selecione a profissão:",
            resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]
        )

        cbo_codigo = cbo_selecionado.split(" - ")[0]

        if st.button("Gerar análise completa"):
            predictor.relatorio_previsao(cbo_codigo)
