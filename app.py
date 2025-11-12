import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class MercadoTrabalhoPredictor:
    def __init__(self, df: pd.DataFrame, df_codigos: pd.DataFrame):
        self.df = df
        self.df_codigos = df_codigos
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None

    def formatar_moeda(self, valor):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def limpar_dados(self):
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)
        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.cleaned = True
        self._identificar_colunas()

    def _identificar_colunas(self):
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocupa' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower and 'mov' in col_lower:
                self.coluna_data = col
            if 'salario' in col_lower and 'fixo' in col_lower:
                self.coluna_salario = col

    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        if not self.cleaned:
            return pd.DataFrame()

        if entrada.isdigit():
            resultados = self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]
            return resultados
        
        mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
        resultados = self.df_codigos[mask]
        return resultados

    def prever_mercado(self, cbo_codigo: str, anos_futuros=[5, 10, 15, 20]):
        if not self.cleaned:
            st.warning("Dataset não limpo.")
            return

        if not all([self.coluna_cbo, self.coluna_data, self.coluna_salario]):
            st.warning("Colunas não identificadas.")
            return

        df_cbo = self.df[self.df[self.coluna_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.warning("Nenhum registro encontrado.")
            return

        st.subheader("ANÁLISE DEMOGRÁFICA")
        if 'idade' in df_cbo.columns:
            st.write(f"Idade média: {df_cbo['idade'].mean():.1f} anos")
        if 'sexo' in df_cbo.columns:
            sexo_dist = df_cbo['sexo'].value_counts(normalize=True) * 100
            st.write("Distribuição por sexo:")
            st.write(sexo_dist)

        if 'graudeinstrucao' in df_cbo.columns:
            escolaridade = df_cbo['graudeinstrucao'].value_counts().head(3)
            st.write("Principais níveis de escolaridade:")
            st.write(escolaridade)

        if 'uf' in df_cbo.columns:
            uf_dist = df_cbo['uf'].value_counts().head(5)
            st.write("Distribuição geográfica:")
            st.write(uf_dist)

        st.subheader("PREVISÃO SALARIAL")
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        df_cbo['tempo_meses'] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                  df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby('tempo_meses')[self.coluna_salario].mean().reset_index()
        salario_atual = df_cbo[self.coluna_salario].mean()
        st.write(f"Salário médio atual: R$ {self.formatar_moeda(salario_atual)}")

        if len(df_mensal) >= 2:
            X = df_mensal[['tempo_meses']]
            y = df_mensal[self.coluna_salario]
            model = LinearRegression()
            model.fit(X, y)
            ult_mes = df_mensal['tempo_meses'].max()
            st.write("Previsões de salário médio:")
            for anos in anos_futuros:
                mes_futuro = ult_mes + anos * 12
                pred = model.predict(np.array([[mes_futuro]]))[0]
                st.write(f"{anos} anos → R$ {self.formatar_moeda(max(pred, 0))}")

# ===== STREAMLIT =====
st.title("Previsão de Mercado de Trabalho")

uploaded_dataset = st.file_uploader("Carregue o arquivo .parquet", type="parquet")
uploaded_codigos = st.file_uploader("Carregue o arquivo de códigos CBO (.xlsx)", type="xlsx")

if uploaded_dataset and uploaded_codigos:
    df = pd.read_parquet(uploaded_dataset)
    df_codigos = pd.read_excel(uploaded_codigos)
    df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
    df_codigos['cbo_codigo'] = df_codigos['cbo_codigo'].astype(str)

    app = MercadoTrabalhoPredictor(df, df_codigos)
    app.limpar_dados()

    entrada = st.text_input("Digite o nome ou código da profissão")
    if entrada:
        resultados = app.buscar_profissao(entrada)
        if resultados.empty:
            st.warning("Nenhuma profissão encontrada.")
        else:
            cbo_selecionado = None
            if len(resultados) == 1:
                cbo_selecionado = resultados['cbo_codigo'].iloc[0]
            else:
                cbo_selecionado = st.selectbox("Selecione o código CBO desejado", resultados['cbo_codigo'])
            
            if cbo_selecionado:
                app.prever_mercado(cbo_selecionado)
