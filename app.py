import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
        dfs = [pd.read_csv(path, encoding='utf-8', sep=';', on_bad_lines='skip') for path in self.csv_files]
        self.df = pd.concat(dfs, ignore_index=True)
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
        saldo_col = "saldomovimentação"

        prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
        st.markdown(f"### Profissão: <span style='color:#365ebf'><b>{prof_info.iloc[0]['cbo_descricao']}</b></span>" if not prof_info.empty else f"CBO: {cbo_codigo}", unsafe_allow_html=True)
        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para a profissão selecionada.")
            return

        st.write(f"**Registros encontrados:** {len(df_cbo):,}")

        # --- Perfil Demográfico ---
        with st.expander("👥 Perfil Demográfico"):
            left, right = st.columns(2)
            if 'idade' in df_cbo.columns:
                idade_media = pd.to_numeric(df_cbo['idade'], errors='coerce').mean()
                left.metric("Idade média", f"{idade_media:.1f} anos")
            if 'sexo' in df_cbo.columns:
                sexo_dist = df_cbo['sexo'].value_counts()
                sexo_map = {'1.0':'Masculino','3.0':'Feminino','1':'Masculino','3':'Feminino'}
                right.bar_chart(sexo_dist.rename(index=sexo_map))
            if 'graudeinstrucao' in df_cbo.columns:
                escolaridade = df_cbo['graudeinstrucao'].value_counts().head(3)
                escolaridade_map = {
                    '1': 'Analfabeto','2': 'Até 5ª inc. Fundamental','3': '5ª completo Fundamental',
                    '4': '6ª a 9ª Fundamental','5': 'Fundamental completo','6': 'Médio incompleto',
                    '7': 'Médio completo','8': 'Superior incompleto','9': 'Superior completo',
                    '10': 'Mestrado','11': 'Doutorado','80':'Pós-graduação'
                }
                esc_strings = []
                for nivel,count in escolaridade.items():
                    nivel_nome = escolaridade_map.get(str(int(float(nivel))), str(nivel))
                    esc_strings.append(f"{nivel_nome}: {(count/len(df_cbo))*100:.1f}%")
                st.write("**Principais escolaridades:**", ", ".join(esc_strings))
            if 'uf' in df_cbo.columns:
                uf_map = {'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO','21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR','42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'}
                uf_dist = df_cbo['uf'].value_counts().head(5)
                uf_lista = [f"{uf_map.get(str(int(float(uf))),str(uf))}: {count:,} ({(count/len(df_cbo))*100:.1f}%)"
                            for uf,count in uf_dist.items()]
                st.write("**Principais UF:**", ", ".join(uf_lista))

        # --- Mercado de Trabalho Atual ---
        st.subheader("📊 Situação do Mercado de Trabalho")
        if saldo_col in df_cbo.columns:
            saldo_total = pd.to_numeric(df_cbo[saldo_col], errors='coerce').sum()
            if saldo_total > 0: status = "EXPANSÃO (mais admissões que desligamentos)"
            elif saldo_total < 0: status = "RETRAÇÃO (mais desligamentos que admissões)"
            else: status = "MERCADO ESTÁVEL"
            st.markdown(f"**Saldo total de movimentação:** <span style='font-size:20px'>{saldo_total:+,.0f}</span> postos de trabalho  →  <b>{status}</b>", unsafe_allow_html=True)

        # --- PREVISÃO SALARIAL ---
        st.markdown("---")
        st.subheader("💰 Previsão Salarial (5, 10, 15, 20 anos)")
        df_cbo[col_salario] = pd.to_numeric(df_cbo[col_salario].astype(str).str.replace(",",".").str.replace(" ",""), errors="coerce")
        df_cbo = df_cbo.dropna(subset=[col_salario])
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[col_data])
        if df_cbo.empty:
            st.warning("Não há dados temporais válidos.")
            return
        df_cbo['tempo_meses'] = ((df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month)

        df_mensal = df_cbo.groupby('tempo_meses')[col_salario].mean().reset_index()
        salario_atual = df_cbo[col_salario].mean()
        st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")
        col1, col2 = st.columns([2,3])
        if len(df_mensal) >= 2:
            X = df_mensal[['tempo_meses']]
            y = df_mensal[col_salario]
            model = LinearRegression().fit(X, y)
            ult_mes = df_mensal['tempo_meses'].max()
            previsoes = []
            meses_prev = []
            sal_prev = []
            for anos in anos_futuros:
                mes_futuro = ult_mes + anos * 12
                pred = model.predict(np.array([[mes_futuro]]))[0]
                variacao = ((pred-salario_atual)/salario_atual)*100
                previsoes.append((anos, self.formatar_moeda(max(pred,0)), f"{variacao:+.1f}%"))
                meses_prev.append(ult_mes + anos*12)
                sal_prev.append(pred)
            col1.table(pd.DataFrame(previsoes,columns=['Anos','Salário Previsto','Variação (%)']))

            # Gráfico: Salário histórico + previsão
            future_meses = [ult_mes + anos * 12 for anos in anos_futuros]
            future_sal = [model.predict(np.array([[mes]]))[0] for mes in future_meses]
            plt.figure(figsize=(5,3))
            plt.plot(df_mensal['tempo_meses'], df_mensal[col_salario], label="Salário histórico", marker="o")
            plt.plot(future_meses, future_sal, "r--o", label="Previsão", linewidth=2)
            plt.xlabel("Meses desde 2020")
            plt.ylabel("Salário Médio (R$)")
            plt.title("Histórico e Previsão Salarial")
            plt.legend()
            col2.pyplot(plt.gcf())
            plt.close()
        else:
            st.info("Previsão baseada apenas na média atual.")

# --- Streamlit App ---
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED/CBO)")

csv_files = [
    "2020_PE1.csv","2021_PE1.csv","2022_PE1.csv","2023_PE1.csv","2024_PE1.csv","2025_PE1.csv"
]
codigos_filepath = "cbo.xlsx"
with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(csv_files, codigos_filepath)
    app.carregar_dados()

st.success("Dados prontos!")

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
        if st.button("Gerar análise e previsão"):
            app.relatorio_previsao(cbo_codigo, anos_futuros=[5,10,15,20])
