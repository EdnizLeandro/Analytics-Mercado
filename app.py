import streamlit as st
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(
    page_title="Dashboard Profissões - Salários & Tendências",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔎 Consulta de Profissões pelo Nome ou CBO")
st.markdown("""
Pesquise por profissão _digitando o nome completo ou parcial_ (ex: **pintor**, **analista**, **enfermeiro**) e escolha o CBO desejado para visualizar projeções salariais e tendências de mercado.
""")

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("cache_Jobin.csv")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = carregar_dados()

if df is not None:
    # Busca textual pela profissão
    termo = st.text_input(
        "Digite parte do nome da profissão para buscar (exemplo: pintor):",
        placeholder="Exemplo: pintor"
    )

    cbo_selecionado = None
    resultado_filtro = pd.DataFrame()
    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]
        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada para o termo digitado. Tente outro termo.")
        else:
            st.write(f"**Foram encontrados {resultado_filtro.shape[0]} resultados para:** '{termo}'")
            # Mostra tabela resumida de opções
            nomes_cbos = [
                f"{row['codigo']} - {row['descricao']}"
                for _, row in resultado_filtro.iterrows()
            ]
            cbo_str = st.selectbox(
                "Selecione o código CBO e profissão desejada:",
                options=nomes_cbos,
                format_func=lambda x: x
            )
            if cbo_str:
                cbo_selecionado = int(cbo_str.split(' - ')[0])
    elif termo == "":
        st.info("Digite parte do nome da profissão para começar a pesquisa. Exemplo: **pintor**")

    # Se o usuário selecionou algum CBO válido
    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]
        st.subheader(f"Profissão: {info['descricao']} (CBO {info['codigo']})")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Salário Médio Atual",
                value=f"R$ {info['salario_medio_atual']:.2f}",
                help="Salário médio considerado na base mais recente"
            )
            st.metric(
                label="Modelo Vencedor",
                value=f"{info['modelo_vencedor']}",
                help="Modelo estatístico escolhido para previsão"
            )
        with col2:
            st.metric(
                label="Score do Modelo",
                value=f"{info['score']:.4f}",
                help="Score baseado na variância das previsões (quanto mais próximo de 1, mais estável)"
            )
            st.metric(
                label="Tendência Salarial",
                value=f"{info['tendencia_salarial']}",
                help="Projeção para crescimento ou retração do salário"
            )

        # Visualização das previsões salariais
        st.markdown("#### Projeção Salarial (5/10/15/20 anos)")
        anos_futuro = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        salarios_futuro = [
            info['previsao_5'],
            info['previsao_10'],
            info['previsao_15'],
            info['previsao_20']
        ]
        fig = go.Figure(
            go.Scatter(
                x=anos_futuro,
                y=salarios_futuro,
                mode='lines+markers',
                line=dict(color='royalblue'),
                marker=dict(size=10)
            )
        )
        fig.update_layout(
            title=f"Salário Previsto para {info['descricao']}",
            xaxis_title="Horizonte de tempo",
            yaxis_title="Salário (R$)",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"**Tendência de Mercado**: {info['tendencia_mercado']}",
            icon="📊"
        )

        # Detalhes técnicos
        with st.expander("Detalhes Técnicos do Modelo"):
            st.write("Modelo vencedor, score, projeções salariais e interpretação das tendências.")
            st.json({
                "Modelo Vencedor": info['modelo_vencedor'],
                "Score": info['score'],
                "Projeções Salariais": {
                    "+5 anos": info["previsao_5"],
                    "+10 anos": info["previsao_10"],
                    "+15 anos": info["previsao_15"],
                    "+20 anos": info["previsao_20"]
                },
                "Tendência Salarial": info["tendencia_salarial"],
                "Tendência Mercado": info["tendencia_mercado"]
            })
else:
    st.error("Dados não carregados. Verifique o arquivo 'cache_Jobin.csv'.")

# Rodapé
st.markdown(
    "<hr style='margin-top:2em;margin-bottom:1em;'>"
    "<div style='text-align:center; color:grey;'>"
    "© 2025 Jobin Analytics | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
