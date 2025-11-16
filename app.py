import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Função para formatar moeda
def formatar_moeda(valor):
    return f"{valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# Simulação de base de dados histórica de salários (exemplo)
data = {
    'Ano': np.arange(2000, 2024),
    'Salario': np.random.uniform(1200, 2500, 24)  # Salários fictícios
}
df = pd.DataFrame(data)

# Função para treinar modelos e escolher o melhor
def treinar_modelos(X, y):
    modelos = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    }
    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        resultados[nome] = {"modelo": modelo, "r2": r2, "mae": mae}
    melhor = max(resultados.items(), key=lambda x: x[1]['r2'])
    return melhor, resultados

# Função para fazer previsões futuras
def previsoes_futuras(modelo, anos_futuros):
    ultimo_ano = df['Ano'].max()
    anos = np.array([ultimo_ano + i for i in anos_futuros]).reshape(-1, 1)
    previsoes = modelo.predict(anos)
    return pd.DataFrame({
        'Ano': anos.flatten(),
        'Salário Previsto (R$)': [formatar_moeda(v) for v in previsoes]
    })

# Simulação de base CBO
cbo_dict = {
    "Pintor de Obras": "514105",
    "Pintor de Automóveis": "514110",
    "Pintor Artístico": "514115"
}

# Streamlit app
st.title("Previsão Salarial e Tendência de Mercado")

placeholder = st.empty()
with placeholder.container():
    profissao_input = st.text_input("Digite o nome ou código da profissão:")

    if profissao_input:
        # Verifica se é genérico e pede seleção do CBO
        opcoes = [nome for nome in cbo_dict.keys() if profissao_input.lower() in nome.lower()]
        
        if len(opcoes) > 1:
            cbo_selecionado = st.selectbox("Selecione o CBO correto:", opcoes)
        elif len(opcoes) == 1:
            cbo_selecionado = opcoes[0]
        else:
            st.warning("Profissão não encontrada. Digite outro nome ou código.")
            st.stop()

        # Dados da profissão selecionada
        profissao = cbo_selecionado
        salario_atual = df['Salario'].iloc[-1]

        st.markdown(f"### Profissão: **{profissao}**")
        st.markdown(f"Salário médio atual: **R$ {formatar_moeda(salario_atual)}**")

        # Treinar modelos
        X = df['Ano'].values.reshape(-1, 1)
        y = df['Salario'].values
        melhor_modelo_nome, resultados_modelos = treinar_modelos(X, y)
        modelo = resultados_modelos[melhor_modelo_nome[0]]['modelo']
        r2 = resultados_modelos[melhor_modelo_nome[0]]['r2']*100
        mae = resultados_modelos[melhor_modelo_nome[0]]['mae']

        st.markdown(f"*Melhor modelo:* **{melhor_modelo_nome[0]}** (R²={r2:.2f}%, MAE={mae:.2f})")

        # Previsões futuras
        anos_futuros = [5, 10, 15, 20]
        df_prev = previsoes_futuras(modelo, anos_futuros)
        st.markdown("### Previsão salarial futura do melhor modelo:")
        st.table(df_prev)

        # Tendência de mercado (simulação)
        st.markdown("======================================================================")
        st.markdown("TENDÊNCIA DE MERCADO (Projeção de demanda para a profissão):")
        st.markdown("======================================================================")

        tendencia = pd.DataFrame({
            "Ano": anos_futuros,
            "Saldo de Vagas": [0, 0, 0, 0],  # Simulação
            "Tendência": ["→", "→", "→", "→"]
        })
        st.markdown("Situação histórica recente: **CRESCIMENTO LEVE**")
        st.table(tendencia)
