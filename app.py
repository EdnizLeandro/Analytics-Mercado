#!/usr/bin/env python3
"""
Streamlit app: Previsão de Mercado de Trabalho — Versão Final Silenciosa
- Interface interativa para busca por profissão / CBO
- Carregamento de arquivos locais (/mnt/data/dados.parquet, /mnt/data/CBO.xlsx) ou via upload
- Modelos: Linear, ARIMA, (AutoARIMA opcional), SARIMA, Holt-Winters, ETS, Prophet, XGBoost, LSTM
- Logs e warnings suprimidos para execução "silenciosa"
- Reusa modelo LSTM por instância / sessão para evitar tf.function retracing
- Painel (dashboard) com gráficos, tabela de previsões e exportação CSV

Salve como: streamlit_mercado_trabalho_app.py
Execute: streamlit run streamlit_mercado_trabalho_app.py
"""

import os
import io
import sys
import warnings
import logging
from datetime import datetime

# ---------------------------
# Supressão global de logs
# ---------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CMDSTAN_LOG_LEVEL'] = 'ERROR'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('cmdstan').setLevel(logging.ERROR)

try:
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
except Exception:
    pass

# ---------------------------
# Imports principais
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# XGBoost
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

# Prophet (silencioso)
from contextlib import redirect_stderr
with redirect_stderr(io.StringIO()):
    try:
        from prophet import Prophet
    except Exception:
        Prophet = None

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
except Exception:
    tf = None

# pmdarima (opcional)
try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except Exception:
    PMDARIMA_OK = False

# ---------------------------
# Utilitários
# ---------------------------

def formatar_moeda(valor):
    try:
        return f"{float(valor):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return str(valor)


@st.cache_data
def carregar_codigos(path_local='/mnt/data/CBO.xlsx', uploaded_file=None):
    if uploaded_file is not None:
        try:
            df_cod = pd.read_excel(uploaded_file)
        except Exception:
            return pd.DataFrame()
    else:
        try:
            df_cod = pd.read_excel(path_local)
        except Exception:
            return pd.DataFrame()
    df_cod.columns = [str(c).strip() for c in df_cod.columns]
    # tenta normalizar colunas mais comuns
    if df_cod.shape[1] >= 2:
        df_cod = df_cod.iloc[:, :2]
        df_cod.columns = ['cbo_codigo', 'cbo_descricao']
        df_cod['cbo_codigo'] = df_cod['cbo_codigo'].astype(str)
    return df_cod


@st.cache_data
def carregar_dados(path_local='/mnt/data/dados.parquet', uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_parquet(path_local)
    except Exception:
        return pd.DataFrame()
    return df


def identificar_colunas(df):
    coluna_cbo = None
    coluna_data = None
    coluna_salario = None
    for col in df.columns:
        col_lower = str(col).lower().replace(' ', '').replace('_', '')
        if 'cbo' in col_lower and 'ocupa' in col_lower:
            coluna_cbo = col
        if 'competencia' in col_lower and 'mov' in col_lower:
            coluna_data = col
        if 'salario' in col_lower:
            coluna_salario = col
    # heurística alternativa
    if coluna_cbo is None:
        for col in df.columns:
            if 'cbo' in str(col).lower():
                coluna_cbo = col
                break
    if coluna_data is None:
        for col in df.columns:
            if 'data' in str(col).lower() or 'competencia' in str(col).lower():
                coluna_data = col
                break
    if coluna_salario is None:
        for col in df.columns:
            if 'salario' in str(col).lower() or 'remuneracao' in str(col).lower() or 'valor' in str(col).lower():
                coluna_salario = col
                break
    return coluna_cbo, coluna_data, coluna_salario


def converter_data_robusta(df_cbo, coluna_data):
    df_cbo = df_cbo.copy()
    try:
        df_cbo[coluna_data] = df_cbo[coluna_data].astype(str).str.strip().str.replace('.', '').str.replace(',', '')
        df_cbo = df_cbo[df_cbo[coluna_data].str.match(r'^\d{6}$', na=False)]
        if df_cbo.empty:
            return pd.DataFrame()
        df_cbo['ano'] = df_cbo[coluna_data].str[:4].astype(int)
        df_cbo['mes'] = df_cbo[coluna_data].str[4:].astype(int)
        df_cbo = df_cbo[(df_cbo['ano'] >= 1900) & (df_cbo['ano'] <= 2100)]
        df_cbo = df_cbo[(df_cbo['mes'] >= 1) & (df_cbo['mes'] <= 12)]
        df_cbo['data_convertida'] = pd.to_datetime({'year': df_cbo['ano'], 'month': df_cbo['mes'], 'day': 1})
        return df_cbo.sort_values('data_convertida')
    except Exception:
        return pd.DataFrame()


def preparar_lags(df, lag=12):
    df = df.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['valor'].shift(i)
    df = df.dropna()
    return df


def _safe_forecast_list(forecast_list):
    safe = []
    for v in forecast_list:
        try:
            vv = float(v)
            if not np.isfinite(vv):
                vv = 0.0
        except Exception:
            vv = 0.0
        safe.append(vv)
    return safe


@st.cache_resource
def treinar_lstm(X_lstm, Y, epochs=20, batch_size=8):
    if tf is None:
        return None
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_lstm.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
    model.fit(X_lstm, Y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def prever_com_modelos_avancados(df_serie, anos_futuros=[5, 10, 15, 20], lstm_lag=12):
    resultados = {}
    df_serie = df_serie.sort_values('data').reset_index(drop=True)
    datas = df_serie['data']
    X = np.arange(len(df_serie)).reshape(-1, 1)
    y = df_serie['valor'].values

    # Linear
    try:
        model_lr = LinearRegression().fit(X, y)
        y_pred = model_lr.predict(X)
        ult_mes = len(df_serie) - 1
        previsoes = [model_lr.predict([[ult_mes + anos * 12]])[0] for anos in anos_futuros]
        resultados['Linear'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
    except Exception:
        resultados['Linear'] = None

    # ARIMA
    try:
        model_arima = ARIMA(y, order=(1, 1, 1)).fit()
        y_pred = model_arima.fittedvalues
        previsoes = [model_arima.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
        resultados['ARIMA'] = {'r2': r2_score(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0, 'mae': mean_absolute_error(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0, 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
    except Exception:
        resultados['ARIMA'] = None

    # AutoARIMA (opcional)
    if PMDARIMA_OK:
        try:
            model_auto = auto_arima(y, seasonal=True, m=12, suppress_warnings=True)
            y_pred = model_auto.predict_in_sample()
            previsoes = [model_auto.predict(anos * 12)[-1] for anos in anos_futuros]
            resultados['AutoARIMA'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
        except Exception:
            resultados['AutoARIMA'] = None

    # SARIMA
    try:
        model_sarima = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)
        y_pred = model_sarima.fittedvalues
        previsoes = [model_sarima.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
        resultados['SARIMA'] = {'r2': r2_score(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0, 'mae': mean_absolute_error(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0, 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
    except Exception:
        resultados['SARIMA'] = None

    # Holt-Winters
    try:
        model_hw = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12).fit()
        y_pred = model_hw.fittedvalues
        previsoes = [model_hw.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
        resultados['Holt-Winters'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
    except Exception:
        resultados['Holt-Winters'] = None

    # ETS (simple)
    try:
        model_ets = ExponentialSmoothing(y).fit()
        y_pred = model_ets.fittedvalues
        previsoes = [model_ets.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
        resultados['ETS'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
    except Exception:
        resultados['ETS'] = None

    # Prophet
    if Prophet is not None and not df_serie.empty:
        try:
            df_prophet = pd.DataFrame({'ds': datas, 'y': y})
            with redirect_stderr(io.StringIO()):
                model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model_prophet.fit(df_prophet)
                y_pred = model_prophet.predict(df_prophet)['yhat'].values
                previsoes = []
                for anos in anos_futuros:
                    future = model_prophet.make_future_dataframe(periods=anos * 12, freq='M')
                    forecast = model_prophet.predict(future)
                    previsoes.append(forecast['yhat'].iloc[-1])
                resultados['Prophet'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
        except Exception:
            resultados['Prophet'] = None
    else:
        resultados['Prophet'] = None

    # LSTM
    try:
        df_lstm = preparar_lags(df_serie, lag=lstm_lag)
        if not df_lstm.empty and tf is not None:
            X_lstm = df_lstm[[f'lag_{i}' for i in range(1, lstm_lag + 1)]].values
            Y = df_lstm['valor'].values
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

            model = treinar_lstm(X_lstm, Y, epochs=20, batch_size=8)
            if model is not None:
                y_pred = model.predict(X_lstm, verbose=0).flatten()
                previsoes = []
                x_next = X_lstm[-1:].copy()
                for anos in anos_futuros:
                    pred = float(model.predict(x_next, verbose=0)[0][0])
                    previsoes.append(pred)
                    x_next = np.roll(x_next, -1)
                    x_next[:, -1, 0] = pred
                resultados['LSTM'] = {'r2': r2_score(Y, y_pred), 'mae': mean_absolute_error(Y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
            else:
                resultados['LSTM'] = None
        else:
            resultados['LSTM'] = None
    except Exception:
        resultados['LSTM'] = None

    # XGBoost
    try:
        if XGBRegressor is not None:
            df_xgb = preparar_lags(df_serie, lag=12)
            if not df_xgb.empty:
                X_xgb = df_xgb[[f'lag_{i}' for i in range(1, 13)]].values
                Y = df_xgb['valor'].values
                model_xgb = XGBRegressor(n_estimators=100, verbosity=0)
                model_xgb.fit(X_xgb, Y)
                y_pred = model_xgb.predict(X_xgb)
                previsoes = []
                x_next = X_xgb[-1:].copy()
                for anos in anos_futuros:
                    pred = float(model_xgb.predict(x_next)[0])
                    previsoes.append(pred)
                    x_next = np.roll(x_next, -1)
                    x_next[:, -1] = pred
                resultados['XGBoost'] = {'r2': r2_score(Y, y_pred), 'mae': mean_absolute_error(Y, y_pred), 'historico': y_pred, 'previsoes': _safe_forecast_list(previsoes)}
            else:
                resultados['XGBoost'] = None
        else:
            resultados['XGBoost'] = None
    except Exception:
        resultados['XGBoost'] = None

    return resultados


# ---------------------------
# Streamlit UI
# ---------------------------

def app_ui():
    st.set_page_config(page_title='Previsão Mercado de Trabalho', layout='wide')
    st.title('📈 Previsão de Mercado de Trabalho — Versão Final Silenciosa')

    with st.sidebar:
        st.header('Configuração')
        uploaded_data = st.file_uploader('(opcional) Upload: arquivo dados (.parquet)', type=['parquet'])
        uploaded_cbo = st.file_uploader('(opcional) Upload: planilha CBO (.xlsx)', type=['xlsx'])
        anos_futuros = st.multiselect('Horizontes de previsão (anos)', [1, 3, 5, 10, 15, 20], default=[5, 10, 15, 20])
        run_btn = st.button('Executar previsão')
        st.markdown('---')
        st.markdown('Dependências principais: statsmodels, pandas, numpy, scikit-learn, xgboost (opcional), prophet (opcional), tensorflow (opcional), streamlit')

    df = carregar_dados(uploaded_file=uploaded_data)
    df_cod = carregar_codigos(uploaded_file=uploaded_cbo)

    if df.empty:
        st.warning('Nenhum dataset de movimentos encontrado. Faça upload do arquivo `dados.parquet` ou coloque em `/mnt/data/dados.parquet`.')
        st.stop()

    if df_cod.empty:
        st.warning('Nenhuma tabela CBO encontrada. Faça upload de `CBO.xlsx` ou coloque em `/mnt/data/CBO.xlsx`.')

    coluna_cbo, coluna_data, coluna_salario = identificar_colunas(df)
    st.write('**Colunas detectadas:**')
    st.write({'coluna_cbo': coluna_cbo, 'coluna_data': coluna_data, 'coluna_salario': coluna_salario})

    # busca por profissão
    q = st.text_input('Nome ou código da profissão (ex.: 1234 ou "auxiliar")')
    buscar = st.button('Buscar profissão')

    selecionado = None
    if buscar and q:
        if q.isdigit():
            res = df_cod[df_cod['cbo_codigo'] == q] if not df_cod.empty else pd.DataFrame()
            if not res.empty:
                selecionado = res.iloc[0]['cbo_codigo']
                st.success(f"Selecionado: [{selecionado}] {res.iloc[0]['cbo_descricao']}")
            else:
                st.info('Código CBO não encontrado na tabela de códigos.')
        else:
            if not df_cod.empty:
                mask = df_cod['cbo_descricao'].str.contains(q, case=False, na=False)
                candidatos = df_cod[mask]
                if candidatos.shape[0] == 1:
                    selecionado = candidatos['cbo_codigo'].iloc[0]
                    st.success(f"Selecionado: [{selecionado}] {candidatos['cbo_descricao'].iloc[0]}")
                elif candidatos.shape[0] > 1:
                    st.write('Escolha uma das opções:')
                    sel = st.selectbox('Profissões encontradas', candidatos.apply(lambda r: f"[{r.cbo_codigo}] {r.cbo_descricao}", axis=1).tolist())
                    selecionado = sel.split(']')[0].replace('[', '')
                else:
                    st.info('Nenhuma profissão encontrada.')

    # seleção manual
    st.markdown('---')
    st.subheader('Pesquisar manualmente')
    cbo_manual = st.text_input('Ou digite o código CBO diretamente:')
    if cbo_manual:
        selecionado = cbo_manual.strip()

    if run_btn and not selecionado:
        st.warning('Selecione ou digite um código CBO antes de executar.')

    if selecionado and run_btn:
        with st.spinner('Gerando previsões — isso pode demorar alguns segundos...'):
            try:
                df_cbo = df[df[coluna_cbo].astype(str) == str(selecionado)].copy()
            except Exception:
                st.error('Erro ao filtrar dados pelo CBO selecionado.')
                st.stop()

            if df_cbo.empty:
                st.error('Nenhum registro encontrado para o código informado.')
                st.stop()

            df_cbo_conv = converter_data_robusta(df_cbo, coluna_data)
            if df_cbo_conv.empty:
                st.error('Não foi possível converter as datas. Verifique a coluna de competência.')
                st.stop()

            df_mensal = df_cbo_conv.groupby('data_convertida')[coluna_salario].mean()
            df_mensal = df_mensal.asfreq('MS').fillna(method='ffill').reset_index()
            df_mensal.columns = ['data', 'valor']

            st.subheader('Série histórica')
            st.write(df_mensal.tail(10))

            if len(df_mensal) < 10:
                st.info('Série muito curta — exibindo valor atual e mantendo como projeção constante.')
                atual = df_mensal['valor'].iloc[-1]
                st.metric('Salário médio atual (R$)', formatar_moeda(atual))
            else:
                resultados = prever_com_modelos_avancados(df_mensal, anos_futuros=anos_futuros)

                # Apresentar resultados resumidos
                modelos_validos = {k: v for k, v in resultados.items() if v is not None}
                if not modelos_validos:
                    st.error('Nenhum modelo gerou resultados válidos.')
                else:
                    # escolher melhor por R2
                    melhor = max([(k, v) for k, v in modelos_validos.items()], key=lambda x: x[1].get('r2', -np.inf))
                    nome_melhor, dados_melhor = melhor[0], melhor[1]

                    st.success(f"Modelo vencedor: {nome_melhor} — R²={dados_melhor['r2'] if np.isfinite(dados_melhor['r2']) else 0:.4f}")

                    # tabela de previsões
                    previsoes_df = pd.DataFrame({
                        'anos': anos_futuros,
                        'previsao': dados_melhor['previsoes']
                    })
                    previsoes_df['previsao_formatada'] = previsoes_df['previsao'].apply(lambda x: 'R$ ' + formatar_moeda(x))

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader('Gráfico histórico + previsão (melhor modelo)')
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df_mensal['data'], df_mensal['valor'], label='Histórico')
                        # plot histórico do modelo
                        hist = dados_melhor['historico']
                        if len(hist) == len(df_mensal):
                            ax.plot(df_mensal['data'], hist, label=f'{nome_melhor} - fitted')
                        # pontos de previsão
                        ult_data = df_mensal['data'].iloc[-1]
                        pontos = [ult_data + pd.DateOffset(months=anos * 12) for anos in anos_futuros]
                        ax.scatter(pontos, dados_melhor['previsoes'], marker='o', label='Previsões')
                        for i, p in enumerate(pontos):
                            ax.annotate(formatar_moeda(dados_melhor['previsoes'][i]), (p, dados_melhor['previsoes'][i]), textcoords='offset points', xytext=(0, 8), ha='center')
                        ax.set_title(f'Salário médio histórico e previsões — {selecionado}')
                        ax.set_ylabel('R$')
                        ax.legend()
                        st.pyplot(fig)

                    with col2:
                        st.subheader('Previsões (melhor modelo)')
                        st.table(previsoes_df[['anos', 'previsao_formatada']].rename(columns={'anos': 'Horizonte (anos)', 'previsao_formatada': 'Previsão'}))

                    st.markdown('---')
                    st.subheader('Comparativo entre modelos')
                    resumo = []
                    for k, v in resultados.items():
                        if v is None:
                            resumo.append({'modelo': k, 'r2': np.nan, 'mae': np.nan})
                        else:
                            resumo.append({'modelo': k, 'r2': v.get('r2', np.nan), 'mae': v.get('mae', np.nan)})
                    resumo_df = pd.DataFrame(resumo).sort_values(by='r2', ascending=False)
                    st.dataframe(resumo_df)

                    # permitir exportar previsões
                    csv = previsoes_df.to_csv(index=False)
                    st.download_button('Exportar previsões CSV', data=csv, file_name=f'previsoes_{selecionado}.csv')

            # TENDÊNCIA DE MERCADO (saldomovimentacao)
            if 'saldomovimentacao' in df_cbo_conv.columns:
                df_cbo_conv['ano'] = df_cbo_conv['data_convertida'].dt.year
                saldo_ano = df_cbo_conv.groupby('ano')['saldomovimentacao'].sum().reset_index(drop=False).set_index('ano')['saldomovimentacao']
                if saldo_ano.shape[0] >= 2:
                    media_saldo = float(saldo_ano.tail(12).mean()) if hasattr(saldo_ano, 'tail') else float(saldo_ano.mean())
                    crescimento_medio = float(saldo_ano.pct_change().mean())
                else:
                    media_saldo = float(saldo_ano.sum()) if len(saldo_ano) > 0 else 0.0
                    crescimento_medio = 0.0
                if not np.isfinite(crescimento_medio):
                    crescimento_medio = 0.0
                if not np.isfinite(media_saldo):
                    media_saldo = 0.0

                st.markdown('---')
                st.subheader('Tendência de mercado (saldo de movimentação)')
                st.write({'media_saldo': int(media_saldo), 'crescimento_medio': crescimento_medio})

    st.sidebar.markdown('\n\n---\nVersão silenciosa • Gerada por assistant')


if __name__ == '__main__':
    app_ui()
