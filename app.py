import os
import sys
import io
import warnings
import logging
import streamlit as st
import numpy as np
import pandas as pd

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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from contextlib import redirect_stderr
with redirect_stderr(io.StringIO()):
    from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except Exception:
    PMDARIMA_OK = False

def formatar_moeda(valor):
    try:
        return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(valor)

def preparar_lags(df, lag=12):
    df = df.copy()
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['valor'].shift(i)
    df = df.dropna()
    return df

def safe_forecast_list(forecast_list):
    safe = []
    for v in forecast_list:
        try:
            vv = float(v)
            if not np.isfinite(vv):
                vv = 0.0
        except:
            vv = 0.0
        safe.append(vv)
    return safe

class MercadoTrabalhoPredictor:
    def __init__(self, df, df_cbo):
        self.df = df
        self.df_cbo = df_cbo
        # Força nomes certos das colunas pela sua base padrão
        self.coluna_cbo = "cbo2002ocupacao"
        self.coluna_data = "competenciamov"
        self.coluna_salario = "valorsalariofixo"
        self.lstm_model = None
        self.lstm_lag = 12
        self._preparar_dados()

    def _preparar_dados(self):
        self.df[self.coluna_cbo] = self.df[self.coluna_cbo].astype(str)
        self.df_cbo = self.df_cbo.rename(columns={"Código": "cbo_codigo", "Descrição": "cbo_descricao"})
        self.df_cbo['cbo_codigo'] = self.df_cbo['cbo_codigo'].astype(str)

    def buscar_profissao(self, entrada: str):
        entrada = entrada.strip()
        if entrada.isdigit():
            resultados = self.df_cbo[self.df_cbo['cbo_codigo'] == entrada]
        else:
            resultados = self.df_cbo[self.df_cbo['cbo_descricao'].str.contains(entrada, case=False, na=False)]
        return resultados

    def filtrar_registros_dados(self, cbo_codigo):
        self.df[self.coluna_cbo] = self.df[self.coluna_cbo].astype(str)
        return self.df[self.df[self.coluna_cbo] == str(cbo_codigo)].copy()

    def converter_data_robusta(self, df_cbo):
        try:
            df_cbo = df_cbo.copy()
            df_cbo[self.coluna_data] = df_cbo[self.coluna_data].astype(str).str.strip().str.replace(".", "").str.replace(",", "")
            df_cbo = df_cbo[df_cbo[self.coluna_data].str.match(r'^\d{6}$', na=False)]
            if df_cbo.empty:
                return pd.DataFrame()
            df_cbo['ano'] = df_cbo[self.coluna_data].str[:4].astype(int)
            df_cbo['mes'] = df_cbo[self.coluna_data].str[4:].astype(int)
            df_cbo = df_cbo[(df_cbo['ano'] >= 2000) & (df_cbo['ano'] <= 2100)]
            df_cbo = df_cbo[(df_cbo['mes'] >= 1) & (df_cbo['mes'] <= 12)]
            df_cbo['data_convertida'] = pd.to_datetime({'year': df_cbo['ano'],'month': df_cbo['mes'],'day': 1})
            return df_cbo.sort_values('data_convertida')
        except Exception:
            return pd.DataFrame()

    def prever_com_modelos_avancados(self, df_serie, anos_futuros=[5,10,15,20]):
        resultados = {}
        df_serie = df_serie.sort_values('data').reset_index(drop=True)
        datas = df_serie['data']
        X = np.arange(len(df_serie)).reshape(-1,1)
        y = df_serie['valor'].values

        # Linear
        try:
            model_lr = LinearRegression().fit(X, y)
            y_pred = model_lr.predict(X)
            ult_mes = len(df_serie)-1
            previsoes = [model_lr.predict([[ult_mes+anos*12]])[0] for anos in anos_futuros]
            resultados['Linear'] = {'r2': r2_score(y,y_pred), 'mae':mean_absolute_error(y,y_pred),
                                    'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
        except:
            resultados['Linear'] = None
        # ARIMA
        try:
            model_arima = ARIMA(y, order=(1,1,1)).fit()
            y_pred = model_arima.fittedvalues
            previsoes = [model_arima.forecast(steps=anos*12)[-1] for anos in anos_futuros]
            resultados['ARIMA'] = {'r2': r2_score(y[1:],y_pred[1:]) if len(y_pred)>1 else 0,
                                   'mae':mean_absolute_error(y[1:],y_pred[1:]) if len(y_pred)>1 else 0,
                                   'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
        except:
            resultados['ARIMA'] = None
        if PMDARIMA_OK:
            try:
                model_auto = auto_arima(y, seasonal=True, m=12, suppress_warnings=True)
                y_pred = model_auto.predict_in_sample()
                previsoes = [model_auto.predict(anos*12)[-1] for anos in anos_futuros]
                resultados['AutoARIMA'] = {'r2':r2_score(y,y_pred),'mae':mean_absolute_error(y,y_pred),
                                           'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
            except:
                resultados['AutoARIMA'] = None
        try:
            model_sarima = SARIMAX(y, order=(1,1,1), seasonal_order=(1,0,1,12)).fit(disp=False)
            y_pred = model_sarima.fittedvalues
            previsoes=[model_sarima.forecast(steps=anos*12)[-1] for anos in anos_futuros]
            resultados['SARIMA'] = {'r2': r2_score(y[1:],y_pred[1:]) if len(y_pred)>1 else 0,
                                    'mae':mean_absolute_error(y[1:],y_pred[1:]) if len(y_pred)>1 else 0,
                                    'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
        except:
            resultados['SARIMA'] = None
        try:
            model_hw = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12).fit()
            y_pred = model_hw.fittedvalues
            previsoes=[model_hw.forecast(steps=anos*12)[-1] for anos in anos_futuros]
            resultados['Holt-Winters'] = {'r2':r2_score(y,y_pred),'mae':mean_absolute_error(y,y_pred),
                                          'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
        except:
            resultados['Holt-Winters'] = None
        try:
            model_ets = ExponentialSmoothing(y).fit()
            y_pred = model_ets.fittedvalues
            previsoes=[model_ets.forecast(steps=anos*12)[-1] for anos in anos_futuros]
            resultados['ETS'] = {'r2':r2_score(y,y_pred), 'mae':mean_absolute_error(y,y_pred),
                                 'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
        except:
            resultados['ETS'] = None
        try:
            df_prophet = pd.DataFrame({'ds':datas, 'y':y})
            with redirect_stderr(io.StringIO()):
                model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model_prophet.fit(df_prophet)
                y_pred = model_prophet.predict(df_prophet)['yhat'].values
                previsoes=[]
                for anos in anos_futuros:
                    future = model_prophet.make_future_dataframe(periods=anos*12, freq='M')
                    forecast = model_prophet.predict(future)
                    previsoes.append(forecast['yhat'].iloc[-1])
                resultados['Prophet'] = {'r2':r2_score(y,y_pred),'mae':mean_absolute_error(y,y_pred),
                                         'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
        except:
            resultados['Prophet'] = None
        try:
            df_lstm = preparar_lags(df_serie, lag=self.lstm_lag)
            if not df_lstm.empty:
                X_lstm = df_lstm[[f'lag_{i}' for i in range(1,self.lstm_lag+1)]].values
                Y = df_lstm['valor'].values
                X_lstm = X_lstm.reshape((X_lstm.shape[0],X_lstm.shape[1],1))
                if self.lstm_model is None:
                    model = Sequential()
                    model.add(LSTM(50,input_shape=(X_lstm.shape[1],1)))
                    model.add(Dense(1))
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
                    model.fit(X_lstm,Y,epochs=20,batch_size=8,verbose=0)
                    self.lstm_model = model
                y_pred = self.lstm_model.predict(X_lstm,verbose=0).flatten()
                previsoes=[]
                x_next = X_lstm[-1:].copy()
                for anos in anos_futuros:
                    pred = float(self.lstm_model.predict(x_next,verbose=0)[0][0])
                    previsoes.append(pred)
                    x_next = np.roll(x_next,-1)
                    x_next[:,-1,0]=pred
                resultados['LSTM'] = {'r2':r2_score(Y,y_pred),'mae':mean_absolute_error(Y,y_pred),
                                      'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
            else:
                resultados['LSTM'] = None
        except Exception:
            resultados['LSTM'] = None
        try:
            df_xgb = preparar_lags(df_serie, lag=12)
            if not df_xgb.empty:
                X_xgb = df_xgb[[f'lag_{i}' for i in range(1,13)]].values
                Y = df_xgb['valor'].values
                model_xgb = XGBRegressor(n_estimators=100, verbosity=0)
                model_xgb.fit(X_xgb,Y)
                y_pred = model_xgb.predict(X_xgb)
                previsoes=[]
                x_next = X_xgb[-1:].copy()
                for anos in anos_futuros:
                    pred = float(model_xgb.predict(x_next)[0])
                    previsoes.append(pred)
                    x_next = np.roll(x_next,-1)
                    x_next[:,-1]=pred
                resultados['XGBoost'] = {'r2':r2_score(Y,y_pred),'mae':mean_absolute_error(Y,y_pred),
                                         'historico':y_pred,'previsoes':safe_forecast_list(previsoes)}
            else:
                resultados['XGBoost'] = None
        except Exception:
            resultados['XGBoost'] = None
        return resultados

    def prever_mercado_streamlit(self, cbo_codigo, anos_futuros=[5,10,15,20]):
        df_cbo = self.filtrar_registros_dados(cbo_codigo)
        if df_cbo.empty:
            st.error(f"Nenhum registro para o CBO {cbo_codigo} na base de dados! Selecione outro.")
            return
        prof_info = self.df_cbo[self.df_cbo['cbo_codigo'] == cbo_codigo]
        nome_profissao = prof_info.iloc[0]['cbo_descricao'] if not prof_info.empty else f"CBO_{cbo_codigo}"
        st.subheader(f"{nome_profissao} (CBO {cbo_codigo})")
        try:
            df_cbo = self.converter_data_robusta(df_cbo)
        except Exception as e:
            st.error(f"Erro ao processar datas: {e}")
            return
        if df_cbo.empty:
            st.warning("Dados temporais inválidos.")
            return
        try:
            df_mensal = df_cbo.groupby('data_convertida')[self.coluna_salario].mean().reset_index()
            df_mensal.columns=['data','valor']
        except Exception as e:
            st.error(f"Erro ao calcular salários agrupados: {e}")
            st.write(df_cbo.head())
            return
        if df_mensal.empty:
            st.warning("Não há histórico suficiente para projeção salarial.")
            return
        salario_atual = df_mensal['valor'].iloc[-1]
        st.write(f"Salário médio atual: **R$ {formatar_moeda(salario_atual)}**")
        if len(df_mensal)<10:
            st.info("Dados insuficientes para modelos avançados. Mostrando projeção constante.")
            for anos in anos_futuros:
                st.write(f"- {anos} anos → R$ {formatar_moeda(salario_atual)}")
            return
        resultados = self.prever_com_modelos_avancados(df_mensal, anos_futuros)
        melhores = [(m, d) for m, d in resultados.items() if d is not None]
        if melhores:
            melhor = max(melhores, key=lambda x: x[1]['r2'] if np.isfinite(x[1]['r2']) else -np.inf)
            nome_melhor = melhor[0]
            dados_melhor = melhor[1]
            st.success(f"Modelo vencedor: {nome_melhor} (R²={dados_melhor['r2']:.2%}, MAE={dados_melhor['mae']:.2f})")
            st.subheader("Previsão Salarial Futura (melhor modelo)")
            for i, anos in enumerate(anos_futuros):
                st.write(f"- {anos} anos → R$ {formatar_moeda(dados_melhor['previsoes'][i])}")
        else:
            st.warning("Nenhum modelo gerou resultados válidos.")
        # Tendência do saldo
        if 'saldomovimentacao' in df_cbo.columns:
            df_cbo['ano'] = df_cbo['data_convertida'].dt.year
            saldo_ano = df_cbo.groupby('ano')['saldomovimentacao'].sum().reset_index(drop=False).set_index('ano')['saldomovimentacao']
            if saldo_ano.shape[0] >=2:
                media_saldo = float(saldo_ano.tail(12).mean()) if hasattr(saldo_ano,'tail') else float(saldo_ano.mean())
                crescimento_medio = float(saldo_ano.pct_change().mean())
            else:
                media_saldo = float(saldo_ano.sum()) if len(saldo_ano) > 0 else 0.0
                crescimento_medio = 0.0
            if not np.isfinite(crescimento_medio):
                crescimento_medio = 0.0
            if not np.isfinite(media_saldo):
                media_saldo = 0.0
            st.subheader("Tendência de Mercado (Saldo)")
            if media_saldo > 100:
                status_atual = "ALTA DEMANDA"
            elif media_saldo > 0:
                status_atual = "CRESCIMENTO LEVE"
            elif media_saldo > -100:
                status_atual = "RETRAÇÃO LEVE"
            else:
                status_atual = "RETRAÇÃO FORTE"
            st.write(f"Situação histórica recente: **{status_atual}**")
            saldo_projetado = []
            saldo_atual = float(saldo_ano.iloc[-1]) if len(saldo_ano)>0 else 0.0
            for a in anos_futuros:
                try:
                    proj = saldo_atual * ((1.0 + crescimento_medio) ** a)
                    if not np.isfinite(proj) or abs(proj) > 1e12:
                        proj = 0.0
                except Exception:
                    proj = 0.0
                saldo_projetado.append(int(proj))
            st.write("Projeção de saldo de vagas (admissões - desligamentos):")
            for i, anos in enumerate(anos_futuros):
                sinal = "↑" if saldo_projetado[i] > 0 else ("↓" if saldo_projetado[i] < 0 else "→")
                st.write(f"{anos} anos: {saldo_projetado[i]} ({sinal})")
        else:
            st.info("Coluna 'saldomovimentacao' não disponível para tendência do mercado.")

# ---- STREAMLIT APP INICIO ----
st.set_page_config(page_title="Mercado de Trabalho Avançado", layout="wide")
st.title("📊 Análise Avançada do Mercado de Trabalho")

filepath = os.path.join(os.path.dirname(__file__), "dados.parquet")
cbopath = os.path.join(os.path.dirname(__file__), "CBO.xlsx")
df = pd.read_parquet(filepath)
df_cbo = pd.read_excel(cbopath)

app = MercadoTrabalhoPredictor(df, df_cbo)

entrada = st.text_input("Nome ou Código da profissão:")
if entrada:
    resultados = app.buscar_profissao(entrada)
    if resultados.empty:
        st.warning("Nenhum registro encontrado na CBO.")
    else:
        st.write(f"{len(resultados)} profissão(ões) encontrada(s):")
        for idx, row in resultados.iterrows():
            st.write(f"- [{row['cbo_codigo']}] {row['cbo_descricao']}")
        if len(resultados) == 1:
            cbo = resultados["cbo_codigo"].iloc[0]
        else:
            menu = resultados.apply(lambda r: f"{r['cbo_descricao']} [{r['cbo_codigo']}]", axis=1).tolist()
            idx_escolhido = st.selectbox(
                "Escolha a profissão:",
                range(len(menu)),
                format_func=lambda i: menu[i]
            )
            cbo = resultados["cbo_codigo"].iloc[idx_escolhido]
        app.prever_mercado_streamlit(cbo)
