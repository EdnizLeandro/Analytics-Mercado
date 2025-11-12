import os
import warnings
import logging
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from xgboost import XGBRegressor
    from prophet import Prophet
    try:
        from pmdarima import auto_arima
        PMDARIMA_OK = True
    except Exception:
        PMDARIMA_OK = False
except Exception as e:
    st.error(f"Erro ao importar bibliotecas de previsão: {e}")
    st.stop()

def formatar_moeda(valor):
    try:
        return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(valor)

def preparar_lags(df, lag=12):
    df = df.copy()
    for i in range(1, lag + 1):
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

class MercadoTrabalhoStreamlit:
    def __init__(self, df, df_cbo):
        self.df = df
        self.df_cbo = df_cbo
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None
        self.lstm_model = None
        self.lstm_lag = 12
        self._preparar_dados()

    def _preparar_dados(self):
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)
        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocupa' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower or 'data' in col_lower or 'mov' in col_lower:
                self.coluna_data = col
            if 'salario' in col_lower:
                self.coluna_salario = col
        self.cleaned = True

        # Ajusta o DataFrame dos códigos
        self.df_cbo.columns = [col.strip().lower() for col in self.df_cbo.columns]
        # Tenta identificar coluna de código e descrição
        cbo_cod_col = next((c for c in self.df_cbo.columns if 'cbo' in c and ('código' in c or 'codigo' in c or 'cod' in c)), self.df_cbo.columns[0])
        cbo_desc_col = next((c for c in self.df_cbo.columns if 'descri' in c), self.df_cbo.columns[-1])
        self.df_cbo = self.df_cbo.rename(columns={cbo_cod_col: "cbo_codigo", cbo_desc_col: "cbo_descricao"})
        self.df_cbo['cbo_codigo'] = self.df_cbo['cbo_codigo'].astype(str)

    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        # Busca primeiro por código exato na CBO.xlsx
        entrada = entrada.strip()
        if entrada.isdigit():
            resultados = self.df_cbo[self.df_cbo['cbo_codigo'] == entrada]
        else:
            resultados = self.df_cbo[self.df_cbo['cbo_descricao'].str.contains(entrada, case=False, na=False)]
        return resultados

    def filtrar_registros_dados(self, cbo_codigo):
        col = self.coluna_cbo
        self.df[col] = self.df[col].astype(str)
        return self.df[self.df[col] == str(cbo_codigo)]

    def converter_data(self, df_cbo):
        df_cbo = df_cbo.copy()
        col = self.coluna_data
        try:
            if df_cbo[col].dtype != 'datetime64[ns]':
                df_cbo[col] = df_cbo[col].astype(str)
            mask_yyyymm = df_cbo[col].str.match(r'^\d{6}$', na=False)
            if mask_yyyymm.any():
                df_cbo['ano'] = df_cbo.loc[mask_yyyymm, col].str[:4].astype(int)
                df_cbo['mes'] = df_cbo.loc[mask_yyyymm, col].str[4:6].astype(int)
            else:
                df_cbo['ano'] = pd.to_datetime(df_cbo[col], errors='coerce').dt.year
                df_cbo['mes'] = pd.to_datetime(df_cbo[col], errors='coerce').dt.month
            df_cbo['data_convertida'] = pd.to_datetime(dict(year=df_cbo['ano'], month=df_cbo['mes'], day=1))
            return df_cbo.sort_values('data_convertida')
        except Exception:
            return pd.DataFrame()

    def prever_com_modelos(self, df_serie, anos_futuros=[5, 10, 15, 20]):
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
            resultados['Linear'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred),
                                    'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
        except:
            resultados['Linear'] = None

        # ARIMA
        try:
            model_arima = ARIMA(y, order=(1,1,1)).fit()
            y_pred = model_arima.fittedvalues
            previsoes = [model_arima.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
            resultados['ARIMA'] = {'r2': r2_score(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0,
                                   'mae': mean_absolute_error(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0,
                                   'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
        except:
            resultados['ARIMA'] = None

        # AutoARIMA (opcional)
        if 'PMDARIMA_OK' in globals() and PMDARIMA_OK:
            try:
                model_auto = auto_arima(y, seasonal=True, m=12, suppress_warnings=True)
                y_pred = model_auto.predict_in_sample()
                previsoes = [model_auto.predict(anos * 12)[-1] for anos in anos_futuros]
                resultados['AutoARIMA'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred),
                                           'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
            except:
                resultados['AutoARIMA'] = None

        # SARIMA
        try:
            model_sarima = SARIMAX(y, order=(1,1,1), seasonal_order=(1,0,1,12)).fit(disp=False)
            y_pred = model_sarima.fittedvalues
            previsoes = [model_sarima.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
            resultados['SARIMA'] = {'r2': r2_score(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0,
                                    'mae': mean_absolute_error(y[1:], y_pred[1:]) if len(y_pred) > 1 else 0,
                                    'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
        except:
            resultados['SARIMA'] = None

        # Holt-Winters
        try:
            model_hw = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12).fit()
            y_pred = model_hw.fittedvalues
            previsoes = [model_hw.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
            resultados['Holt-Winters'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred),
                                          'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
        except:
            resultados['Holt-Winters'] = None

        # ETS
        try:
            model_ets = ExponentialSmoothing(y).fit()
            y_pred = model_ets.fittedvalues
            previsoes = [model_ets.forecast(steps=anos * 12)[-1] for anos in anos_futuros]
            resultados['ETS'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred),
                                 'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
        except:
            resultados['ETS'] = None

        # Prophet
        try:
            df_prophet = pd.DataFrame({'ds': datas, 'y': y})
            model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model_prophet.fit(df_prophet)
            y_pred = model_prophet.predict(df_prophet)['yhat'].values
            previsoes = []
            for anos in anos_futuros:
                future = model_prophet.make_future_dataframe(periods=anos * 12, freq='M')
                forecast = model_prophet.predict(future)
                previsoes.append(forecast['yhat'].iloc[-1])
            resultados['Prophet'] = {'r2': r2_score(y, y_pred), 'mae': mean_absolute_error(y, y_pred),
                                     'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
        except:
            resultados['Prophet'] = None

        # LSTM
        try:
            df_lstm = preparar_lags(df_serie, lag=self.lstm_lag)
            if not df_lstm.empty:
                X_lstm = df_lstm[[f'lag_{i}' for i in range(1, self.lstm_lag + 1)]].values
                Y = df_lstm['valor'].values
                X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
                if self.lstm_model is None:
                    model = Sequential()
                    model.add(LSTM(50, input_shape=(X_lstm.shape[1], 1)))
                    model.add(Dense(1))
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
                    model.fit(X_lstm, Y, epochs=20, batch_size=8, verbose=0)
                    self.lstm_model = model
                y_pred = self.lstm_model.predict(X_lstm, verbose=0).flatten()
                previsoes = []
                x_next = X_lstm[-1:].copy()
                for anos in anos_futuros:
                    pred = float(self.lstm_model.predict(x_next, verbose=0)[0][0])
                    previsoes.append(pred)
                    x_next = np.roll(x_next, -1)
                    x_next[:, -1, 0] = pred
                resultados['LSTM'] = {'r2': r2_score(Y, y_pred), 'mae': mean_absolute_error(Y, y_pred),
                                      'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
            else:
                resultados['LSTM'] = None
        except Exception:
            resultados['LSTM'] = None

        # XGBoost
        try:
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
                resultados['XGBoost'] = {'r2': r2_score(Y, y_pred), 'mae': mean_absolute_error(Y, y_pred),
                                         'historico': y_pred, 'previsoes': safe_forecast_list(previsoes)}
            else:
                resultados['XGBoost'] = None
        except Exception:
            resultados['XGBoost'] = None
        return resultados

    def prever_mercado(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):
        df_cbo = self.filtrar_registros_dados(cbo_codigo)
        if df_cbo.empty or self.coluna_data not in df_cbo or self.coluna_salario not in df_cbo:
            st.warning("Nenhum dado disponível para análise.")
            return
        nome_profissao = self.df_cbo.query("cbo_codigo == @cbo_codigo")["cbo_descricao"].values
        if len(nome_profissao) > 0:
            nome_profissao = nome_profissao[0]
        else:
            nome_profissao = f"CBO {cbo_codigo}"
        st.subheader(f"{nome_profissao} (CBO {cbo_codigo})")
        plot_area = st.empty()
        df_cbo = self.converter_data(df_cbo)
        if df_cbo.empty or df_cbo['data_convertida'].isnull().all():
            plot_area.warning("Dados de datas inválidos para previsão.")
            return
        df_mensal = df_cbo.groupby('data_convertida')[self.coluna_salario].mean().reset_index()
        df_mensal.columns = ['data', 'valor']
        salario_atual = df_mensal['valor'].iloc[-1]
        st.write(f"Salário médio atual: **R$ {formatar_moeda(salario_atual)}**")
        if len(df_mensal) < 10:
            plot_area.info("Dados insuficientes para modelos avançados. Mostrando projeção constante.")
            for anos in anos_futuros:
                st.write(f"- {anos} anos → R$ {formatar_moeda(salario_atual)}")
        else:
            resultados = self.prever_com_modelos(df_mensal, anos_futuros)
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
                plot_area.warning("Nenhum modelo gerou resultados válidos.")

# ----------- STREAMLIT APP -----------
st.set_page_config(page_title="Mercado de Trabalho Avançado", layout="wide")
st.title("📊 Análise Avançada do Mercado de Trabalho")
st.write("Digite nome ou código da profissão para previsões detalhadas.")

filepath = os.path.join(os.path.dirname(__file__), "dados.parquet")
cbopath = os.path.join(os.path.dirname(__file__), "CBO.xlsx")

df = pd.read_parquet(filepath)
df_cbo = pd.read_excel(cbopath)

app = MercadoTrabalhoStreamlit(df, df_cbo)

entrada = st.text_input("Nome ou Código da profissão:")
if entrada:
    resultados = app.buscar_profissao(entrada)
    if resultados.empty:
        st.warning("Nenhum registro encontrado na CBO.")
    else:
        st.write(f"**{len(resultados)} profissão(ões) encontrada(s):**")
        for index, row in resultados.iterrows():
            st.write(f"- [{row['cbo_codigo']}] {row['cbo_descricao']}")
        # Solicita escolha se múltiplos encontrados
        if len(resultados) == 1:
            cbo = resultados["cbo_codigo"].iloc[0]
        else:
            cbo = st.selectbox("Escolha o código CBO para análise:", resultados["cbo_codigo"])
        app.prever_mercado(cbo)
