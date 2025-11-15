# app.py — Plataforma Jovem Futuro (seleciona e mostra apenas o melhor modelo por RMSE)
import os
import io
import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# suppress warnings in UI
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CMDSTAN_LOG_LEVEL'] = 'ERROR'

# Optional libs — import safely
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except Exception:
    HAS_TF = False

st.set_page_config(page_title="Plataforma Jovem Futuro — Melhor Modelo por RMSE", layout="wide")

# -------------------------
# Helpers
# -------------------------
def format_brl(x):
    try:
        # manual BRL formatting (avoids locale issues on some servers)
        s = f"{x:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    except:
        return str(x)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def safe_read_parquet(path):
    return pd.read_parquet(path)

def find_column(df, candidates):
    for c in df.columns:
        low = c.lower().replace(" ", "").replace("_","")
        for cand in candidates:
            if cand in low:
                return c
    return None

# -------------------------
# Files
# -------------------------
PARQUET = "dados.parquet"
CBO_XLSX = "cbo.xlsx"

st.title("📊 Plataforma Jovem Futuro — Melhor modelo (por RMSE)")

# load files with friendly messages
if not os.path.exists(PARQUET):
    st.error(f"Arquivo não encontrado: {PARQUET}")
    st.stop()
if not os.path.exists(CBO_XLSX):
    st.error(f"Arquivo não encontrado: {CBO_XLSX}")
    st.stop()

with st.spinner("Carregando dados..."):
    df = safe_read_parquet(PARQUET)
    df_cbo = pd.read_excel(CBO_XLSX)

# normalize cbo sheet
df_cbo.columns = [str(c).strip() for c in df_cbo.columns]
# try to find 'codigo' and 'descricao' columns
col_code = next((c for c in df_cbo.columns if 'cod' in c.lower()), df_cbo.columns[0])
col_desc = next((c for c in df_cbo.columns if 'descr' in c.lower() or 'nome' in c.lower() or 'titulo' in c.lower()), df_cbo.columns[1] if len(df_cbo.columns)>1 else df_cbo.columns[0])
df_cbo = df_cbo.rename(columns={col_code:'codigo', col_desc:'descricao'})
df_cbo['codigo'] = df_cbo['codigo'].astype(str)

st.success("Dados carregados.")
st.write("Colunas do dataset principal:", list(df.columns))

# -------------------------
# detect expected columns robustly
# -------------------------
col_cbo = find_column(df, ['cbo','ocupacao','ocupação'])
col_date = find_column(df, ['competencia', 'competenciamov', 'data', 'competenciadec'])
col_salary = find_column(df, ['salario','remuneracao','valorsalario'])
col_saldo = find_column(df, ['saldomovimentacao','saldomovimentação','saldo'])

if not all([col_cbo, col_date, col_salary, col_saldo]):
    st.error("Não localizei todas as colunas necessárias automaticamente. Verifique os nomes: precisa conter algo como 'cbo...', 'competencia...', 'salario...', 'saldomovimentacao...'.")
    st.stop()

# -------------------------
# Fix/parse dates robustly (avoid 1970)
# -------------------------
def parse_competencia(series):
    s = series.astype(str).str.strip().str.replace(r'\D','', regex=True)
    # detect YYYYMM or YYYY-MM or YYYYMMDD
    def parse_val(v):
        if pd.isna(v) or v=='':
            return pd.NaT
        if len(v)==6:
            return pd.to_datetime(v, format='%Y%m', errors='coerce')
        if len(v)==8:
            return pd.to_datetime(v, format='%Y%m%d', errors='coerce')
        if len(v)==7 and '-' in v:
            try:
                return pd.to_datetime(v, format='%Y-%m', errors='coerce')
            except:
                return pd.to_datetime(v, errors='coerce')
        try:
            return pd.to_datetime(v, errors='coerce')
        except:
            return pd.NaT
    return s.apply(parse_val)

df[col_date] = parse_competencia(df[col_date])
if df[col_date].isna().sum()>0:
    st.warning(f"{df[col_date].isna().sum():,} linhas com datas inválidas foram convertidas para NaT e serão ignoradas nas séries temporais.")

# -------------------------
# Fix Brazilian financial format for salary
# -------------------------
def fix_salary_col(s):
    s = s.astype(str).str.strip()
    # remove thousands sep '.' and convert comma decimal to dot
    s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    s = pd.to_numeric(s, errors='coerce')
    return s

df[col_salary] = fix_salary_col(df[col_salary])
# sanitize extremes
df.loc[df[col_salary] < 0, col_salary] = np.nan
# replace absurd highs ( > 1e6 ) with NaN
df.loc[df[col_salary] > 1_000_000, col_salary] = np.nan
# fill with median
median_salary = df[col_salary].median()
df[col_salary] = df[col_salary].fillna(median_salary)

# -------------------------
# UI: search profession
# -------------------------
st.header("🔎 Buscar profissão (nome ou código CBO)")
q = st.text_input("Digite nome ou código da profissão:")
if not q:
    st.info("Digite um termo para buscar profissões (ex: 'pintor' ou código '716610').")
    st.stop()

mask = df_cbo['descricao'].astype(str).str.contains(q, case=False, na=False) | df_cbo['codigo'].astype(str).str.contains(q, na=False)
candidates = df_cbo[mask]
if candidates.empty:
    st.warning("Nenhuma profissão encontrada.")
    st.stop()

st.write("Profissões encontradas:")
st.dataframe(candidates[['codigo','descricao']].head(50))

selected = st.selectbox("Selecione o código CBO:", candidates['codigo'].astype(str).unique())
st.write(f"Selecionado: {selected} — {df_cbo.loc[df_cbo['codigo']==selected,'descricao'].values[0]}")

# subset job data
df_job = df[df[col_cbo].astype(str)==str(selected)].copy()
df_job = df_job.dropna(subset=[col_date])
if df_job.empty:
    st.warning("Sem registros temporais para a profissão selecionada.")
    st.stop()

# aggregate monthly for target series (use saldo column for demand series; salary series separately)
ts_demand = df_job.set_index(col_date).resample('M')[col_saldo].mean().ffill().reset_index().rename(columns={col_date:'ds', col_saldo:'y'})
ts_salary = df_job.set_index(col_date).resample('M')[col_salary].mean().ffill().reset_index().rename(columns={col_date:'ds', col_salary:'y'})

st.subheader("Série temporal (amostra de demanda - média mensal)")
st.write(f"Período: {ts_demand['ds'].min().date()} → {ts_demand['ds'].max().date()}")
fig = px.line(ts_demand, x='ds', y='y', title='Saldo médio mensal (demanda)')
st.plotly_chart(fig, use_container_width=True)

# require minimum length
MIN_LEN = 12
if len(ts_demand) < MIN_LEN:
    st.warning(f"Série muito curta para treinar vários modelos (mínimo recomendado = {MIN_LEN}). Será usado fallback com média/linear.")
# select forecast horizon in months (default 12)
h_months = st.selectbox("Horizonte de previsão (meses):", [6, 12, 24], index=1)

# -------------------------
# Prepare train/test split (last N months as test)
# -------------------------
test_months = min(6, max(1, int(h_months//6)))  # small test size heuristic
train = ts_demand[:-test_months] if len(ts_demand)>test_months else ts_demand
test = ts_demand[-test_months:] if len(ts_demand)>=test_months else ts_demand.copy()

y_train = train['y'].values
y_test = test['y'].values
ds_train = train['ds']
ds_test = test['ds']

results = {}  # store metrics and forecasts

# -------------------------
# 1) Linear regression on time index
# -------------------------
try:
    X_train = np.arange(len(train)).reshape(-1,1)
    X_test = np.arange(len(train), len(train)+len(test)).reshape(-1,1)
    lr = LinearRegression().fit(X_train, y_train)
    pred_train = lr.predict(X_train)
    pred_test = lr.predict(X_test)
    results['Linear'] = {
        'pred_test': pred_test,
        'pred_full_future': lr.predict(np.arange(len(ts_demand), len(ts_demand)+h_months).reshape(-1,1)),
        'rmse': rmse(y_test, pred_test) if len(y_test)>0 else float('inf'),
        'mae': mean_absolute_error(y_test, pred_test) if len(y_test)>0 else float('inf'),
        'r2': r2_score(y_test, pred_test) if len(y_test)>0 else float('-inf')
    }
except Exception as e:
    results['Linear'] = None

# -------------------------
# 2) Prophet
# -------------------------
if HAS_PROPHET:
    try:
        dfp = ts_demand.rename(columns={'ds':'ds','y':'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp.iloc[:len(train)])
        future = m.make_future_dataframe(periods=len(test), freq='M')
        fc = m.predict(future)
        pred_test = fc['yhat'].iloc[-len(test):].values if len(test)>0 else np.array([])
        future_full = m.make_future_dataframe(periods=h_months, freq='M')
        fc_full = m.predict(future_full)
        results['Prophet'] = {
            'pred_test': pred_test,
            'pred_full_future': fc_full['yhat'].iloc[-h_months:].values,
            'rmse': rmse(y_test, pred_test) if len(y_test)>0 else float('inf'),
            'mae': mean_absolute_error(y_test, pred_test) if len(y_test)>0 else float('inf'),
            'r2': r2_score(y_test, pred_test) if len(y_test)>0 else float('-inf'),
            'model': m,
            'forecast_full': fc_full
        }
    except Exception:
        results['Prophet'] = None
else:
    results['Prophet'] = None

# -------------------------
# 3) ARIMA / SARIMA (statsmodels) — if available
# -------------------------
if HAS_STATSMODELS:
    try:
        y = ts_demand['y'].values
        # simple ARIMA(1,1,1) on train
        arima_model = ARIMA(y[:len(train)], order=(1,1,1)).fit()
        pred_test = arima_model.forecast(steps=len(test)) if len(test)>0 else np.array([])
        pred_full = arima_model.forecast(steps=h_months)
        results['ARIMA'] = {
            'pred_test': np.array(pred_test),
            'pred_full_future': np.array(pred_full),
            'rmse': rmse(y_test, pred_test) if len(y_test)>0 else float('inf'),
            'mae': mean_absolute_error(y_test, pred_test) if len(y_test)>0 else float('inf'),
            'r2': r2_score(y_test, pred_test) if len(y_test)>0 else float('-inf'),
            'model': arima_model
        }
    except Exception:
        results['ARIMA'] = None
else:
    results['ARIMA'] = None

# -------------------------
# 4) XGBoost (lag features)
# -------------------------
def create_lag_df(series, lags=12):
    df_l = pd.DataFrame({'y':series})
    for i in range(1,lags+1):
        df_l[f'lag_{i}'] = df_l['y'].shift(i)
    return df_l.dropna()

if HAS_XGBOOST:
    try:
        lags = min(12, max(3, len(ts_demand)//6))
        df_lag = create_lag_df(ts_demand['y'].values, lags=lags)
        X = df_lag.drop(columns='y').values
        y = df_lag['y'].values
        split = int(0.8*len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        xgb = XGBRegressor(n_estimators=200, verbosity=0)
        xgb.fit(X_tr, y_tr)
        # build X_test from the last windows
        # For test: create rolling windows aligned to test length
        preds_test = []
        last_window = ts_demand['y'].values[-lags:].tolist()
        for _ in range(len(test)):
            arr = np.array(last_window[-lags:]).reshape(1,-1)
            p = xgb.predict(arr)[0]
            preds_test.append(p)
            last_window.append(p)
        # forecast full future
        last_window2 = ts_demand['y'].values[-lags:].tolist()
        preds_full = []
        for _ in range(h_months):
            arr = np.array(last_window2[-lags:]).reshape(1,-1)
            p = xgb.predict(arr)[0]
            preds_full.append(p)
            last_window2.append(p)
        results['XGBoost'] = {
            'pred_test': np.array(preds_test),
            'pred_full_future': np.array(preds_full),
            'rmse': rmse(y_test, np.array(preds_test)) if len(y_test)>0 else float('inf'),
            'mae': mean_absolute_error(y_test, np.array(preds_test)) if len(y_test)>0 else float('inf'),
            'r2': r2_score(y_test, np.array(preds_test)) if len(y_test)>0 else float('-inf'),
            'model': xgb
        }
    except Exception:
        results['XGBoost'] = None
else:
    results['XGBoost'] = None

# -------------------------
# 5) LSTM (if TF available) — small model, reuse not retraced in loop
# -------------------------
if HAS_TF:
    try:
        # prepare with small window
        window = min(6, max(3, len(ts_demand)//12))
        arr = ts_demand['y'].values
        Xs, ys = [], []
        for i in range(window, len(arr)):
            Xs.append(arr[i-window:i])
            ys.append(arr[i])
        Xs = np.array(Xs); ys = np.array(ys)
        if len(Xs) > 10:
            # scale
            minv, maxv = Xs.min(), Xs.max()
            scale = maxv - minv if maxv!=minv else 1.0
            Xs_s = (Xs - minv)/scale
            ys_s = (ys - minv)/scale
            # reshape for LSTM
            Xs_s = Xs_s.reshape((Xs_s.shape[0], Xs_s.shape[1], 1))
            # build small model
            tf.keras.backend.clear_session()
            model_l = Sequential()
            model_l.add(LSTM(32, input_shape=(Xs_s.shape[1],1)))
            model_l.add(Dropout(0.2))
            model_l.add(Dense(1))
            model_l.compile(optimizer='adam', loss='mse')
            model_l.fit(Xs_s, ys_s, epochs=15, batch_size=8, verbose=0)
            # predict test
            preds_test = []
            last = arr[-(window+len(test)):-len(test)] if len(test)>0 else arr[-window:]
            # if not enough for rolling, fallback
            last_window = arr[-window:].tolist()
            # produce test preds by iteratively predicting
            for _ in range(len(test)):
                arr_in = (np.array(last_window[-window:]) - minv)/scale
                p = model_l.predict(arr_in.reshape(1,window,1), verbose=0)[0][0]
                p = p*scale + minv
                preds_test.append(p)
                last_window.append(p)
            # full future
            last_window2 = arr[-window:].tolist()
            preds_full = []
            for _ in range(h_months):
                arr_in = (np.array(last_window2[-window:]) - minv)/scale
                p = model_l.predict(arr_in.reshape(1,window,1), verbose=0)[0][0]
                p = p*scale + minv
                preds_full.append(p)
                last_window2.append(p)
            results['LSTM'] = {
                'pred_test': np.array(preds_test),
                'pred_full_future': np.array(preds_full),
                'rmse': rmse(y_test, np.array(preds_test)) if len(y_test)>0 else float('inf'),
                'mae': mean_absolute_error(y_test, np.array(preds_test)) if len(y_test)>0 else float('inf'),
                'r2': r2_score(y_test, np.array(preds_test)) if len(y_test)>0 else float('-inf'),
                'model': model_l
            }
        else:
            results['LSTM'] = None
    except Exception:
        results['LSTM'] = None
else:
    results['LSTM'] = None

# -------------------------
# Evaluate and pick best by RMSE
# -------------------------
evaluated = {k:v for k,v in results.items() if v is not None}
if not evaluated:
    st.error("Nenhum modelo foi executado com sucesso. Verifique dependências (Prophet, statsmodels, xgboost, tensorflow).")
    st.stop()

metrics_df = []
for name, r in evaluated.items():
    metrics_df.append({
        'model': name,
        'rmse': r['rmse'],
        'mae': r['mae'],
        'r2': r['r2']
    })
metrics_df = pd.DataFrame(metrics_df).sort_values('rmse')
best_model_name = metrics_df.iloc[0]['model']
best = evaluated[best_model_name]

st.subheader("🏆 Seleção automática do melhor modelo (critério = RMSE)")
st.table(metrics_df.style.format({"rmse":"{:.2f}", "mae":"{:.2f}", "r2":"{:.3f}"}))

st.markdown(f"**Melhor modelo:** `{best_model_name}` — **RMSE = {best['rmse']:.2f}**, MAE = {best['mae']:.2f}, R² = {best['r2']:.3f}")

# -------------------------
# Show only best forecast (test vs predicted + future)
# -------------------------
# assemble test plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=ds_train, y=train['y'], mode='lines', name='Histórico (train)'))
if len(test)>0:
    fig.add_trace(go.Scatter(x=ds_test, y=test['y'], mode='markers+lines', name='Real (test)'))
# predicted test
if 'pred_test' in best and len(best['pred_test'])>0:
    fig.add_trace(go.Scatter(x=ds_test, y=best['pred_test'], mode='lines+markers', name=f'Predito ({best_model_name})'))
# future forecast dates
last_date = ts_demand['ds'].max()
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h_months, freq='M')
fig.add_trace(go.Scatter(x=future_dates, y=best['pred_full_future'], mode='lines+markers', name=f'Forecast {h_months}m ({best_model_name})', line=dict(dash='dash')))
fig.update_layout(title=f"Melhor modelo: {best_model_name} — Previsão ({h_months} meses)", yaxis_title='Saldo médio', xaxis_title='Data')
st.plotly_chart(fig, use_container_width=True)

# Show numeric future forecast formatted in BRL if salary; here it's demand (not monetary).
# But user asked salary formatting for finances — we will show salary forecast too if user asks.
st.subheader("🔢 Valores previstos (melhor modelo)")
df_forecast = pd.DataFrame({'date': future_dates, 'forecast': best['pred_full_future']})
# format forecast numbers; these are demand numbers (not monetary); if user wants salary forecasting, script can be run for salary series similarly
st.dataframe(df_forecast.assign(forecast=lambda d: d['forecast'].round(2)))

# Download forecast CSV
csv = df_forecast.to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar forecast (CSV)", data=csv, file_name=f"forecast_{selected}_{best_model_name}.csv", mime="text/csv")

st.success("✅ Previsão mostrada para o melhor modelo (por RMSE).")

# Optional: also run salary forecast using the same winner model (if user wants)
if st.checkbox("Também gerar previsão salarial média (meses) para esta profissão"):
    # repeat pipeline on ts_salary using only the best model type
    st.info("Gerando previsão salarial usando o mesmo critério (meses selecionados).")
    ts = ts_salary.copy()
    if len(ts) < MIN_LEN:
        st.warning("Série salarial curta; saída limitada.")
    # iterate depending on best_model_name
    try:
        if best_model_name == 'Prophet' and HAS_PROPHET:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(ts[['ds','y']])
            future = m.make_future_dataframe(periods=h_months, freq='M')
            fc = m.predict(future)
            vals = fc['yhat'].iloc[-h_months:].values
        elif best_model_name == 'XGBoost' and HAS_XGBOOST:
            # build lag features
            lags = min(12, max(3, len(ts)//6))
            arr = ts['y'].values
            last_window = arr[-lags:].tolist()
            preds = []
            from xgboost import XGBRegressor
            # train simple XGB on full ts
            df_lag = create_lag_df(arr, lags=lags)
            X = df_lag.drop(columns='y').values; y = df_lag['y'].values
            if len(X)>0:
                xgb = XGBRegressor(n_estimators=100, verbosity=0)
                xgb.fit(X,y)
                for _ in range(h_months):
                    p = xgb.predict(np.array(last_window[-lags:]).reshape(1,-1))[0]
                    preds.append(p)
                    last_window.append(p)
            vals = np.array(preds)
        else:
            # fallback: linear trend
            X = np.arange(len(ts)).reshape(-1,1)
            y = ts['y'].values
            lr = LinearRegression().fit(X,y)
            vals = lr.predict(np.arange(len(ts), len(ts)+h_months).reshape(-1,1))
        # present salary forecast formatted as BRL
        df_sal_fore = pd.DataFrame({'date': pd.date_range(ts['ds'].max()+pd.offsets.MonthBegin(1), periods=h_months, freq='M'), 'predicted_salary': vals})
        df_sal_fore['predicted_salary_brl'] = df_sal_fore['predicted_salary'].apply(format_brl)
        st.dataframe(df_sal_fore[['date','predicted_salary_brl']])
    except Exception as e:
        st.error(f"Erro gerando previsão salarial: {e}")
