import os
import sys
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Optional imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ---------------- CONFIG ----------------
DATA_FILENAME = 'PLTR_2020-09-30_2025-09-09.csv'  # filename only; put file under data/
DATA_DIR = 'data'
DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)
# If your CSV has that extra second row like in screenshot, keep skiprows=[1]
CSV_SKIPROWS = [1]  # set to [] or None if not needed
OUTPUT_DIR = 'outputs_palantir'
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Helpers ----------------

def normalize_columns(cols):
    """Return mapping old->new and list of normalized column names."""
    mapping = {}
    normalized = []
    for c in cols:
        c0 = str(c).strip()
        c_norm = c0.lower().replace(' ', '_')
        mapping[c0] = c_norm
        normalized.append(c_norm)
    return mapping, normalized

def load_data(path=DATA_PATH, skiprows=CSV_SKIPROWS):
    if not os.path.exists(path):
        print(f"ERROR: data file not found at {path}. Please download from Kaggle and place it in the {DATA_DIR}/ folder.")
        sys.exit(1)

    # read CSV with optional skiprows
    read_kwargs = {}
    if skiprows:
        read_kwargs['skiprows'] = skiprows

    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception as e:
        print("Error reading CSV:", e)
        sys.exit(1)

    # Normalize column names to simple snake_case keys
    original_cols = list(df.columns)
    mapping, normalized = normalize_columns(original_cols)
    df.columns = normalized

    # detect date column among normalized names
    date_candidates = [c for c in df.columns if 'date' in c]
    if date_candidates:
        date_col = date_candidates[0]
    else:
        # fallback to first column
        date_col = df.columns[0]

    # rename date column to 'date' for consistency
    if date_col != 'date':
        df.rename(columns={date_col: 'date'}, inplace=True)

    # ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().any():
        print("Warning: some dates could not be parsed and will be NaT.")

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df

def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {path}")

# ---------------- Feature Engineering ----------------

def detect_price_column(df):
    # Try common variants in normalized column names
    for name in ['adj_close', 'adj_close', 'adjclose', 'close', 'close_price', 'close_adj']:
        if name in df.columns:
            return name
    # fallback to any numeric column that looks like price (heuristic)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        return numeric[0]
    return None

def add_technical_indicators(df):
    df = df.copy()
    price_col = detect_price_column(df)
    if price_col is None:
        raise ValueError('No Close/Adj Close column found in dataset (checked normalized names).')

    # Basic returns
    df['return'] = df[price_col].pct_change()
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'ma_{w}'] = df[price_col].rolling(window=w, min_periods=1).mean()
        df[f'ma_{w}_diff'] = df[price_col] - df[f'ma_{w}']

    # Exponential moving averages
    for w in [12, 26]:
        df[f'ema_{w}'] = df[price_col].ewm(span=w, adjust=False).mean()

    # MACD
    df['macd'] = df.get('ema_12', df[price_col].ewm(span=12, adjust=False).mean()) - df.get('ema_26', df[price_col].ewm(span=26, adjust=False).mean())
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    def compute_rsi(series, window=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=window, min_periods=1).mean()
        ma_down = down.rolling(window=window, min_periods=1).mean()
        rs = ma_up / (ma_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi_14'] = compute_rsi(df[price_col], 14)

    # Bollinger Bands
    df['bb_mid_20'] = df[price_col].rolling(window=20, min_periods=1).mean()
    df['bb_std_20'] = df[price_col].rolling(window=20, min_periods=1).std()
    df['bb_upper_20'] = df['bb_mid_20'] + 2 * df['bb_std_20']
    df['bb_lower_20'] = df['bb_mid_20'] - 2 * df['bb_std_20']

    # Volatility (rolling std of log returns)
    df['vol_20'] = df['log_return'].rolling(window=20, min_periods=1).std()

    return df

# ---------------- EDA ----------------

def eda_plots(df):
    price_col = detect_price_column(df)
    if price_col is None:
        print("EDA skipped: no price column detected.")
        return

    # Price plot
    fig = plt.figure()
    plt.plot(df['date'], df[price_col])
    plt.title('Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    save_plot(fig, 'price_history.png')

    # Returns distribution
    if 'log_return' in df.columns:
        fig = plt.figure()
        sns.histplot(df['log_return'].dropna(), bins=100)
        plt.title('Log Return Distribution')
        save_plot(fig, 'log_return_dist.png')

    # Rolling volatility
    if 'vol_20' in df.columns:
        fig = plt.figure()
        plt.plot(df['date'], df['vol_20'])
        plt.title('20-day Rolling Volatility (log returns)')
        save_plot(fig, 'volatility_20.png')

# ---------------- Forecasting: ARIMA (if available) ----------------

def adf_test(series):
    if not STATSMODELS_AVAILABLE:
        print('statsmodels not available; skipping ADF test')
        return None
    series = series.dropna()
    result = adfuller(series)
    out = {
        'adf_stat': result[0],
        'pvalue': result[1],
        'usedlag': result[2],
        'nobs': result[3]
    }
    return out

def fit_arima_forecast(df, target_col=None, forecast_horizon=30):
    if not STATSMODELS_AVAILABLE:
        print('statsmodels not available; skipping ARIMA')
        return None
    if target_col is None:
        target_col = detect_price_column(df)
    if target_col is None:
        print("ARIMA skipped: no price column detected.")
        return None

    series = df.set_index('date')[target_col].asfreq('D').interpolate()
    order = (5, 1, 0)
    print('Fitting ARIMA order', order)
    model = ARIMA(series, order=order)
    res = model.fit()
    pred = res.get_forecast(steps=forecast_horizon)
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()

    fig = plt.figure()
    plt.plot(series.index[-500:], series.values[-500:], label='history')
    plt.plot(forecast.index, forecast.values, label='forecast')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.2)
    plt.legend()
    plt.title('ARIMA Forecast')
    save_plot(fig, 'arima_forecast.png')
    return res

# ---------------- Forecasting: LSTM ----------------

def create_lstm_dataset(series, look_back=30):
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def fit_lstm(df, target_col=None, look_back=30, epochs=30, batch_size=32):
    if not TENSORFLOW_AVAILABLE:
        print('TensorFlow not available; skipping LSTM')
        return None
    if target_col is None:
        target_col = detect_price_column(df)
    if target_col is None:
        print("LSTM skipped: no price column detected.")
        return None

    series = df.set_index('date')[target_col].asfreq('D').interpolate().values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X, y = create_lstm_dataset(series_scaled, look_back=look_back)
    if len(X) == 0:
        print("Not enough data for LSTM with given look_back.")
        return None
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    pred_scaled = model.predict(X_test).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    fig = plt.figure()
    plt.plot(y_test_real, label='actual')
    plt.plot(pred_real, label='pred')
    plt.legend()
    plt.title('LSTM Test Predictions')
    save_plot(fig, 'lstm_predictions.png')
    return model

# ---------------- Backtesting ----------------

def backtest_ma_crossover(df, short_window=20, long_window=50, initial_capital=100000):
    price_col = detect_price_column(df)
    if price_col is None:
        print("MA Backtest skipped: no price column.")
        return pd.DataFrame()

    data = df[['date', price_col]].copy().rename(columns={'date': 'Date'})
    data.set_index('Date', inplace=True)
    data['ma_short'] = data[price_col].rolling(window=short_window, min_periods=1).mean()
    data['ma_long'] = data[price_col].rolling(window=long_window, min_periods=1).mean()
    data['signal'] = 0
    data.loc[data.index[short_window:], 'signal'] = np.where(data['ma_short'][short_window:] > data['ma_long'][short_window:], 1, 0)
    data['positions'] = data['signal'].diff()

    cash = initial_capital
    shares = 0
    portfolio = []

    for idx, row in data.iterrows():
        price = row[price_col]
        pos = row['positions']
        if pos == 1:
            # buy with all cash (integer shares)
            shares = int(cash // price) if price > 0 else 0
            cost = shares * price
            cash -= cost
        elif pos == -1 and shares > 0:
            cash += shares * price
            shares = 0
        total_value = cash + shares * price
        portfolio.append({'date': idx, 'total': total_value, 'cash': cash, 'shares': shares, 'price': price})

    port_df = pd.DataFrame(portfolio).set_index('date')
    if not port_df.empty:
        port_df['returns'] = port_df['total'].pct_change().fillna(0)

    # plots
    fig = plt.figure()
    plt.plot(data.index, data[price_col], label='price')
    buy_signals = data[data['positions'] == 1].index
    sell_signals = data[data['positions'] == -1].index
    plt.scatter(buy_signals, data.loc[buy_signals, price_col], marker='^', color='g', label='Buy')
    plt.scatter(sell_signals, data.loc[sell_signals, price_col], marker='v', color='r', label='Sell')
    plt.legend()
    plt.title(f'MA Crossover Strategy ({short_window}/{long_window})')
    save_plot(fig, f'ma_crossover_{short_window}_{long_window}.png')

    return port_df

def backtest_rsi_strategy(df, rsi_col='rsi_14', buy_thresh=30, sell_thresh=70, initial_capital=100000):
    price_col = detect_price_column(df)
    if price_col is None or rsi_col not in df.columns:
        print("RSI Backtest skipped: missing price or RSI column.")
        return pd.DataFrame()

    data = df[['date', price_col, rsi_col]].copy().rename(columns={'date': 'Date'})
    data.set_index('Date', inplace=True)
    data['signal'] = np.where(data[rsi_col] < buy_thresh, 1, np.where(data[rsi_col] > sell_thresh, -1, 0))
    data['positions'] = data['signal'].diff()

    cash = initial_capital
    shares = 0
    portfolio = []
    for idx, row in data.iterrows():
        price = row[price_col]
        pos = row['positions']
        if pos == 1 and shares == 0:
            shares = int(cash // price) if price > 0 else 0
            cash -= shares * price
        elif pos == -1 and shares > 0:
            cash += shares * price
            shares = 0
        total = cash + shares * price
        portfolio.append({'date': idx, 'total': total, 'cash': cash, 'shares': shares, 'price': price})

    port_df = pd.DataFrame(portfolio).set_index('date')
    if not port_df.empty:
        port_df['returns'] = port_df['total'].pct_change().fillna(0)

    fig = plt.figure()
    plt.plot(data.index, data[price_col], label='price')
    buys = data[data['positions'] == 1].index
    sells = data[data['positions'] == -1].index
    plt.scatter(buys, data.loc[buys, price_col], marker='^', color='g')
    plt.scatter(sells, data.loc[sells, price_col], marker='v', color='r')
    plt.title('RSI Strategy Signals')
    plt.legend()
    save_plot(fig, 'rsi_signals.png')

    return port_df

# ---------------- Performance Metrics ----------------

def performance_summary(port_df, initial_capital=100000):
    if port_df.empty:
        return {}
    total_return = port_df['total'].iloc[-1] / initial_capital - 1
    cumulative_returns = (1 + port_df['returns']).cumprod() - 1
    daily_returns = port_df['returns']
    ann_return = (1 + cumulative_returns.iloc[-1]) ** (252 / len(port_df)) - 1 if len(port_df) > 0 else np.nan
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    summary = {
        'total_return_pct': float(total_return * 100),
        'annualized_return_pct': float(ann_return * 100),
        'annualized_vol_pct': float(ann_vol * 100),
        'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else None,
        'max_drawdown_pct': float(max_dd * 100) if not np.isnan(max_dd) else None
    }
    return summary

# ---------------- Streamlit App (optional) ----------------

def run_streamlit_app(df):
    if not STREAMLIT_AVAILABLE:
        print('Streamlit not installed. Install streamlit to run dashboard: pip install streamlit')
        return
    app_code = f"""
import streamlit as st
import pandas as pd

@st.cache
def load_df():
    return pd.read_csv(r'{DATA_PATH}', parse_dates=['date'])

df = load_df()
st.title('Palantir Stock Dashboard')
price_col = 'adj_close' if 'adj_close' in df.columns else ('close' if 'close' in df.columns else df.columns[1])
start = st.date_input('Start date', df['date'].min().date())
end = st.date_input('End date', df['date'].max().date())
mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
sub = df.loc[mask]
st.line_chart(sub.set_index('date')[price_col])
"""
    app_path = os.path.join(OUTPUT_DIR, 'streamlit_app.py')
    with open(app_path, 'w') as f:
        f.write(app_code)
    print(f"Wrote Streamlit helper to {app_path}. Run: streamlit run {app_path}")

# ---------------- Main Pipeline ----------------

def main(mode='full'):
    df = load_data()
    df2 = add_technical_indicators(df)
    eda_plots(df2)

    # Forecasting
    if STATSMODELS_AVAILABLE:
        _ = fit_arima_forecast(df2, forecast_horizon=60)
    else:
        print('Skipping ARIMA (statsmodels not installed)')

    if TENSORFLOW_AVAILABLE:
        _ = fit_lstm(df2, look_back=60, epochs=20)
    else:
        print('Skipping LSTM (tensorflow not installed)')

    # Backtesting
    port_ma = backtest_ma_crossover(df2, short_window=20, long_window=50)
    summary_ma = performance_summary(port_ma)
    print('\nMA Crossover Performance Summary:')
    print(summary_ma)

    port_rsi = backtest_rsi_strategy(df2)
    summary_rsi = performance_summary(port_rsi)
    print('\nRSI Strategy Performance Summary:')
    print(summary_rsi)

    # Save sample outputs
    if summary_ma:
        pd.DataFrame([summary_ma]).to_csv(os.path.join(OUTPUT_DIR, 'ma_summary.csv'), index=False)
    if summary_rsi:
        pd.DataFrame([summary_rsi]).to_csv(os.path.join(OUTPUT_DIR, 'rsi_summary.csv'), index=False)
    df2.to_csv(os.path.join(OUTPUT_DIR, 'data_with_indicators.csv'), index=False)
    print(f"Saved processed data to {os.path.join(OUTPUT_DIR, 'data_with_indicators.csv')}")

    if mode == 'streamlit':
        run_streamlit_app(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', help='full / streamlit')
    args = parser.parse_args()
    main(mode=args.mode)
