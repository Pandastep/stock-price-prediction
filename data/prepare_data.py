import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import yaml

def download_data(tickers, start_date, end_date):
    """Download OHLCV data for multiple tickers"""
    os.makedirs("data/raw", exist_ok=True)
    for ticker in tqdm(tickers, desc="Downloading data"):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index().to_csv(f"data/raw/{ticker}.csv", index=False)

    return tickers

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df['RSI'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26

    # Moving Averages
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_21'] = df['Close'].rolling(21).mean()

    # Other features
    df['Volatility'] = df['High'] - df['Low']
    df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Open_Close_Diff'] = df['Open'] - df['Close']
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Low']
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_upper'] = rolling_mean + (2 * rolling_std)
    df['Bollinger_lower'] = rolling_mean - (2 * rolling_std)
    df['Bollinger_width'] = df['Bollinger_upper'] - df['Bollinger_lower']

    df['Momentum_3'] = df['Close'] - df['Close'].shift(3)
    df['Momentum_7'] = df['Close'] - df['Close'].shift(7)

    import warnings
    warnings.filterwarnings("ignore")

    df['OBV'] = 0.0

    obv = [0]

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])

    df['OBV'] = obv

    return df.dropna()


def create_dataset(df, window_size):
    features = df[[
        'RSI', 'MACD', 'MACD_signal',
        'Volatility', 'Daily_Return', 'Volume_Change',
        'Open_Close_Diff', 'High_Low_Pct',
        'Momentum_3', 'Momentum_7',
        'Bollinger_width',
        'OBV'
    ]].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        change = (df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1]
        y.append(1 if change > 0.002 else 0)  # рост более 0.2%

    print(f"Label counts: {np.unique(y, return_counts=True)}")

    return np.array(X), np.array(y), scaler

def prepare_all_data(config):
    """Prepare data for all tickers"""
    os.makedirs("data/processed", exist_ok=True)
    all_X, all_y = [], []
    
    for ticker in config['data']['tickers']:
        df = pd.read_csv(f"data/raw/{ticker}.csv", index_col='Date')
        df.index = pd.to_datetime(df.index)

        df = add_technical_indicators(df)
        df = df.sort_index()

        X, y, _ = create_dataset(df, config['data']['window_size'])

        all_X.append(X)
        all_y.append(y)
        print(f"{ticker} → X.shape: {X.shape}, y.shape: {y.shape}")
    
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    
    # Train/val/test split
    n = len(X)
    train_end = int(n * config['data']['train_ratio'])
    val_end = train_end + int(n * config['data']['val_ratio'])
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    print("New label counts:", np.bincount(np.array(y).astype(int)))

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    download_data(config['data']['tickers'], 
                 config['data']['start_date'], 
                 config['data']['end_date'])
    prepare_all_data(config)
