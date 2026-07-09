"""
特徵工程與標籤建立
輸入: 單一股票的日線 DataFrame (欄位: date, open, high, low, close, volume)
輸出: 附加技術指標欄位 + 未來漲跌標籤 (label) 的 DataFrame
"""
import numpy as np
import pandas as pd

import config


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_features(df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
    """加入技術指標特徵及未來 N 日漲跌標籤。"""
    horizon = horizon or config.PREDICTION_HORIZON
    df = df.sort_values("date").reset_index(drop=True).copy()

    close = df["close"]
    volume = df["volume"]

    # 均線與乖離
    for w in config.MA_WINDOWS:
        df[f"ma{w}"] = close.rolling(w).mean()
        df[f"ma{w}_bias"] = (close - df[f"ma{w}"]) / df[f"ma{w}"]

    # 動能 / 報酬率特徵
    for w in [1, 5, 10, 20]:
        df[f"return_{w}d"] = close.pct_change(w)

    # 波動度
    df["volatility_20d"] = close.pct_change().rolling(20).std()

    # RSI
    df["rsi"] = _rsi(close, config.RSI_WINDOW)

    # MACD
    macd, signal, hist = _macd(close)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    # 成交量特徵
    df["volume_ma"] = volume.rolling(config.VOLUME_MA_WINDOW).mean()
    df["volume_change"] = volume.pct_change()
    df["volume_ratio"] = volume / df["volume_ma"].replace(0, np.nan)

    # 布林通道寬度
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bollinger_width"] = (4 * std20) / ma20.replace(0, np.nan)

    # 標籤: 未來 horizon 日後的收盤價是否上漲
    df["future_close"] = close.shift(-horizon)
    df["future_return"] = (df["future_close"] - close) / close
    df["label"] = (df["future_return"] > 0).astype(int)

    return df


FEATURE_COLUMNS = (
    [f"ma{w}_bias" for w in config.MA_WINDOWS]
    + [f"return_{w}d" for w in [1, 5, 10, 20]]
    + [
        "volatility_20d",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "volume_change",
        "volume_ratio",
        "bollinger_width",
    ]
)


def build_dataset(df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
    """計算特徵並移除因 rolling / shift 產生的 NaN 列。"""
    feat = compute_features(df, horizon=horizon)
    cols = ["date", "close"] + FEATURE_COLUMNS + ["future_return", "label"]
    cols = [c for c in cols if c in feat.columns]
    feat = feat[cols].dropna().reset_index(drop=True)
    return feat
