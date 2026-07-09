"""
技術指標模組（選股評分模型專用）

重要說明：使用者原本希望以「60分鐘K線」計算 60MA / KD(60,3,3)，但台股目前沒有
免費的官方 API 可取得個股的分鐘級歷史資料（TWSE/TPEx 開放資料僅提供日K）。
經與使用者確認，本模組改以「日K線」計算對應參數（60日MA、KD(60,3,3)、
布林通道(20,2)），指標定義與運算邏輯完全相同，只是時間單位由「60分鐘」
換成「日」。未來若要接上券商/看盤軟體的分K資料，只需把 daily OHLCV
換成分K OHLCV 餵進本模組的函式即可，不需改動計算邏輯。

包含：
- MA / KD(N,K,D) / 布林通道(20,2) + 帶寬
- 乖離率 (BIAS)
- 葛蘭碧法則 8 大信號中，使用者指定的 5 條判斷規則
- 布林型態分類（窄/正常/寬、上緣/中間/下緣、是否為「開喇叭」擴張型態）
"""
import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------
# 基礎指標
# ----------------------------------------------------------------------
def add_moving_averages(df: pd.DataFrame, windows) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    return df


def add_kd(df: pd.DataFrame, n: int = None, k_smooth: int = None, d_smooth: int = None) -> pd.DataFrame:
    """KD 指標 (RSV 週期 n，K/D 平滑期 k_smooth/d_smooth)。
    採台股慣用公式：K_t = K_{t-1}*(1-1/m) + RSV_t*(1/m)，D 同理由 K 平滑。
    初始 K=D=50。
    """
    n = n or config.KD_N
    k_smooth = k_smooth or config.KD_K_SMOOTH
    d_smooth = d_smooth or config.KD_D_SMOOTH

    df = df.copy()
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)

    k_vals, d_vals = [], []
    k_prev, d_prev = 50.0, 50.0
    for v in rsv:
        k_prev = k_prev * (1 - 1 / k_smooth) + v * (1 / k_smooth)
        d_prev = d_prev * (1 - 1 / d_smooth) + k_prev * (1 / d_smooth)
        k_vals.append(k_prev)
        d_vals.append(d_prev)

    df["k_value"] = k_vals
    df["d_value"] = d_vals
    return df


def add_bollinger(df: pd.DataFrame, window: int = None, std_mult: float = None) -> pd.DataFrame:
    """布林通道：中軌=window日SMA，上下軌=中軌±std_mult個標準差。
    帶寬 = (上軌-下軌)/中軌，這是業界標準定義（使用者原提供公式疑似筆誤，已依標準公式實作，
    並在報告中沿用「10%正常、5%以下窄、20%以上寬」的判斷門檻）。
    """
    window = window or config.BOLL_WINDOW
    std_mult = std_mult or config.BOLL_STD_MULT

    df = df.copy()
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std

    df["boll_mid"] = mid
    df["boll_upper"] = upper
    df["boll_lower"] = lower
    df["boll_bandwidth"] = (upper - lower) / mid.replace(0, np.nan)
    df["boll_percent_b"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    return df


def add_bias(df: pd.DataFrame, ma_window: int = None) -> pd.DataFrame:
    """乖離率 BIAS = (收盤價 - MA) / MA * 100，預設用 60 日均線近似「季線」。"""
    ma_window = ma_window or config.BIAS_MA_WINDOW
    df = df.copy()
    ma = df["close"].rolling(ma_window).mean()
    df[f"bias_{ma_window}"] = (df["close"] - ma) / ma.replace(0, np.nan) * 100
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次算好選股模型需要的全部指標，df 需含 date, open, high, low, close, volume，依日期排序。"""
    df = df.sort_values("date").reset_index(drop=True).copy()
    df = add_moving_averages(df, [config.GRANVILLE_MA_WINDOW, config.BIAS_MA_WINDOW])
    df = add_kd(df)
    df = add_bollinger(df)
    df = add_bias(df, config.BIAS_MA_WINDOW)
    return df


# ----------------------------------------------------------------------
# 葛蘭碧法則（使用者指定的 5 條規則）
# ----------------------------------------------------------------------
def classify_granville(df: pd.DataFrame, lookback: int = 5, bias_extreme: float = 8.0) -> dict:
    """
    依「月線」(config.GRANVILLE_MA_WINDOW，預設20日) 的斜率與股價相對關係，
    判斷屬於使用者提供的哪一條葛蘭碧規則：
      買點: 1) 均線上揚且股價突破均線  2) 均線上揚，股價拉回但獲得均線支撐
      賣點: 3) 均線上揚但乖離過大  4) 均線準備往下且股價跌破均線
            5) 均線往下，股價反彈超過均線
    回傳 dict: {signal: 'buy_breakout'/'buy_support'/'sell_overbought'/'sell_breakdown'/'sell_rebound'/'neutral',
                reason: 文字說明, ma_slope: 斜率方向, bias_pct: 乖離率}
    """
    ma_col = f"ma{config.GRANVILLE_MA_WINDOW}"
    if ma_col not in df.columns or len(df) < lookback + config.GRANVILLE_MA_WINDOW + 1:
        return {"signal": "neutral", "reason": "資料不足，無法判斷葛蘭碧法則", "ma_slope": 0.0, "bias_pct": np.nan}

    recent = df.tail(lookback + 1).reset_index(drop=True)
    ma_now = recent[ma_col].iloc[-1]
    ma_prev = recent[ma_col].iloc[-lookback - 1]
    ma_slope = (ma_now - ma_prev) / abs(ma_prev) if ma_prev else 0.0

    close_now = recent["close"].iloc[-1]
    close_prev = recent["close"].iloc[-2]
    ma_prev1 = recent[ma_col].iloc[-2]

    bias_pct = (close_now - ma_now) / ma_now * 100 if ma_now else np.nan

    crossed_up = close_prev <= ma_prev1 and close_now > ma_now
    crossed_down = close_prev >= ma_prev1 and close_now < ma_now
    near_support = (
        recent["low"].iloc[-1] <= ma_now * 1.01
        and close_now >= ma_now
        and ma_slope > 0
    )

    ma_rising = ma_slope > 0.002   # 均線近似走平以上才算「上揚」
    ma_falling = ma_slope < -0.002

    if ma_rising and crossed_up:
        return {"signal": "buy_breakout", "reason": "月線上揚，股價向上突破月線 -> 葛蘭碧買點①",
                "ma_slope": ma_slope, "bias_pct": bias_pct}
    if ma_rising and pd.notna(bias_pct) and bias_pct > bias_extreme:
        return {"signal": "sell_overbought", "reason": f"月線上揚但乖離率達{bias_pct:.1f}%，過度乖離 -> 葛蘭碧賣點③",
                "ma_slope": ma_slope, "bias_pct": bias_pct}
    if ma_rising and near_support:
        return {"signal": "buy_support", "reason": "月線上揚，股價拉回獲得月線支撐 -> 葛蘭碧買點②",
                "ma_slope": ma_slope, "bias_pct": bias_pct}
    if ma_falling and crossed_down:
        return {"signal": "sell_breakdown", "reason": "月線準備往下，股價跌破月線 -> 葛蘭碧賣點④",
                "ma_slope": ma_slope, "bias_pct": bias_pct}
    if ma_falling and crossed_up:
        return {"signal": "sell_rebound", "reason": "月線往下，股價反彈超過月線 -> 葛蘭碧賣點⑤",
                "ma_slope": ma_slope, "bias_pct": bias_pct}

    return {"signal": "neutral", "reason": "未明顯符合葛蘭碧買賣點規則", "ma_slope": ma_slope, "bias_pct": bias_pct}


# ----------------------------------------------------------------------
# 布林型態分類
# ----------------------------------------------------------------------
def classify_bollinger(df: pd.DataFrame, squeeze_lookback: int = 10, expand_ratio: float = 1.3) -> dict:
    """
    回傳布林通道目前的型態描述：
      bandwidth: 目前帶寬 (比例，如 0.08 代表 8%)
      band_state: 'narrow' / 'normal' / 'wide'
      position: 'upper' / 'lower' / 'middle'  (percent_b 決定)
      is_expanding: 近期帶寬是否明顯較 squeeze_lookback 根K棒之前擴大（開喇叭訊號）
      was_squeezed: squeeze_lookback 期間內帶寬是否曾經處於窄幅（開喇叭的前提）
    """
    if "boll_bandwidth" not in df.columns or len(df) < squeeze_lookback + config.BOLL_WINDOW:
        return {
            "bandwidth": np.nan, "band_state": "unknown", "position": "unknown",
            "is_expanding": False, "was_squeezed": False,
        }

    recent = df.tail(squeeze_lookback + 1)
    bandwidth_now = recent["boll_bandwidth"].iloc[-1]
    bandwidth_prior = recent["boll_bandwidth"].iloc[0]
    was_squeezed = bool((recent["boll_bandwidth"] < config.BOLL_BANDWIDTH_NARROW).any())
    is_expanding = bool(
        bandwidth_prior and bandwidth_now > bandwidth_prior * expand_ratio
    )

    if bandwidth_now < config.BOLL_BANDWIDTH_NARROW:
        band_state = "narrow"
    elif bandwidth_now >= config.BOLL_BANDWIDTH_WIDE:
        band_state = "wide"
    else:
        band_state = "normal"

    percent_b = df["boll_percent_b"].iloc[-1]
    if pd.isna(percent_b):
        position = "unknown"
    elif percent_b >= 0.8:
        position = "upper"
    elif percent_b <= 0.2:
        position = "lower"
    else:
        position = "middle"

    return {
        "bandwidth": float(bandwidth_now) if pd.notna(bandwidth_now) else np.nan,
        "band_state": band_state,
        "position": position,
        "is_expanding": is_expanding,
        "was_squeezed": was_squeezed,
    }


# ----------------------------------------------------------------------
# 左側交易「出量創高後拉回季線」分析
# ----------------------------------------------------------------------
def analyze_left_side_setup(df: pd.DataFrame) -> dict:
    """左側交易員思維的拉回買點分析：找出「近期曾經量增創高、目前拉回到季線
    (BIAS_MA_WINDOW=60日線)附近或以下」的股票，並計算：
      - current_pullback_pct：從近期高點到目前收盤價的回檔幅度(%)
      - typical_pullback_pct：這檔股票「歷史慣性」的拉回深度（用全部歷史資料的
        滾動高點回檔幅度分布之中位數估計，只採計>設定門檻的有效拉回，排除雜訊）
    回傳 dict，applicable=False 代表近期沒有出現這種「量增創高後拉回」的型態
    （不是每檔股票都適用這個條件，只是用來額外標記出這種特定型態的股票）。
    """
    lookback_days = config.LEFT_SIDE_LOOKBACK_DAYS
    volume_ma_window = config.LEFT_SIDE_VOLUME_MA_WINDOW
    volume_spike_mult = config.LEFT_SIDE_VOLUME_SPIKE_MULT
    near_ma_band = config.LEFT_SIDE_NEAR_MA_BAND
    ma_col = f"ma{config.BIAS_MA_WINDOW}"

    result = {
        "applicable": False,
        "volume_spike_confirmed": False,
        "near_or_below_ma": False,
        "high_price": np.nan,
        "days_since_high": None,
        "current_pullback_pct": np.nan,
        "typical_pullback_pct": np.nan,
        "reason": "近期未出現「量增創高後拉回」型態，此項不適用",
    }

    if ma_col not in df.columns or "volume" not in df.columns or len(df) < lookback_days + volume_ma_window:
        result["reason"] = "資料不足，無法判斷左側交易拉回型態，此項不適用"
        return result

    window = df.tail(lookback_days).reset_index(drop=True)
    if window["close"].dropna().empty:
        return result

    high_idx = int(window["close"].idxmax())
    high_price = float(window["close"].iloc[high_idx])
    current_close = float(window["close"].iloc[-1])
    days_since_high = len(window) - 1 - high_idx

    vol_ma = window["volume"].rolling(volume_ma_window).mean()
    lo, hi = max(0, high_idx - 5), min(len(window) - 1, high_idx + 5)
    vol_near_high = window["volume"].iloc[lo:hi + 1].max()
    vol_ma_at_high = vol_ma.iloc[high_idx]
    volume_spike_confirmed = bool(
        pd.notna(vol_ma_at_high) and vol_ma_at_high > 0
        and vol_near_high >= volume_spike_mult * vol_ma_at_high
    )

    ma_now = df[ma_col].iloc[-1]
    near_or_below_ma = bool(pd.notna(ma_now) and ma_now > 0 and current_close <= ma_now * (1 + near_ma_band))

    current_pullback_pct = (high_price - current_close) / high_price * 100 if high_price else np.nan

    # 歷史慣性拉回深度：全歷史資料，滾動高點(cummax)回檔幅度分布之中位數
    roll_max = df["close"].cummax()
    drawdown = (roll_max - df["close"]) / roll_max.replace(0, np.nan) * 100
    meaningful = drawdown[drawdown > config.LEFT_SIDE_DRAWDOWN_MIN_PCT]
    typical_pullback_pct = float(meaningful.median()) if len(meaningful) >= 10 else np.nan

    result.update({
        "high_price": high_price,
        "days_since_high": days_since_high,
        "volume_spike_confirmed": volume_spike_confirmed,
        "near_or_below_ma": near_or_below_ma,
        "current_pullback_pct": float(current_pullback_pct) if pd.notna(current_pullback_pct) else np.nan,
        "typical_pullback_pct": typical_pullback_pct,
    })

    applicable = bool(
        volume_spike_confirmed
        and days_since_high >= config.LEFT_SIDE_MIN_DAYS_SINCE_HIGH
        and pd.notna(current_pullback_pct) and current_pullback_pct > 0
    )
    result["applicable"] = applicable

    if applicable:
        typical_txt = f"，歷史慣性拉回幅度中位數約{typical_pullback_pct:.1f}%" if pd.notna(typical_pullback_pct) else ""
        ma_txt = "已拉回至季線附近或以下" if near_or_below_ma else "尚未拉回到季線附近"
        result["reason"] = (
            f"近{lookback_days}個交易日內量增創高{high_price:.2f}元（{days_since_high}日前），"
            f"目前拉回{current_pullback_pct:.1f}%，{ma_txt}{typical_txt}"
        )
    return result
