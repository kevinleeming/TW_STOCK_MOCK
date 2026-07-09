"""
回測模組
策略邏輯（單一股票）：
  在每個交易日，若模型預測「未來 horizon 日後上漲」機率 >= SIGNAL_THRESHOLD，
  則於當日收盤買進，持有 horizon 日後賣出；否則空手。

重要方法論說明（避免報酬灌水）：
  每一列 (stock, date) 的 future_return 是「未來 horizon 天」的報酬，同一檔股票
  相鄰日期的 future_return 期間會互相重疊。若把每一列都當成獨立、依序發生的單日
  報酬直接複利相乘，會把同一段價格波動重複計算好幾次，導致總報酬被嚴重高估。
  本模組的作法：
    1. 將每列的 horizon 日報酬換算成「等效日報酬」 (1+future_return)^(1/horizon) - 1，
       視為該筆持倉期間平均分攤到每一天的報酬。
    2. 依「實際日曆日期」把當天所有股票的等效日報酬做等權重平均，得到單一的
       投資組合日報酬序列（未持倉的股票當天貢獻 0）。
    3. 只對這條「依日期、不重疊」的組合報酬序列做複利加總，計算總報酬 / 年化報酬 /
       Sharpe / 最大回撤，避免重疊期間造成的灌水問題。
  勝率 (win_rate) 則是以「每一筆訊號」為單位計算，不受上述時間重疊問題影響。
"""
import numpy as np
import pandas as pd

import config


def simulate_strategy(df: pd.DataFrame, proba: np.ndarray, horizon: int = None,
                       threshold: float = None, cost_rate: float = None) -> pd.DataFrame:
    """
    df: 需含 date, close, future_return, return_1d 欄位（與 proba 對齊的同一組資料列）
    proba: 模型預測「上漲」機率，長度需與 df 相同
    回傳: 附加 signal / trade_return / daily_equiv_return 欄位的 DataFrame
    """
    horizon = horizon or config.PREDICTION_HORIZON
    threshold = threshold if threshold is not None else config.SIGNAL_THRESHOLD
    cost_rate = cost_rate if cost_rate is not None else config.TRANSACTION_COST_RATE

    out = df.copy().reset_index(drop=True)
    out["proba"] = proba
    out["signal"] = (out["proba"] >= threshold).astype(int)

    # 每筆「交易」的實際報酬（用於勝率等交易層級統計，不涉及跨列複利）
    out["trade_return"] = np.where(
        out["signal"] == 1, out["future_return"] - cost_rate, np.nan
    )

    # 換算成等效日報酬，用於之後依日期彙整成不重疊的組合日報酬序列
    growth = np.clip(1 + out["future_return"].values, 1e-6, None)
    daily_equiv = growth ** (1.0 / horizon) - 1
    daily_cost = cost_rate / horizon
    out["daily_equiv_return"] = np.where(
        out["signal"] == 1, daily_equiv - daily_cost, 0.0
    )
    return out


def build_portfolio_daily_series(sim: pd.DataFrame):
    """把逐列 (stock, date) 資料，依日期等權重彙整成投資組合日報酬序列。
    回傳 (strategy_daily, benchmark_daily)：兩者皆為以 date 為 index 的 pd.Series。
    """
    g = sim.groupby("date")
    strategy_daily = g["daily_equiv_return"].mean().sort_index()
    # benchmark: 同一組股票的等權重買進持有，用「真實」單日報酬 (return_1d) 彙整，
    # 不使用 future_return，避免同樣的重疊問題。
    if "return_1d" in sim.columns:
        benchmark_daily = g["return_1d"].mean().sort_index()
    else:
        benchmark_daily = pd.Series(0.0, index=strategy_daily.index)
    return strategy_daily, benchmark_daily


def compute_equity_curve(returns: pd.Series, initial_capital: float = None) -> pd.Series:
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    equity = initial_capital * (1 + returns.fillna(0)).cumprod()
    return equity


def compute_metrics(sim: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """計算策略績效指標，並與 benchmark（等權重買進持有）比較。
    sim: simulate_strategy() 的輸出（逐列 stock x date）。
    """
    strategy_daily, benchmark_daily = build_portfolio_daily_series(sim)

    trade_returns = sim["trade_return"].dropna()
    n_trades = int(len(trade_returns))
    win_rate = float((trade_returns > 0).mean()) if n_trades > 0 else float("nan")

    n_days = len(strategy_daily)
    total_return = float((1 + strategy_daily).prod() - 1)
    bench_total_return = float((1 + benchmark_daily).prod() - 1)

    years = max(n_days / periods_per_year, 1e-6)
    ann_return = (1 + total_return) ** (1 / years) - 1
    bench_ann_return = (1 + bench_total_return) ** (1 / years) - 1

    vol = strategy_daily.std() * np.sqrt(periods_per_year) if strategy_daily.std() > 0 else np.nan
    sharpe = (strategy_daily.mean() * periods_per_year) / vol if vol and vol > 0 else np.nan

    equity = compute_equity_curve(strategy_daily)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else float("nan")

    return {
        "n_obs": n_days,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "annual_return": ann_return,
        "benchmark_total_return": bench_total_return,
        "benchmark_annual_return": bench_ann_return,
        "excess_annual_return": ann_return - bench_ann_return,
        "sharpe": float(sharpe) if sharpe == sharpe else float("nan"),
        "max_drawdown": max_drawdown,
    }
