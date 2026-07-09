"""
主流程：
  1. 取得全台股清單 -> 篩選近一個月成交量前 50 大
  2. 抓取這 50 檔股票的歷史日線 -> 特徵工程 + 標籤
  3. 依時間切分 train / validation / test
  4. 訓練模型 -> 在 validation 上回測 -> 檢查是否達到績效門檻
  5. 未達門檻則自動調整模型參數 / 特徵設定，重新訓練，最多 MAX_ITERATIONS 輪
  6. 用「最後選定」的設定，在從未看過的 test 集上做一次最終驗證與回測，產出報告
"""
import json
import os
import datetime as dt

import numpy as np
import pandas as pd

import config
import data_fetch
import features
import backtest
from model import DirectionModel


def get_top50_dataset(use_mock: bool = False):
    """回傳 (top50_list_df, {stock_id: history_df})"""
    if use_mock:
        import mock_data
        return mock_data.generate_mock_universe()

    stock_list = data_fetch.get_all_stock_list()
    if stock_list.empty:
        raise RuntimeError("無法取得股票清單，請確認網路連線可存取 TWSE / TPEx 開放資料 API。")

    top50 = data_fetch.get_top_volume_stocks(stock_list)
    if top50.empty:
        raise RuntimeError("無法取得近一個月成交量資料，請確認 TWSE MI_INDEX / TPEx API 是否正常。")

    histories = {}
    for _, row in top50.iterrows():
        sid, market = row["stock_id"], row["market"]
        print(f"抓取歷史資料: {sid} ({row.get('name','')}, {market})")
        hist = data_fetch.get_stock_history(sid, market)
        if hist is not None and len(hist) > 100:
            histories[sid] = hist
        else:
            print(f"  [略過] {sid} 歷史資料不足")
    return top50, histories


def build_feature_dataset(histories: dict, horizon: int):
    frames = []
    for sid, hist in histories.items():
        feat = features.build_dataset(hist, horizon=horizon)
        if feat.empty:
            continue
        feat["stock_id"] = sid
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    return full.sort_values("date").reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    """依「整體日期」切分 train/val/test，避免同一天的不同股票同時落在不同集合造成資訊外洩過細，
    這裡以日期分位數切分，train 用最早的資料，test 是最新一段從未參與訓練/調參的資料。"""
    dates = df["date"].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * config.TRAIN_RATIO) - 1]
    val_end = dates[int(n * (config.TRAIN_RATIO + config.VAL_RATIO)) - 1]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test


def evaluate(model: DirectionModel, dataset: pd.DataFrame, feature_cols, horizon, threshold):
    X = dataset[feature_cols].values
    y = dataset["label"].values
    proba = model.predict_proba(X)
    pred = (proba >= threshold).astype(int)
    accuracy = float((pred == y).mean())

    sim = backtest.simulate_strategy(dataset, proba, horizon=horizon, threshold=threshold)
    metrics = backtest.compute_metrics(sim)
    metrics["accuracy"] = accuracy
    return metrics, sim


def passes_threshold(metrics: dict) -> bool:
    return (
        metrics["accuracy"] >= config.MIN_TEST_ACCURACY
        and metrics["excess_annual_return"] >= config.MIN_EXCESS_ANNUAL_RETURN
        and (not np.isnan(metrics["sharpe"]) and metrics["sharpe"] >= config.MIN_SHARPE)
    )


# 每輪迭代嘗試的參數/特徵變體（依序嘗試，找出在 validation 上表現最好的一組）
ITERATION_GRID = [
    dict(n_estimators=100, max_depth=6, min_samples_split=20, threshold=0.50, horizon=config.PREDICTION_HORIZON),
    dict(n_estimators=200, max_depth=4, min_samples_split=30, threshold=0.55, horizon=config.PREDICTION_HORIZON),
    dict(n_estimators=200, max_depth=8, min_samples_split=15, threshold=0.55, horizon=config.PREDICTION_HORIZON),
    dict(n_estimators=150, max_depth=5, min_samples_split=25, threshold=0.60, horizon=10),
    dict(n_estimators=300, max_depth=6, min_samples_split=20, threshold=0.60, horizon=3),
]


def run_pipeline(use_mock: bool = False, out_dir: str = "report"):
    os.makedirs(out_dir, exist_ok=True)
    log = {"started_at": dt.datetime.now().isoformat(), "iterations": []}

    print("=" * 60)
    print("STEP 1: 取得近一個月成交量前 50 大股票")
    print("=" * 60)
    top50, histories = get_top50_dataset(use_mock=use_mock)
    top50.to_csv(os.path.join(out_dir, "top50_stocks.csv"), index=False)
    log["n_stocks_used"] = len(histories)
    print(f"完成，共取得 {len(histories)} 檔股票的可用歷史資料。\n")

    best_result = None
    chosen_horizon = None
    dataset_cache = {}

    print("=" * 60)
    print("STEP 2~5: 特徵工程 -> 訓練 -> 驗證集回測 -> 自動調整迭代")
    print("=" * 60)

    for i, params in enumerate(ITERATION_GRID[: config.MAX_ITERATIONS], start=1):
        horizon = params["horizon"]
        if horizon not in dataset_cache:
            dataset_cache[horizon] = build_feature_dataset(histories, horizon=horizon)
        full = dataset_cache[horizon]
        if full.empty:
            print(f"[Iteration {i}] horizon={horizon} 無可用資料，略過。")
            continue

        train, val, test = chronological_split(full)
        feature_cols = features.FEATURE_COLUMNS

        model = DirectionModel(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        )
        model.fit(train[feature_cols].values, train["label"].values)

        val_metrics, _ = evaluate(model, val, feature_cols, horizon, params["threshold"])

        print(f"\n[Iteration {i}] 參數={params}")
        print(f"  模型後端: {model.backend}")
        print(f"  Validation -> 準確率={val_metrics['accuracy']:.3f}, "
              f"策略年化報酬={val_metrics['annual_return']:.2%}, "
              f"benchmark年化={val_metrics['benchmark_annual_return']:.2%}, "
              f"Sharpe={val_metrics['sharpe']:.2f}")

        iter_log = {
            "iteration": i,
            "params": params,
            "backend": model.backend,
            "val_metrics": val_metrics,
        }
        log["iterations"].append(iter_log)

        ok = passes_threshold(val_metrics)
        score = val_metrics["accuracy"] + val_metrics["excess_annual_return"]
        if best_result is None or score > best_result["score"]:
            best_result = {
                "score": score,
                "params": params,
                "model": model,
                "val_metrics": val_metrics,
                "horizon": horizon,
            }

        if ok:
            print(f"  -> 達到績效門檻，停止迭代。")
            break
        else:
            print(f"  -> 未達門檻 (需準確率>={config.MIN_TEST_ACCURACY:.0%}, "
                  f"excess年化報酬>={config.MIN_EXCESS_ANNUAL_RETURN:.0%}, "
                  f"Sharpe>={config.MIN_SHARPE})，調整參數重試...")

    if best_result is None:
        raise RuntimeError("所有迭代皆無可用資料，請確認股票歷史資料是否成功取得。")

    print("\n" + "=" * 60)
    print("STEP 6: 使用最佳設定，於從未參與訓練/調參的 Test 集做最終驗證")
    print("=" * 60)
    horizon = best_result["horizon"]
    full = dataset_cache[horizon]
    train, val, test = chronological_split(full)
    feature_cols = features.FEATURE_COLUMNS
    final_model = best_result["model"]
    final_threshold = best_result["params"]["threshold"]

    test_metrics, test_sim = evaluate(final_model, test, feature_cols, horizon, final_threshold)
    final_ok = passes_threshold(test_metrics)

    print(f"最終選定參數: {best_result['params']}")
    print(f"模型後端: {final_model.backend}")
    print(f"Test 集 -> 準確率={test_metrics['accuracy']:.3f}, "
          f"策略年化報酬={test_metrics['annual_return']:.2%}, "
          f"benchmark年化={test_metrics['benchmark_annual_return']:.2%}, "
          f"Sharpe={test_metrics['sharpe']:.2f}, "
          f"最大回撤={test_metrics['max_drawdown']:.2%}")
    print(f"是否達到預設績效門檻: {'是' if final_ok else '否（已達最大迭代次數，回報目前最佳結果）'}")

    log["final"] = {
        "chosen_params": best_result["params"],
        "backend": final_model.backend,
        "test_metrics": test_metrics,
        "passed_threshold": final_ok,
        "finished_at": dt.datetime.now().isoformat(),
    }

    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)

    test_sim.to_csv(os.path.join(out_dir, "test_backtest_detail.csv"), index=False)

    _write_markdown_report(top50, log, out_dir)
    _plot_equity_curve(test_sim, out_dir)

    print(f"\n報告已輸出至: {out_dir}/")
    return log


def _write_markdown_report(top50: pd.DataFrame, log: dict, out_dir: str):
    lines = []
    lines.append(f"# 台股量能前50檔 模型分析與回測報告\n")
    lines.append(f"產出時間: {log['final']['finished_at']}\n")
    lines.append(f"## 一、選股清單（近一個月成交量前 {len(top50)} 大）\n")
    lines.append(top50[["stock_id", "name", "market", "avg_volume"]].to_markdown(index=False))
    lines.append("\n## 二、自動迭代紀錄\n")
    lines.append("| 輪次 | n_estimators | max_depth | threshold | horizon | Val準確率 | Val超額年化報酬 | Val Sharpe |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for it in log["iterations"]:
        p, m = it["params"], it["val_metrics"]
        lines.append(
            f"| {it['iteration']} | {p['n_estimators']} | {p['max_depth']} | {p['threshold']} | "
            f"{p['horizon']} | {m['accuracy']:.3f} | {m['excess_annual_return']:.2%} | {m['sharpe']:.2f} |"
        )

    final = log["final"]
    tm = final["test_metrics"]
    lines.append("\n## 三、最終模型（Test 集，從未參與訓練/調參）\n")
    lines.append(f"- 選定參數: `{final['chosen_params']}`")
    lines.append(f"- 模型後端: {final['backend']}")
    lines.append(f"- 準確率: {tm['accuracy']:.3f}")
    lines.append(f"- 策略總報酬: {tm['total_return']:.2%}　策略年化報酬: {tm['annual_return']:.2%}")
    lines.append(f"- Benchmark(等權重買進持有) 年化報酬: {tm['benchmark_annual_return']:.2%}")
    lines.append(f"- 超額年化報酬: {tm['excess_annual_return']:.2%}")
    lines.append(f"- 勝率: {tm['win_rate']:.2%}　交易次數: {tm['n_trades']}")
    lines.append(f"- Sharpe: {tm['sharpe']:.2f}　最大回撤: {tm['max_drawdown']:.2%}")
    lines.append(f"- 是否達到預設績效門檻: {'是' if final['passed_threshold'] else '否'}")
    lines.append(
        "\n> 免責聲明：本報告與程式僅供研究與教育用途，不構成投資建議。"
        "歷史績效不代表未來表現，實際交易請自行評估風險。\n"
    )

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_equity_curve(test_sim: pd.DataFrame, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        strategy_daily, benchmark_daily = backtest.build_portfolio_daily_series(test_sim)
        strat_equity = backtest.compute_equity_curve(strategy_daily)
        bench_equity = backtest.compute_equity_curve(benchmark_daily)

        # 圖表文字使用英文，避免使用者環境缺少中文字型導致亂碼 / 方框
        plt.figure(figsize=(10, 5))
        plt.plot(strat_equity.index, strat_equity.values, label="Model Strategy")
        plt.plot(bench_equity.index, bench_equity.values, label="Benchmark (Equal-weight Buy & Hold)")
        plt.title("Test Set Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"[警告] 繪圖失敗: {e}")
