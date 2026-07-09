"""
模擬資料產生器 —— 僅用於在無法連線 TWSE / TPEx / TDCC 的環境中，
驗證整條流程（ML回測工具 main.py 與 選股評分工具 screener_main.py）是否能正常運作。
正式使用時請改用 data_fetch.py / chip_data.py 抓取真實資料（執行時不加 --mock 即可）。
"""
import numpy as np
import pandas as pd

import config


def _random_walk_ohlcv(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    # 帶一點動能與均值回歸的隨機報酬，讓標籤不是純雜訊，方便驗證模型/回測邏輯
    momentum = 0.0
    returns = []
    for _ in range(n_days):
        momentum = 0.9 * momentum + rng.normal(0, 0.015)
        returns.append(momentum * 0.3 + rng.normal(0, 0.012))
    returns = np.array(returns)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    # mean=13, sigma=0.6 -> 中位數約44萬股(約440張)，貼近真實股票規模，
    # 確保 screener_pipeline 新增的流動性門檻(近期均量>=300張)在 --mock 模式下
    # 也能正常運作、不會把所有模擬股票都濾掉。
    volume = rng.lognormal(mean=13.0, sigma=0.6, size=n_days)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def generate_mock_universe(n_stocks: int = 50, n_days: int = 600):
    """給 ML 回測工具 (main.py / pipeline.py) 使用的模擬資料。"""
    rows = []
    histories = {}
    for i in range(n_stocks):
        sid = f"MOCK{i:03d}"
        hist = _random_walk_ohlcv(n_days, seed=i)
        histories[sid] = hist
        rows.append(
            {
                "stock_id": sid,
                "name": f"模擬股{i:03d}",
                "market": "TWSE" if i % 2 == 0 else "TPEx",
                "avg_volume": hist["volume"].tail(22).mean(),
            }
        )
    top50 = pd.DataFrame(rows).sort_values("avg_volume", ascending=False).reset_index(drop=True)
    return top50, histories


# ========================================================================
# 以下為「選股評分模型」(screener_*.py) 專用的模擬資料
# ========================================================================

MOCK_HISTORIES = {}  # screener_pipeline 會從這裡讀取模擬的日K歷史


def generate_mock_screener_universe(n_stocks: int = 80, n_days: int = 300):
    """模擬全市場股票清單 + 最新收盤快照，並把日K歷史存進 MOCK_HISTORIES 供後續評分使用。"""
    global MOCK_HISTORIES
    MOCK_HISTORIES = {}

    list_rows, snap_rows = [], []
    for i in range(n_stocks):
        sid = f"S{i:04d}"
        market = "TWSE" if i % 2 == 0 else "TPEx"
        hist = _random_walk_ohlcv(n_days, seed=1000 + i)
        MOCK_HISTORIES[sid] = hist

        mock_industries = ["半導體業", "電子零組件業", "電腦及週邊設備業", "金融保險業",
                           "塑膠工業", "紡織纖維", "航運業", "生技醫療業", "食品工業", "鋼鐵工業"]
        list_rows.append({
            "stock_id": sid, "name": f"模擬公司{i:04d}", "market": market,
            "industry": mock_industries[i % len(mock_industries)],
        })
        snap_rows.append({
            "stock_id": sid,
            "close": float(hist["close"].iloc[-1]),
            "volume": float(hist["volume"].tail(22).mean()),
            "date": hist["date"].iloc[-1].date().isoformat(),
        })

    stock_list = pd.DataFrame(list_rows)
    snapshot = pd.DataFrame(snap_rows)
    return stock_list, snapshot


def generate_mock_chip_data(stock_ids, n_weeks: int = 4, seed: int = 42):
    """模擬 TDCC 集保股權分散表(多週歷史)、三大法人買賣超、融資融券餘額。"""
    rng = np.random.RandomState(seed)

    # --- TDCC 集保股權分散表：模擬 n_weeks 週，每週每檔股票給幾個級距 bucket ---
    brackets = {
        "1": (1, 999), "9": (50001, 100000), "10": (100001, 200000),
        "11": (200001, 400000), "15": (1000001, 99999999),
    }
    tdcc_rows = []
    for w in range(n_weeks):
        week_date = (pd.Timestamp.today() - pd.Timedelta(weeks=(n_weeks - w))).date().isoformat()
        for sid in stock_ids:
            # 讓大戶比例隨週次緩步上升、股東人數緩步下降，模擬「籌碼變好」的個股情境；
            # 用 stock_id 的 hash 決定每檔股票的基準隨機性，讓不同股票有不同籌碼走勢
            stock_seed = abs(hash(sid)) % 1000
            trend = np.random.RandomState(stock_seed).normal(0, 1)  # >0 表示這檔籌碼變好

            base_holders_small = 8000 + stock_seed % 500
            base_big_shares_pct = 15 + trend * 2  # 大戶(15分級)基準占比

            for level, (lo, hi) in brackets.items():
                if level == "15":
                    shares = (base_big_shares_pct + trend * w * 0.3) / 100 * 50_000_000
                    holders = 50 + int(trend * w)
                elif level == "1":
                    shares = 5_000_000 - trend * w * 20_000
                    holders = base_holders_small - int(trend * w * 10)
                else:
                    shares = 8_000_000
                    holders = 500
                tdcc_rows.append({
                    "date": week_date, "stock_id": sid, "bracket": level,
                    "holders": max(holders, 1), "shares": max(shares, 1),
                })
    tdcc_history = pd.DataFrame(tdcc_rows)

    # --- 三大法人買賣超：近 INSTITUTION_LOOKBACK_DAYS 個交易日 ---
    inst_rows = []
    for d in range(config.INSTITUTION_LOOKBACK_DAYS):
        day = (pd.Timestamp.today() - pd.Timedelta(days=config.INSTITUTION_LOOKBACK_DAYS - d)).date().isoformat()
        for sid in stock_ids:
            stock_seed = abs(hash(sid)) % 1000
            bias = np.random.RandomState(stock_seed + 1).normal(0, 1)
            net = int(bias * 200_000 + rng.normal(0, 50_000))
            inst_rows.append({"stock_id": sid, "net_buy": net, "date": day})
    institution_df = pd.DataFrame(inst_rows)

    # --- 融資餘額：近 MARGIN_LOOKBACK_DAYS 個交易日 ---
    margin_rows = []
    for sid in stock_ids:
        stock_seed = abs(hash(sid)) % 1000
        bias = np.random.RandomState(stock_seed + 2).normal(0, 1)  # <0 表示融資持續下降(籌碼變好)
        base = 10_000_000
        for d in range(config.MARGIN_LOOKBACK_DAYS):
            day = (pd.Timestamp.today() - pd.Timedelta(days=config.MARGIN_LOOKBACK_DAYS - d)).date().isoformat()
            balance = base * (1 + bias * 0.02 * d)
            margin_rows.append({"stock_id": sid, "margin_balance": max(balance, 0), "date": day})
    margin_df = pd.DataFrame(margin_rows)

    return tdcc_history, institution_df, margin_df


# ========================================================================
# 以下為「產業/題材熱力圖」(sector_*.py) 專用模擬資料
# ========================================================================

def generate_mock_sector_universe(n_extra_stocks: int = 120, seed: int = 7):
    """模擬全市場股票清單 + 當日(對前一交易日)漲跌幅快照，用於在無網路環境驗證
    sector_pipeline.py 整條流程。刻意把 theme_map.py 裡真實的 stock_id 也納入，
    讓題材分組邏輯能被實際跑到；其餘用隨機產生的模擬股票代號補滿候選池、
    也用來驗證「官方產業別」保底分組路徑。
    """
    import theme_map
    rng = np.random.RandomState(seed)

    real_industries = ["半導體業", "電子零組件業", "電腦及週邊設備業", "金融保險業",
                        "塑膠工業", "紡織纖維", "航運業", "生技醫療業", "食品工業", "鋼鐵工業"]

    list_rows, price_rows = [], []

    # 1) theme_map.py 裡的真實 stock_id，確保熱門題材分組看得到資料
    theme_ids = sorted({sid for ids in theme_map.THEME_STOCKS.values() for sid in ids})
    for sid in theme_ids:
        market = "TWSE"
        industry = "半導體業" if rng.rand() < 0.6 else rng.choice(real_industries)
        list_rows.append({"stock_id": sid, "name": f"個股{sid}", "market": market, "industry": industry})
        prev_close = float(rng.uniform(30, 800))
        pct = float(rng.normal(0, 2.5))
        close = prev_close * (1 + pct / 100)
        volume = float(rng.lognormal(mean=15, sigma=1.0))
        price_rows.append({
            "stock_id": sid, "close": close, "prev_close": prev_close,
            "pct_change": pct, "volume": volume,
            "date": pd.Timestamp.today().date().isoformat(),
        })

    # 2) 額外模擬股票補滿候選池
    for i in range(n_extra_stocks):
        sid = f"M{9000 + i}"
        market = "TWSE" if i % 2 == 0 else "TPEx"
        industry = rng.choice(real_industries)
        list_rows.append({"stock_id": sid, "name": f"模擬公司{i:03d}", "market": market, "industry": industry})
        prev_close = float(rng.uniform(20, 500))
        pct = float(rng.normal(0, 2.0))
        close = prev_close * (1 + pct / 100)
        volume = float(rng.lognormal(mean=13, sigma=1.2))
        price_rows.append({
            "stock_id": sid, "close": close, "prev_close": prev_close,
            "pct_change": pct, "volume": volume,
            "date": pd.Timestamp.today().date().isoformat(),
        })

    stock_list = pd.DataFrame(list_rows)
    price_df = pd.DataFrame(price_rows)
    return stock_list, price_df


def generate_mock_institution_flow(candidates: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """模擬候選池「當日」三大法人買賣超（單位：股），用於驗證 sector_pipeline。"""
    rng = np.random.RandomState(seed)
    rows = []
    for _, row in candidates.iterrows():
        sid = row["stock_id"]
        # 讓漲的股票傾向法人買超、跌的傾向賣超，但帶隨機性，製造出「籌碼與價格不同步」的個案
        bias = row.get("pct_change", 0) * 50_000 * (1 if rng.rand() < 0.75 else -1)
        net_buy = bias + rng.normal(0, 300_000)
        rows.append({"stock_id": sid, "net_buy": net_buy, "date": row.get("date")})
    return pd.DataFrame(rows)
