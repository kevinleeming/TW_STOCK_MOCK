"""
資料抓取模組
- 上市 (TWSE) 資料來源：
    * 股票清單: https://openapi.twse.com.tw/v1/opendata/t187ap03_L
    * 每日全市場行情(可指定歷史日期): https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=YYYYMMDD&type=ALLBUT0999&response=json
    * 個股歷史日K: 統一改走 FinMind 免費 API (dataset=TaiwanStockPrice)，見下方說明。
      TWSE 官方 https://www.twse.com.tw/exchangeReport/STOCK_DAY?date=YYYYMM01&stockNo=XXXX&response=json
      （舊版 /exchangeReport/ 路徑，不是新版 /rwd/zh/.../ 路徑）本身是可用的端點，但實測發現：
      候選池較大時（如300檔 x 近24個月 = 上千次請求）逐檔逐月查詢很容易觸發官網的
      反爬蟲/流量限制，導致部分請求回傳空內容（"Expecting value" 解析錯誤）或逾時，
      即使 URL 正確也會間歇性失敗。因此改以 FinMind 作為「主要」資料來源
      （見 _fetch_stock_history_finmind()，單一股票一次查詢即可涵蓋整段日期區間，
      請求量大幅降低），TWSE 官方逐月查詢(_fetch_twse_stock_month) 保留作為
      FinMind 查無資料時的備援（fallback），不會是每次都優先呼叫。
- 上櫃 (TPEx) 資料來源：
    * 股票清單: https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes
    * 每日全市場行情: https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes?date=YYY/MM/DD (民國年)
    * 個股歷史行情: 同上，改用 FinMind，原因是 TPEx 官網互動式查詢介面（tradingStock）
      沒有穩定可用的 JSON 端點，實測會回傳非預期格式。舊的 _fetch_tpex_stock_month()
      保留在檔案中僅作參考，目前未被呼叫。

FinMind 免費、免註冊即可用（匿名每小時 300 次請求），若候選池較大導致額度不足，
可至 https://finmindtrade.com 免費註冊帳號取得 token 填入 config.py 的
FINMIND_API_TOKEN，提高到每小時 600 次請求上限。

注意：政府 / 交易所網站偶爾會調整 API 格式或加上防爬蟲機制。
若程式執行時發生解析錯誤，請先確認對應網址是否仍回傳原本格式。
本模組所有對外請求皆有重試機制與逐筆延遲，避免對官方主機造成負擔。
"""
import os
import time
import json
import datetime as dt
from typing import List, Optional

import requests
import pandas as pd
import numpy as np

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# ------------------------------------------------------------------------
# FinMind 額度用盡（HTTP 402）偵測：
# FinMind 官方文件說明 402 代表「額度用完」（{'msg': 'Requests reach the upper
# limit', 'status': 402}），不是這個資料集本身要收費（TaiwanStockPrice/
# TaiwanStockMarginPurchaseShortSale/TaiwanStockInstitutionalInvestorsBuySell
# 都是免費資料集）。免費匿名額度是 300次/小時，候選池較大、或同一小時內重跑
# 過幾次，都容易把額度用光。一旦偵測到 402，本次執行剩下的 FinMind 請求會直接
# 跳過、不再逐檔重試 3 次，避免浪費時間、洗版一堆重複錯誤訊息；同時只印一次
# 清楚的解法說明。
# ------------------------------------------------------------------------
_finmind_quota_exhausted = False


def _is_finmind_quota_error(exc) -> bool:
    resp = getattr(exc, "response", None)
    return resp is not None and getattr(resp, "status_code", None) == 402


def _mark_finmind_quota_exhausted():
    global _finmind_quota_exhausted
    if _finmind_quota_exhausted:
        return
    _finmind_quota_exhausted = True
    print(
        "  [錯誤] FinMind 額度已用完（HTTP 402: Requests reach the upper limit）。"
        "免費匿名額度為 300次/小時，候選池較大或同一小時內重跑多次都容易超過。"
        "本次執行剩餘的 FinMind 請求將直接跳過（不再逐檔重試），這些股票的資料會"
        "顯示為「資料不足」。解決方式：(1) 免費註冊 https://finmindtrade.com 取得 "
        "API token，填入 config.py 的 FINMIND_API_TOKEN，額度可提高到 600次/小時"
        "（建議，一勞永逸）；(2) 調低 config.py 的 SCREENER_CANDIDATE_TOP_VOLUME；"
        "(3) 等待約1小時額度重置後再重新執行。"
    )


def _get_json(url: str, params: dict = None) -> Optional[dict]:
    """帶重試機制的 GET 請求，回傳 JSON（失敗回傳 None）。"""
    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = requests.get(
                url, params=params, headers=HEADERS, timeout=config.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            time.sleep(config.REQUEST_SLEEP_SEC)
            return resp.json()
        except Exception as e:
            print(f"  [警告] 第 {attempt} 次請求失敗: {url} params={params} -> {e}")
            time.sleep(1.0 * attempt)
    print(f"  [錯誤] 放棄請求: {url} params={params}")
    return None


def _cache_path(name: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, name)


def _cache_is_fresh(path: str) -> bool:
    """判斷快取檔是否為「今天」寫入的。用於「當天收盤價/近N日」這類每天都應該更新
    的資料；股票清單等變動很少的資料不套用此檢查。若快取是更早之前的日期寫入，
    視為過期，回傳 False（呼叫端應重新抓取並覆寫檔案，而不是永遠沿用舊資料）。

    注意：此函式只檢查「檔案是不是今天寫入的」，並不檢查檔案「內容的日期」是否為
    最新交易日。因此像「今天最新收盤價」這種資料，光靠這個函式並不夠 —— 如果今天
    第一次執行的當下，官方網站的今天收盤資料還沒公布，抓取邏輯會退回抓到前一個
    交易日的資料再寫入快取；此函式會把這個「今天寫入、但內容是舊交易日」的檔案
    誤判為新鮮，導致當天之後即使重新執行也不會再嘗試更新。真正需要「內容日期」
    是否為最新交易日的地方（get_latest_market_snapshot／get_stock_history）改用
    _content_date_is_fresh() 判斷，不要只看檔案 mtime。
    """
    if not os.path.exists(path):
        return False
    mtime_date = dt.date.fromtimestamp(os.path.getmtime(path))
    return mtime_date == dt.date.today()


def _most_recent_expected_trading_day(today: dt.date = None) -> dt.date:
    """回傳「預期」的最新交易日：若今天是週一到週五就是今天，若是週末則往前推到
    上週五。不考慮國定假日（國定假日當天官方本來就不會有資料，抓取迴圈會自動
    再往前一天找，不影響正確性，只是「預期值」會比實際最新交易日晚一點點）。
    """
    d = today or dt.date.today()
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def _content_date_is_fresh(cached_max_date: "dt.date | str", label: str = "") -> bool:
    """判斷快取「內容」的最新日期是否已經追上預期的最新交易日。
    與 _cache_is_fresh()（只看檔案 mtime）不同：這裡看的是資料本身記錄的日期，
    用來避免「今天稍早抓取時官方資料還沒更新、退回抓到前一交易日，之後就被誤判
    為新鮮而整天不再重抓」的問題 —— 只要內容日期還落後於預期交易日，就視為過期，
    讓呼叫端重新嘗試抓取（抓到的話，官方資料屆時應該已經更新了）。
    """
    if cached_max_date is None:
        return False
    if isinstance(cached_max_date, str):
        try:
            cached_max_date = dt.date.fromisoformat(cached_max_date[:10])
        except ValueError:
            return False
    expected = _most_recent_expected_trading_day()
    fresh = cached_max_date >= expected
    if not fresh:
        print(f"  [資訊] {label}快取內容日期為 {cached_max_date.isoformat()}，"
              f"落後於預期交易日 {expected.isoformat()}，將重新嘗試抓取最新資料...")
    return fresh


# ----------------------------------------------------------------------
# 1. 股票清單
# ----------------------------------------------------------------------
def get_twse_stock_list() -> pd.DataFrame:
    """取得上市公司清單 (股票代號、名稱)。"""
    cache = _cache_path("twse_list.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype=str)

    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    data = _get_json(url)
    rows = []
    if data:
        for item in data:
            code = item.get("公司代號")
            name = item.get("公司簡稱") or item.get("公司名稱")
            if code and code.isdigit() and len(code) == 4:
                rows.append({"stock_id": code, "name": name, "market": "TWSE"})
    df = pd.DataFrame(rows).drop_duplicates("stock_id")
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def get_tpex_stock_list() -> pd.DataFrame:
    """取得上櫃公司清單 (股票代號、名稱)。"""
    cache = _cache_path("tpex_list.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype=str)

    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    data = _get_json(url)
    rows = []
    if data:
        for item in data:
            code = item.get("SecuritiesCompanyCode") or item.get("CoID")
            name = item.get("CompanyName") or item.get("CoName")
            if code and str(code).isdigit() and len(str(code)) == 4:
                rows.append({"stock_id": str(code), "name": name, "market": "TPEx"})
    df = pd.DataFrame(rows).drop_duplicates("stock_id")
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def get_all_stock_list() -> pd.DataFrame:
    """合併上市 + 上櫃股票清單。"""
    twse = get_twse_stock_list()
    tpex = get_tpex_stock_list()
    df = pd.concat([twse, tpex], ignore_index=True)
    return df


def get_company_details() -> pd.DataFrame:
    """取得上市公司「詳細資料」（產業別、實收資本額、上市日期等），來源同 t187ap03_L。
    上櫃公司對應資料集欄位不盡相同，本函式為 best-effort，抓不到就跳過不影響其他流程。
    注意：這份只涵蓋上市(TWSE)公司，上櫃(TPEx)股票的產業別請改用下方 get_industry_map()
    （統一用 FinMind，同時涵蓋上市/上櫃/興櫃）。
    """
    cache = _cache_path("company_details.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"stock_id": str})

    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    data = _get_json(url)
    rows = []
    if data:
        for item in data:
            code = item.get("公司代號")
            if not (code and code.isdigit() and len(code) == 4):
                continue
            rows.append({
                "stock_id": code,
                "industry": item.get("產業別"),
                "capital": item.get("實收資本額"),
                "listed_date": item.get("上市日期"),
            })
    df = pd.DataFrame(rows).drop_duplicates("stock_id")
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def get_industry_map() -> pd.DataFrame:
    """取得全市場（上市+上櫃+興櫃）股票的產業別對照表，用於選股報告顯示「產業類型」欄位。
    來源改用 FinMind dataset=TaiwanStockInfo：這是單一份「全市場股票總覽」清單
    （不需要逐檔查詢、也不需要指定日期區間，一次請求就拿到全部），同時涵蓋上市/上櫃/
    興櫃，比 TWSE 官方 t187ap03_L（只有上市公司）更完整，可以避免上櫃股票查不到產業別。
    回傳 DataFrame[stock_id, industry]。
    """
    cache = _cache_path("industry_map.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"stock_id": str})

    headers = dict(HEADERS)
    if config.FINMIND_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.FINMIND_API_TOKEN}"
    params = {"dataset": "TaiwanStockInfo"}

    if _finmind_quota_exhausted:
        return pd.DataFrame()

    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = requests.get(FINMIND_URL, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 402:
                _mark_finmind_quota_exhausted()
                return pd.DataFrame()
            resp.raise_for_status()
            time.sleep(config.FINMIND_REQUEST_SLEEP_SEC)
            payload = resp.json()
            rows = payload.get("data", [])
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            if not {"stock_id", "industry_category"}.issubset(df.columns):
                print(f"[錯誤] FinMind TaiwanStockInfo 欄位無法辨識，實際欄位: {list(df.columns)}")
                return pd.DataFrame()
            out = df[["stock_id", "industry_category"]].rename(columns={"industry_category": "industry"})
            out = out.drop_duplicates("stock_id")
            out.to_csv(cache, index=False)
            return out
        except Exception as e:
            if _is_finmind_quota_error(e):
                _mark_finmind_quota_exhausted()
                return pd.DataFrame()
            print(f"  [警告] FinMind 第 {attempt} 次請求失敗: TaiwanStockInfo -> {e}")
            time.sleep(1.0 * attempt)
    print("[錯誤] FinMind 放棄請求: TaiwanStockInfo，產業類型欄位將留空。")
    return pd.DataFrame()


# ----------------------------------------------------------------------
# 4. 全市場「最新收盤日」快照（收盤價 + 成交量），用於選股評分模型
# ----------------------------------------------------------------------
def _fetch_twse_daily_snapshot(date_: dt.date) -> pd.DataFrame:
    """抓取指定日期上市全市場收盤價 + 成交量 + 當日漲跌幅。

    當日漲跌幅直接用這支API本身回傳的「漲跌(+/-)」(符號，包在一段HTML裡，例如
    "<p style= color:green>-</p>"，TW慣例紅漲綠跌) + 「漲跌價差」(絕對值)算出，
    不需要再另外抓「前一交易日」收盤價做merge計算——見 get_price_change_snapshot()
    說明，改成這樣是為了修正 TPEx 那邊的漲跌幅永遠算出0的bug，TWSE這邊當初其實
    沒有這個bug，但為了讓兩邊資料結構一致、且省掉一次額外的前一日API請求，一併
    改成同樣的「用當日資料本身欄位算漲跌幅」寫法。
    """
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    params = {"date": date_.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"}
    data = _get_json(url, params)
    if not data or "tables" not in data:
        return pd.DataFrame()

    for tbl in data["tables"]:
        fields = tbl.get("fields", [])
        if any("證券代號" in f for f in fields) and any("收盤價" in f for f in fields):
            rows = tbl.get("data", [])
            df = pd.DataFrame(rows, columns=fields)
            col_id = next((f for f in fields if "證券代號" in f), None)
            col_close = next((f for f in fields if "收盤價" in f), None)
            col_vol = next((f for f in fields if "成交股數" in f), None)
            col_sign = next((f for f in fields if f == "漲跌(+/-)"), None)
            col_diff = next((f for f in fields if f == "漲跌價差"), None)
            out = pd.DataFrame({
                "stock_id": df[col_id].astype(str).str.strip(),
                "close": pd.to_numeric(df[col_close].astype(str).str.replace(",", "", regex=False), errors="coerce"),
            })
            if col_vol:
                out["volume"] = pd.to_numeric(df[col_vol].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if col_sign and col_diff:
                sign = df[col_sign].astype(str).apply(lambda s: -1 if "-" in s else (1 if "+" in s else 0))
                diff_mag = pd.to_numeric(
                    df[col_diff].astype(str).str.replace(",", "", regex=False), errors="coerce"
                ).fillna(0)
                signed_diff = sign * diff_mag
                out["prev_close"] = out["close"] - signed_diff
                out["pct_change"] = signed_diff / out["prev_close"].replace(0, np.nan) * 100
            out["date"] = date_.isoformat()
            return out.dropna(subset=["close"])
    return pd.DataFrame()


def _fetch_tpex_daily_snapshot(date_: dt.date = None) -> pd.DataFrame:
    """抓取上櫃全市場最新收盤價 + 成交量 + 當日漲跌幅。

    重要：實測發現這支 TPEx openapi 的 `date` 參數會被「完全忽略」——不論傳入
    今天、昨天、還是一年多前的日期，回傳的都是同一份「目前最新一個交易日」的
    資料（用瀏覽器直接呼叫比對過 date=115/07/08、115/07/07、114/01/05 三種
    輸入，結果的 Date 欄位都固定是同一天）。

    這是先前「全部漲跌幅都是0、且上市權值股(2330等)整批消失」bug的根本原因：
    舊寫法會分別呼叫「今天」與「前一交易日」兩次來算漲跌幅，但 TPEx 這邊不管
    傳哪個日期都拿到同一份資料，兩次結果完全相同，減出來的漲跌幅當然都是0；
    同時因為呼叫當下 TWSE 「今天」的資料通常還沒公布（要收盤後才有），TWSE那
    段會是空的，最後拼出來的「最新一日」快照就只剩下 TPEx 股票，才會出現上市
    權值股整批消失的現象。

    修正方式：不再依賴 date 參數去對應「特定某一天」，直接抓「目前最新」這
    一份，並且用這份資料本身內建的 Close / Change 欄位直接算出當日漲跌幅
    （Change 已經是帶正負號的當日漲跌點數），不需要再抓第二個日期做merge。
    傳入的 date_ 參數目前僅作為抓不到資料時的日期標籤備援，不會實際送給API。
    """
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    data = _get_json(url)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    col_id = next((c for c in df.columns if c in ("SecuritiesCompanyCode", "CoID")), None)
    col_close = next((c for c in df.columns if "Close" in c), None)
    col_vol = next((c for c in df.columns if "Volume" in c or "Shares" in c), None)
    col_change = next((c for c in df.columns if c == "Change"), None)
    col_date = next((c for c in df.columns if c == "Date"), None)
    if col_id is None or col_close is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "stock_id": df[col_id].astype(str).str.strip(),
        "close": pd.to_numeric(df[col_close].astype(str).str.replace(",", "", regex=False), errors="coerce"),
    })
    if col_vol:
        out["volume"] = pd.to_numeric(df[col_vol].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if col_change:
        change = pd.to_numeric(
            df[col_change].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).fillna(0)
        out["prev_close"] = out["close"] - change
        out["pct_change"] = change / out["prev_close"].replace(0, np.nan) * 100
    # 用資料本身回傳的日期（民國年，例如"1150707"）還原實際西元日期，不要用傳入
    # 參數當日期標籤——反正參數也不會真的影響查詢結果，用回傳值才對得上實際資料。
    actual_date = None
    if col_date is not None and not df.empty:
        roc_str = str(df[col_date].iloc[0])
        try:
            roc_year, month, day = int(roc_str[:3]), int(roc_str[3:5]), int(roc_str[5:7])
            actual_date = dt.date(roc_year + 1911, month, day)
        except (ValueError, IndexError):
            actual_date = None
    out["date"] = (actual_date or date_ or dt.date.today()).isoformat()
    return out.dropna(subset=["close"])


def get_latest_market_snapshot(max_days_back: int = 10) -> pd.DataFrame:
    """回推最近幾個日曆日，取得「最新一個有資料的收盤日」全市場收盤價+成交量快照。
    此快照為「當天收盤價」，每天都應該更新。

    快取有效性改用「內容日期」判斷（_content_date_is_fresh），而不是單純看檔案
    mtime：如果只看 mtime，會發生「今天稍早執行時，官方今天的收盤資料還沒公布，
    抓取迴圈退回抓到前一個交易日的資料並寫入快取；之後同一天不管重跑幾次，都會
    因為 mtime 是今天而被誤判為新鮮，導致收盤價一直卡在前一個交易日」的問題。
    改成檢查內容日期後，只要快取內容還沒追上預期的最新交易日，就會重新嘗試抓取，
    抓到當天資料公布後即可自動更新，不需要手動清快取。
    """
    cache = _cache_path("latest_snapshot.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, dtype={"stock_id": str})
        cached_max_date = cached["date"].max() if not cached.empty and "date" in cached.columns else None
        if _content_date_is_fresh(cached_max_date, label="最新收盤快照"):
            return cached

    d = dt.date.today()
    for _ in range(max_days_back):
        if d.weekday() < 5:
            twse_df = _fetch_twse_daily_snapshot(d)
            tpex_df = _fetch_tpex_daily_snapshot(d)
            if not twse_df.empty or not tpex_df.empty:
                out = pd.concat([twse_df, tpex_df], ignore_index=True)
                out.to_csv(cache, index=False)
                print(f"[最新快照] 使用 {d.isoformat()} 收盤資料，共 {len(out)} 檔股票。")
                return out
        d -= dt.timedelta(days=1)

    print("[錯誤] 回推多日仍找不到全市場收盤快照，請確認網路連線或 API 是否異動。")
    return pd.DataFrame()


def get_price_change_snapshot(max_days_back: int = 10) -> pd.DataFrame:
    """取得全市場當日漲跌幅快照，用於產業/題材熱力圖。

    舊寫法是分別抓「最新交易日」+「前一交易日」兩天的收盤價，再merge算出漲跌幅。
    這個寫法在真實資料上會產生「全部漲跌幅都是0、且上市權值股整批消失」的bug，
    根本原因是 TPEx 的 tpex_mainboard_daily_close_quotes 這支openapi的 `date`
    參數其實會被忽略，不論查哪一天都回傳同一份「目前最新」的資料——導致「最新」
    「前一日」兩次TPEx查詢結果一模一樣，減出來的漲跌幅當然是0；同時因為呼叫
    當下 TWSE「今天」的資料通常還沒公布，最新一日的快照就只剩TPEx股票，上市
    股票（2330等）因此整批消失（詳見 _fetch_tpex_daily_snapshot() 說明）。

    修正後改成：TWSE/TPEx各自的「每日收盤行情」API本身就有附上當天的漲跌欄位，
    直接用單一天的資料自己算出漲跌幅即可，不再需要抓第二個日期做merge——TWSE
    那邊仍需要回推日期（找到「最新已公布」的交易日，因為TWSE的date參數是正確
    有效的，收盤前查當天會正確回傳空值）；TPEx則不需要回推，直接抓一次即可
    （見 _fetch_tpex_daily_snapshot() 說明，date參數對它無效）。

    回傳 DataFrame[stock_id, date, close, prev_close, pct_change, volume]。
    快取有效性用「內容日期」判斷（見 _content_date_is_fresh），確保收盤後重跑
    會抓到當天資料，不會卡在前一天。
    """
    cache = _cache_path("price_change_snapshot.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, dtype={"stock_id": str})
        cached_max_date = cached["date"].max() if not cached.empty and "date" in cached.columns else None
        if _content_date_is_fresh(cached_max_date, label="漲跌幅快照"):
            return cached

    # TWSE: date參數正確有效，回推到「最新已公布」的交易日
    d = dt.date.today()
    twse_df, twse_date = pd.DataFrame(), None
    for _ in range(max_days_back):
        if d.weekday() < 5:
            candidate = _fetch_twse_daily_snapshot(d)
            if not candidate.empty:
                twse_df, twse_date = candidate, d
                break
        d -= dt.timedelta(days=1)

    # TPEx: date參數會被忽略，只抓一次「目前最新」即可（見函式說明）
    tpex_df = _fetch_tpex_daily_snapshot()

    if twse_df.empty and tpex_df.empty:
        print("[錯誤] TWSE / TPEx皆抓不到全市場收盤快照，無法計算漲跌幅。")
        return pd.DataFrame()

    merged = pd.concat([twse_df, tpex_df], ignore_index=True)
    merged.to_csv(cache, index=False)
    twse_label = twse_date.isoformat() if twse_date else "（無資料）"
    tpex_label = tpex_df["date"].iloc[0] if not tpex_df.empty else "（無資料）"
    print(f"[漲跌幅快照] TWSE:{twse_label} / TPEx:{tpex_label}，共 {len(merged)} 檔股票。")
    return merged
    return merged


# ----------------------------------------------------------------------
# 2. 近一個月成交量 -> 前 N 大
# ----------------------------------------------------------------------
def _trading_days_back(n_days: int) -> List[dt.date]:
    """回推近 n_days 個「日曆日」中排除週末的日期（近似交易日，實際假日仍可能抓不到資料，程式會自動略過）。"""
    days = []
    d = dt.date.today()
    while len(days) < n_days:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:  # 0-4 = Mon-Fri
            days.append(d)
    return days


def _fetch_twse_daily_all(date_: dt.date) -> pd.DataFrame:
    """抓取指定日期上市全市場成交資訊。"""
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    params = {
        "date": date_.strftime("%Y%m%d"),
        "type": "ALLBUT0999",
        "response": "json",
    }
    data = _get_json(url, params)
    if not data or "tables" not in data:
        return pd.DataFrame()

    # MI_INDEX 回傳多個 table，找出含「成交股數」欄位的表格（個股行情表）
    for tbl in data["tables"]:
        fields = tbl.get("fields", [])
        if any("證券代號" in f for f in fields) and any("成交股數" in f for f in fields):
            rows = tbl.get("data", [])
            df = pd.DataFrame(rows, columns=fields)
            df = df.rename(columns={"證券代號": "stock_id", "成交股數": "volume"})
            df["stock_id"] = df["stock_id"].astype(str).str.strip()
            df["volume"] = (
                df["volume"].astype(str).str.replace(",", "", regex=False)
            )
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df["date"] = date_.isoformat()
            return df[["stock_id", "volume", "date"]].dropna()
    return pd.DataFrame()


def _fetch_tpex_daily_all(date_: dt.date) -> pd.DataFrame:
    """抓取指定日期上櫃全市場成交資訊（民國年格式）。"""
    roc_year = date_.year - 1911
    date_str = f"{roc_year}/{date_.month:02d}/{date_.day:02d}"
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    data = _get_json(url, {"date": date_str})
    rows = []
    if data:
        for item in data:
            code = item.get("SecuritiesCompanyCode") or item.get("CoID")
            vol = item.get("TradingShares") or item.get("Volume")
            if code and vol is not None:
                try:
                    rows.append(
                        {
                            "stock_id": str(code),
                            "volume": float(str(vol).replace(",", "")),
                            "date": date_.isoformat(),
                        }
                    )
                except ValueError:
                    continue
    return pd.DataFrame(rows)


def get_top_volume_stocks(
    stock_list: pd.DataFrame, top_n: int = None, lookback_days: int = None
) -> pd.DataFrame:
    """計算近一個月平均成交量，回傳前 top_n 大股票。"""
    top_n = top_n or config.TOP_N
    lookback_days = lookback_days or config.VOLUME_LOOKBACK_DAYS

    cache = _cache_path(f"top{top_n}_volume.csv")
    if _cache_is_fresh(cache):
        print(f"[快取] 讀取 {cache}")
        return pd.read_csv(cache, dtype={"stock_id": str})

    candidate_days = _trading_days_back(int(lookback_days * 1.6))  # 多抓一些天數以扣除假日
    twse_frames, tpex_frames = [], []
    collected = 0
    for d in candidate_days:
        if collected >= lookback_days:
            break
        twse_df = _fetch_twse_daily_all(d)
        tpex_df = _fetch_tpex_daily_all(d)
        if not twse_df.empty or not tpex_df.empty:
            collected += 1
            if not twse_df.empty:
                twse_frames.append(twse_df)
            if not tpex_df.empty:
                tpex_frames.append(tpex_df)
        print(f"  已取得 {d.isoformat()} 全市場成交資料 ({collected}/{lookback_days})")

    all_daily = pd.concat(twse_frames + tpex_frames, ignore_index=True) if (
        twse_frames or tpex_frames
    ) else pd.DataFrame(columns=["stock_id", "volume", "date"])

    if all_daily.empty:
        print("[錯誤] 無法取得任何每日成交資料，請檢查網路連線或 API 是否異動。")
        return pd.DataFrame()

    avg_vol = (
        all_daily.groupby("stock_id")["volume"].mean().reset_index()
        .rename(columns={"volume": "avg_volume"})
    )
    merged = avg_vol.merge(stock_list, on="stock_id", how="left")
    merged = merged.dropna(subset=["name"])
    top = merged.sort_values("avg_volume", ascending=False).head(top_n).reset_index(drop=True)
    top.to_csv(cache, index=False)
    return top


# ----------------------------------------------------------------------
# 3. 個股歷史 OHLCV（建模用）
# ----------------------------------------------------------------------
def _fetch_twse_stock_month(stock_id: str, year: int, month: int) -> pd.DataFrame:
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    params = {
        "date": f"{year}{month:02d}01",
        "stockNo": stock_id,
        "response": "json",
    }
    data = _get_json(url, params)
    if not data or data.get("stat") != "OK" or "data" not in data:
        return pd.DataFrame()
    fields = data.get("fields", [])
    rows = data.get("data", [])
    df = pd.DataFrame(rows, columns=fields)
    rename_map = {
        "日期": "date",
        "成交股數": "volume",
        "開盤價": "open",
        "最高價": "high",
        "最低價": "low",
        "收盤價": "close",
    }
    df = df.rename(columns=rename_map)
    needed = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]]

    def _roc_to_date(s):
        # 民國年/月/日 -> western date
        try:
            y, m, d = s.split("/")
            return dt.date(int(y) + 1911, int(m), int(d)).isoformat()
        except Exception:
            return None

    if "date" in df.columns:
        df["date"] = df["date"].apply(_roc_to_date)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"])


def _fetch_tpex_stock_month(stock_id: str, year: int, month: int) -> pd.DataFrame:
    """TPEx 個股歷史月資料（best-effort，格式可能隨官網調整）。目前未被呼叫，
    已改用下方 _fetch_stock_history_finmind()，此函式僅保留作參考。"""
    roc_year = year - 1911
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
    params = {
        "code": stock_id,
        "date": f"{roc_year}/{month:02d}",
        "id": "",
        "response": "json",
    }
    data = _get_json(url, params)
    if not data:
        return pd.DataFrame()
    rows = data.get("tables", [{}])[0].get("data", []) if "tables" in data else data.get("aaData", [])
    if not rows:
        return pd.DataFrame()
    cols = ["date", "volume", "amount", "open", "high", "low", "close", "change", "trades"]
    df = pd.DataFrame(rows)
    df = df.iloc[:, : len(cols)]
    df.columns = cols[: df.shape[1]]

    def _roc_to_date(s):
        try:
            y, m, d = str(s).split("/")
            return dt.date(int(y) + 1911, int(m), int(d)).isoformat()
        except Exception:
            return None

    df["date"] = df["date"].apply(_roc_to_date)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["date", "open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["date"])


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def _fetch_stock_history_finmind(stock_id: str, years: int) -> pd.DataFrame:
    """個股歷史日線改用 FinMind（dataset=TaiwanStockPrice），一次查詢整段日期區間即可涵蓋
    TWSE + TPEx（同一資料集不分市場）。原因見檔案開頭說明：
    TWSE 官方 STOCK_DAY 逐月高頻查詢易觸發反爬蟲限流；TPEx 官網無穩定 JSON 端點。"""
    end = dt.date.today()
    start = end - dt.timedelta(days=years * 365 + 30)
    headers = dict(HEADERS)
    if config.FINMIND_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.FINMIND_API_TOKEN}"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    if _finmind_quota_exhausted:
        return pd.DataFrame()

    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = requests.get(FINMIND_URL, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 402:
                _mark_finmind_quota_exhausted()
                return pd.DataFrame()
            resp.raise_for_status()
            time.sleep(config.FINMIND_REQUEST_SLEEP_SEC)
            payload = resp.json()
            rows = payload.get("data", [])
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
            keep = ["date", "open", "high", "low", "close", "volume"]
            return df[[c for c in keep if c in df.columns]].copy()
        except Exception as e:
            if _is_finmind_quota_error(e):
                _mark_finmind_quota_exhausted()
                return pd.DataFrame()
            print(f"  [警告] FinMind 第 {attempt} 次請求失敗: TaiwanStockPrice {stock_id} -> {e}")
            time.sleep(1.0 * attempt)
    print(f"  [錯誤] FinMind 放棄請求: TaiwanStockPrice {stock_id}")
    return pd.DataFrame()


def _fetch_twse_stock_history_legacy(stock_id: str, years: int) -> pd.DataFrame:
    """TWSE 官方 STOCK_DAY 逐月拼接，作為 FinMind 查無資料時的備援。"""
    today = dt.date.today()
    months = []
    y, m = today.year, today.month
    for _ in range(years * 12 + 1):
        months.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1

    frames = []
    for (yy, mm) in months:
        df = _fetch_twse_stock_month(stock_id, yy, mm)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_stock_history(stock_id: str, market: str, years: int = None) -> pd.DataFrame:
    """取得個股近 N 年歷史日線資料。
    優先透過 FinMind 一次查詢整段區間（低請求量、TWSE/TPEx 皆適用）；
    若 FinMind 查無資料，且為 TWSE 個股，才 fallback 到官方 STOCK_DAY 逐月查詢。
    """
    years = years or config.HISTORY_YEARS
    cache = _cache_path(f"hist_{stock_id}.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, parse_dates=["date"])
        cached_max_date = cached["date"].max().date() if not cached.empty else None
        if _content_date_is_fresh(cached_max_date, label=f"{stock_id} 日K歷史"):
            return cached

    full = _fetch_stock_history_finmind(stock_id, years)
    if full.empty and market == "TWSE":
        print(f"  [資訊] FinMind 查無 {stock_id} 資料，改用 TWSE 官方 STOCK_DAY 備援查詢...")
        full = _fetch_twse_stock_history_legacy(stock_id, years)

    if full.empty:
        return pd.DataFrame()

    full["date"] = pd.to_datetime(full["date"])
    full = full.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    full["stock_id"] = stock_id
    full.to_csv(cache, index=False)
    return full
