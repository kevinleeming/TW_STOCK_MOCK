"""
籌碼面資料抓取模組

資料來源與已知限制（請詳閱，這些是本模組能否正確運作的關鍵假設）：

1. TDCC 集保結算所「集保股權分散表」（大戶/散戶持股結構）
   來源A: https://opendata.tdcc.com.tw/getOD.ashx?id=1-5  (集保結算所開放資料，全市場最新一期)
   來源B: https://www.tdcc.com.tw/portal/zh/smWeb/qryStock  (官網「集保戶股權分散表查詢」頁面，
          可查「單一證券」近約1年、任一週的歷史快照，見下方 backfill_tdcc_history() 說明)
   - 來源A「每週更新一次」（通常每週五公告上週五為基準日的資料），且只提供「最新一期」
     全市場快照，不能查歷史。本模組用「快照累積」設計：每次執行時抓取當下最新一期，
     若該週尚未存進本地歷史檔 (cache/tdcc_weekly_history.csv)，就附加進去，逐週累積。
   - 來源B 可以「回補」過去數週資料，不必等每週執行才能累積出趨勢：這是傳統表單頁面
     （非公開 JSON API），有 CSRF 用的 SYNCHRONIZER_TOKEN，且「每次回應都會換發新
     token，下一次查詢要用上一次回應給的 token」（已用瀏覽器實測確認此串接方式可行）。
     實作於 `_tdcc_query_one()` / `backfill_tdcc_history()`：對候選股票清單逐檔、逐週送
     出查詢，把最近 `config.CHIP_TREND_WEEKS` 週的資料一次補齊，不需要真的等好幾週。
     `update_tdcc_history()` 在本地歷史週數不足時，會自動呼叫此回補流程。
   - 若持股分級的代碼/欄位名稱因官方格式調整而對不上，請調整 `_locate_column()` 比對
     規則、`BRACKET_*` 設定，或 `_tdcc_query_one()` 內的表格欄位解析。
   - 官方「持股分級」代碼實際上是 1~15（由小到大，15 = 「1,000,001股以上」的大戶級距），
     且原始資料集裡另外混了 2 個「保留未用（16）」「全市場合計（17）」的列，這兩列如果
     被誤當成一般分級加進加總，會讓大戶持股比例被系統性低估，`compute_big_holder_series()`
     已改為只加總代碼 1~15，排除 16/17。

2. 三大法人買賣超（外資/投信/自營商）
   TWSE: https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALL&response=json
         （全市場單日批次查詢，已實測可用）
   TPEx: TPEx 官方沒有可用的全市場批次開放 API（原本猜測的 tpex_3insti_trading_summary
         端點實測不存在，會回傳非 JSON 內容導致解析失敗）。改用 FinMind 免費 API
         （https://api.finmindtrade.com/api/v4/data，dataset=TaiwanStockInstitutionalInvestorsBuySell）
         逐檔查詢，此資料集同時涵蓋上市/上櫃/興櫃，不需分市場，且已確認免費、免註冊即可用
         （匿名每小時 300 次請求，註冊免費帳號可提高到 600 次/小時，見 config.FINMIND_API_TOKEN）。
   - 這是「每日」資料，可用歷史日期回溯查詢，故可精確算出近N個交易日的累計買賣超。

3. 融資融券餘額
   統一改用 FinMind dataset=TaiwanStockMarginPurchaseShortSale 逐檔查詢（TWSE/TPEx 皆適用）。
   原本 TWSE 端嘗試用 https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN 全市場批次查詢，
   但實測發現大量股票回傳「資料不足」，研判與 STOCK_DAY 屬同一批「新版rwd路徑猜測未經
   完整驗證」的端點，有相同的間歇性失敗風險，故直接統一改走已驗證可用的 FinMind，不再
   依賴 MI_MARGN（舊函式 `_fetch_twse_margin_day()` 保留在檔案中僅供參考）。
   - FinMind 這是「每日」資料，可回溯查詢，用來計算近N個交易日融資餘額的變化趨勢。

4. 「分點券商賣超」（主力券商進出）：使用者已確認以「三大法人 + 融資融券」近似，
   本模組不會嘗試抓取分點資料（這類資料通常僅存在於付費看盤軟體）。

由於本開發環境的網路存取權限被封鎖，以上 API 皆無法在此現場驗證，程式碼以官方文件與
社群慣用的欄位格式撰寫，並全面採用「防禦性解析」（用欄位名稱關鍵字比對，而非死板的
欄位順序），實際使用時如遇解析失敗，請對照上述網址確認回傳格式是否變動。
"""
import os
import time
import datetime as dt
from typing import Optional

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

import config
import data_fetch

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _get_json(url: str, params: dict = None):
    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            time.sleep(config.REQUEST_SLEEP_SEC)
            return resp.json()
        except Exception as e:
            print(f"  [警告] 第 {attempt} 次請求失敗: {url} params={params} -> {e}")
            time.sleep(1.0 * attempt)
    print(f"  [錯誤] 放棄請求: {url} params={params}")
    return None


def _get_csv(url: str, params: dict = None) -> Optional[pd.DataFrame]:
    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            time.sleep(config.REQUEST_SLEEP_SEC)
            from io import StringIO
            return pd.read_csv(StringIO(resp.content.decode("utf-8-sig")))
        except Exception as e:
            print(f"  [警告] 第 {attempt} 次請求失敗: {url} -> {e}")
            time.sleep(1.0 * attempt)
    return None


def _cache_path(name: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, name)


def _cache_is_fresh(path: str) -> bool:
    """[已不再使用，保留供參考] 只看檔案 mtime 判斷是否「今天寫入過」。
    這種判斷方式有個 bug：如果當天第一次執行時剛好卡在資料公布前（例如收盤後
    三大法人/融資資料還沒上架），會抓到前一交易日的資料並寫入快取，但 mtime
    仍是「今天」，導致當天之後重新執行也不會再嘗試更新。三大法人/融資餘額的
    快取已改用 data_fetch._content_date_is_fresh() 比對「內容日期」而非檔案
    mtime（見 get_institution_trend / get_margin_trend），才能在資料真正公布
    後，重新執行時抓到當天資料。"""
    if not os.path.exists(path):
        return False
    mtime_date = dt.date.fromtimestamp(os.path.getmtime(path))
    return mtime_date == dt.date.today()


def _locate_column(columns, must_contain_all):
    """依關鍵字比對找出欄位名稱（防禦性解析，容忍官方欄位命名微調）。"""
    for col in columns:
        if all(k in str(col) for k in must_contain_all):
            return col
    return None


# ----------------------------------------------------------------------
# 1. TDCC 集保股權分散表（大戶 / 股東人數）
# ----------------------------------------------------------------------
TDCC_URL = "https://opendata.tdcc.com.tw/getOD.ashx"
TDCC_DATASET_ID = "1-5"  # 集保股權分散表，若失效請至 https://opendata.tdcc.com.tw/ 確認最新 id

HISTORY_FILE = "tdcc_weekly_history.csv"


def fetch_tdcc_latest_snapshot() -> pd.DataFrame:
    """抓取 TDCC 集保股權分散表「最新一期」全市場快照。
    回傳欄位: date, stock_id, bracket, holders, shares, pct
    """
    df = _get_csv(TDCC_URL, {"id": TDCC_DATASET_ID})
    if df is None or df.empty:
        print("[錯誤] 無法取得 TDCC 集保股權分散表，請確認網址 / 資料集 id 是否仍有效。")
        return pd.DataFrame()

    col_date = _locate_column(df.columns, ["資料日期"]) or _locate_column(df.columns, ["日期"])
    col_id = _locate_column(df.columns, ["證券代號"]) or _locate_column(df.columns, ["股票代號"])
    col_bracket = _locate_column(df.columns, ["持股", "分級"]) or _locate_column(df.columns, ["級距"])
    col_holders = _locate_column(df.columns, ["人數"])
    col_shares = _locate_column(df.columns, ["股數"])
    col_pct = _locate_column(df.columns, ["比例"]) or _locate_column(df.columns, ["占"])

    required = [col_date, col_id, col_bracket, col_holders, col_shares]
    if any(c is None for c in required):
        print(f"[錯誤] TDCC 資料欄位無法辨識，實際欄位: {list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame({
        "date": df[col_date].astype(str),
        "stock_id": df[col_id].astype(str).str.strip(),
        "bracket": df[col_bracket].astype(str).str.strip(),
        "holders": pd.to_numeric(df[col_holders], errors="coerce"),
        "shares": pd.to_numeric(df[col_shares], errors="coerce"),
    })
    if col_pct:
        out["pct"] = pd.to_numeric(df[col_pct], errors="coerce")
    return out.dropna(subset=["stock_id", "bracket"])


# ----------------------------------------------------------------------
# 1b. TDCC 官網「集保戶股權分散表查詢」頁面 (qryStock)：可查單一證券近約1年的歷史
#     週快照，用來「回補」過去數週資料，不必每週執行才能累積出趨勢。
#     這是傳統表單頁面，非公開 JSON API，細節見上方模組說明。
# ----------------------------------------------------------------------
TDCC_QRYSTOCK_URL = "https://www.tdcc.com.tw/portal/zh/smWeb/qryStock"


def _tdcc_input_value(form_tag, name):
    tag = form_tag.find("input", {"name": name})
    return tag.get("value") if tag else None


def _tdcc_parse_form_state(html: str) -> dict:
    """從 qryStock 頁面（不論是第一次 GET 或是查詢結果頁）解析出下一次送出查詢所需的
    表單欄位（含會變動的 CSRF token），以及可查詢的歷史日期清單（新到舊排序）。"""
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form")
    if form is None:
        return {}
    state = {
        "SYNCHRONIZER_TOKEN": _tdcc_input_value(form, "SYNCHRONIZER_TOKEN"),
        "SYNCHRONIZER_URI": _tdcc_input_value(form, "SYNCHRONIZER_URI") or "/portal/zh/smWeb/qryStock",
        "method": _tdcc_input_value(form, "method") or "submit",
        "firDate": _tdcc_input_value(form, "firDate"),
    }
    date_select = form.find("select", {"name": "scaDate"})
    if date_select:
        dates = [opt.get("value") or opt.get_text(strip=True) for opt in date_select.find_all("option")]
        state["available_dates"] = [d for d in dates if d]
    return state


def _tdcc_query_one(session: requests.Session, form_state: dict, stock_id: str, date_str: str):
    """對 qryStock 送出一次「單一股票 + 單一歷史週別」查詢。
    回傳 (下一輪查詢要用的 form_state, DataFrame[bracket, holders, shares, pct] 或 None)。
    None 代表查無資料（該股票在該週可能尚未上市/上櫃，或格式解析失敗）。
    """
    payload = {
        "SYNCHRONIZER_TOKEN": form_state.get("SYNCHRONIZER_TOKEN"),
        "SYNCHRONIZER_URI": form_state.get("SYNCHRONIZER_URI"),
        "method": form_state.get("method", "submit"),
        "firDate": form_state.get("firDate"),
        "scaDate": date_str,
        "sqlMethod": "StockNo",
        "stockNo": stock_id,
        "stockName": "",
    }
    try:
        resp = session.post(TDCC_QRYSTOCK_URL, data=payload, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [警告] TDCC歷史查詢請求失敗: {stock_id} {date_str} -> {e}")
        return form_state, None

    new_state = _tdcc_parse_form_state(resp.text)
    new_state.setdefault("available_dates", form_state.get("available_dates", []))

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "table"})
    if table is None:
        return new_state, None

    data_rows = []
    for tr in table.find_all("tr")[1:]:  # 跳過表頭列
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 5:
            continue
        idx, _bracket_label, holders, shares, pct = cells[:5]
        if not idx.isdigit() or int(idx) > 15:
            continue  # 跳過「合計」列（序號16）與其他非分級列
        try:
            data_rows.append({
                "bracket": idx,
                "holders": float(holders.replace(",", "")),
                "shares": float(shares.replace(",", "")),
                "pct": float(pct.replace(",", "")),
            })
        except ValueError:
            continue

    if not data_rows:
        return new_state, None
    return new_state, pd.DataFrame(data_rows)


def backfill_tdcc_history(stock_ids, weeks: int = None) -> pd.DataFrame:
    """透過 TDCC 官網歷史查詢頁面，一次回補多檔股票最近幾週的股權分散資料，
    取代「必須每週執行才能累積趨勢」的等待。詳細技術說明見模組開頭註解。

    stock_ids: 要回補的股票代號清單（通常是本輪選股候選池，不會是全市場，以控制
               請求量：股票數 x 週數 大約就是總請求次數）。
    weeks: 要回補最近幾週（預設 config.CHIP_TREND_WEEKS）。
    回傳: DataFrame[date, stock_id, bracket, holders, shares, pct]，
          呼叫端負責與本地歷史合併去重後寫檔（見 update_tdcc_history()）。
    """
    weeks = weeks or config.CHIP_TREND_WEEKS
    session = requests.Session()
    try:
        resp = session.get(TDCC_QRYSTOCK_URL, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"[錯誤] 無法連線 TDCC 集保戶股權分散表查詢頁面: {e}")
        return pd.DataFrame()

    state = _tdcc_parse_form_state(resp.text)
    dates = (state.get("available_dates") or [])[:weeks]
    if not dates:
        print("[錯誤] 無法解析 TDCC 查詢頁面可用日期清單，頁面格式可能已變動，略過歷史回補。")
        return pd.DataFrame()

    stock_ids = list(stock_ids)
    total = len(stock_ids) * len(dates)
    done, ok = 0, 0
    rows = []
    for sid in stock_ids:
        for date_str in dates:
            done += 1
            state, df = _tdcc_query_one(session, state, sid, date_str)
            time.sleep(config.REQUEST_SLEEP_SEC)
            if df is not None and not df.empty:
                df = df.copy()
                df["date"] = date_str
                df["stock_id"] = sid
                rows.append(df[["date", "stock_id", "bracket", "holders", "shares", "pct"]])
                ok += 1
            if done % 40 == 0:
                print(f"  TDCC歷史回補 進度 {done}/{total}（成功 {ok} 筆）")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def update_tdcc_history(backfill_stock_ids=None) -> pd.DataFrame:
    """抓取最新快照並附加進本地歷史檔（若該週尚未記錄過）。
    若本地累積週數還不足 config.CHIP_TREND_WEEKS，且有提供 backfill_stock_ids，
    會另外呼叫 backfill_tdcc_history() 向 TDCC 歷史查詢頁面回補最近幾週資料，
    不必真的每週執行好幾次才能算出大戶/股東人數趨勢。
    回傳目前累積的完整歷史。
    """
    hist_path = _cache_path(HISTORY_FILE)
    history = pd.read_csv(hist_path, dtype=str) if os.path.exists(hist_path) else pd.DataFrame()

    snapshot = fetch_tdcc_latest_snapshot()
    if not snapshot.empty:
        snap_date = snapshot["date"].iloc[0]
        if not history.empty and snap_date in set(history["date"].astype(str)):
            print(f"[TDCC] {snap_date} 這週的快照已經記錄過，略過重複寫入。")
        else:
            history = pd.concat([history, snapshot.astype(str)], ignore_index=True) if not history.empty else snapshot.astype(str)
            history.to_csv(hist_path, index=False)
            print(f"[TDCC] 已新增 {snap_date} 週快照，目前本地歷史共 {history['date'].nunique()} 週。")

    existing_weeks = history["date"].nunique() if not history.empty else 0
    if backfill_stock_ids and existing_weeks < config.CHIP_TREND_WEEKS:
        print(f"[TDCC] 本地歷史僅 {existing_weeks} 週（< {config.CHIP_TREND_WEEKS}），"
              f"改用官網歷史查詢頁面回補候選股票最近 {config.CHIP_TREND_WEEKS} 週資料...")
        backfilled = backfill_tdcc_history(backfill_stock_ids, weeks=config.CHIP_TREND_WEEKS)
        if not backfilled.empty:
            if not history.empty:
                existing_keys = set(zip(history["date"].astype(str), history["stock_id"].astype(str)))
                mask_new = ~backfilled.apply(lambda r: (str(r["date"]), str(r["stock_id"])) in existing_keys, axis=1)
                backfilled = backfilled[mask_new]
            if not backfilled.empty:
                history = pd.concat([history, backfilled.astype(str)], ignore_index=True) if not history.empty else backfilled.astype(str)
                history.to_csv(hist_path, index=False)
                print(f"[TDCC] 官網歷史回補完成，新增 {len(backfilled)} 筆快照資料，"
                      f"本地歷史現共 {history['date'].nunique()} 週。")
        else:
            print("[TDCC] 官網歷史回補沒有取得任何資料，大戶/股東人數趨勢暫時仍會是中性分數。")

    return history


def _bracket_is_at_least(bracket: str, min_level_keywords) -> bool:
    """判斷某個持股分級字串是否達到「大戶」門檻。
    優先嘗試把 bracket 解析成數字代碼（1~15，數字越大代表級距越高）；
    若不是純數字，改用關鍵字（如帶有 '1,000,001' 或百萬股以上的字樣）比對。
    """
    b = bracket.strip()
    if b.isdigit():
        return int(b) >= int(min_level_keywords["min_code"])
    return any(k in b for k in min_level_keywords["keywords"])


BIG_HOLDER_1000_RULE = {"min_code": config.BIG_HOLDER_BRACKET_1000, "keywords": ["1,000,001", "1000001", "1,000,000以上", "1000張以上"]}
BIG_HOLDER_100_RULE = {"min_code": config.BIG_HOLDER_BRACKET_100, "keywords": ["100,001", "100張以上", "100,000以上"]}


def compute_big_holder_series(history: pd.DataFrame, stock_id: str, price: float) -> pd.DataFrame:
    """從本地累積的 TDCC 歷史中，算出指定股票「每週」的大戶持股比例(%) 與總股東人數。
    price >= BIG_HOLDER_PRICE_THRESHOLD 時使用「百張大戶」門檻，否則用「千張大戶」門檻。
    回傳: DataFrame[date, big_holder_pct, total_holders]，依日期排序。
    """
    if history.empty:
        return pd.DataFrame()
    sub = history[history["stock_id"] == str(stock_id)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["holders"] = pd.to_numeric(sub["holders"], errors="coerce")
    sub["shares"] = pd.to_numeric(sub["shares"], errors="coerce")

    # 只保留真正的持股分級（代碼 1~15）。TDCC 開放資料集裡另外混了「保留未用(16)」
    # 「全市場合計(17)」兩個特殊代碼，合計列的股數/人數等於 1~15 加總，若沒排除會
    # 被重複計入，讓 total_shares/total_holders 被灌水約2倍，使大戶持股比例被低估。
    def _is_real_bracket(b):
        s = str(b).strip()
        return s.isdigit() and 1 <= int(s) <= 15
    sub = sub[sub["bracket"].apply(_is_real_bracket)]
    if sub.empty:
        return pd.DataFrame()

    rule = BIG_HOLDER_100_RULE if price >= config.BIG_HOLDER_PRICE_THRESHOLD else BIG_HOLDER_1000_RULE

    rows = []
    for date_, g in sub.groupby("date"):
        total_shares = g["shares"].sum()
        total_holders = g["holders"].sum()
        big_mask = g["bracket"].apply(lambda b: _bracket_is_at_least(b, rule))
        big_shares = g.loc[big_mask, "shares"].sum()
        big_pct = (big_shares / total_shares * 100) if total_shares else np.nan
        rows.append({"date": date_, "big_holder_pct": big_pct, "total_holders": total_holders})

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# FinMind 免費 API（用於 TPEx 個股三大法人買賣超 / 融資融券，見檔頭說明）
# 文件: https://finmind.github.io/tutor/TaiwanMarket/Chip/
# 免費、免註冊即可用（匿名每小時 300 次請求）；若請求量較大，可至
# https://finmindtrade.com 免費註冊帳號取得 token 填入 config.py 的 FINMIND_API_TOKEN，
# 提高到每小時 600 次請求上限（data_fetch.py 的個股歷史日K也共用同一組設定）。
# ----------------------------------------------------------------------
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def _finmind_get(dataset: str, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    if data_fetch._finmind_quota_exhausted:
        return pd.DataFrame()

    headers = {}
    if config.FINMIND_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.FINMIND_API_TOKEN}"
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date, "end_date": end_date}

    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = requests.get(FINMIND_URL, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 402:
                data_fetch._mark_finmind_quota_exhausted()
                return pd.DataFrame()
            resp.raise_for_status()
            time.sleep(config.FINMIND_REQUEST_SLEEP_SEC)
            payload = resp.json()
            return pd.DataFrame(payload.get("data", []))
        except Exception as e:
            if data_fetch._is_finmind_quota_error(e):
                data_fetch._mark_finmind_quota_exhausted()
                return pd.DataFrame()
            print(f"  [警告] FinMind 第 {attempt} 次請求失敗: {dataset} {stock_id} -> {e}")
            time.sleep(1.0 * attempt)
    print(f"  [錯誤] FinMind 放棄請求: {dataset} {stock_id}")
    return pd.DataFrame()


def fetch_finmind_institution_batch(stock_ids, start_date: str, end_date: str) -> pd.DataFrame:
    """逐檔查詢 FinMind 三大法人買賣超，彙總成 (stock_id, date, net_buy)。"""
    rows = []
    for i, sid in enumerate(stock_ids, start=1):
        if data_fetch._finmind_quota_exhausted:
            print(f"  [資訊] FinMind 額度已用完，三大法人查詢在 {i}/{len(stock_ids)} 檔提前中止。")
            break
        df = _finmind_get("TaiwanStockInstitutionalInvestorsBuySell", sid, start_date, end_date)
        if not df.empty and {"date", "buy", "sell"}.issubset(df.columns):
            agg = df.groupby("date").apply(lambda g: (g["buy"] - g["sell"]).sum())
            for date_, net in agg.items():
                rows.append({"stock_id": sid, "date": date_, "net_buy": float(net)})
        if i % 20 == 0:
            print(f"  FinMind 三大法人買賣超(上櫃) 進度 {i}/{len(stock_ids)}")
    return pd.DataFrame(rows)


def fetch_finmind_margin_batch(stock_ids, start_date: str, end_date: str) -> pd.DataFrame:
    """逐檔查詢 FinMind 融資融券，取出 (stock_id, date, margin_balance)。"""
    rows = []
    for i, sid in enumerate(stock_ids, start=1):
        if data_fetch._finmind_quota_exhausted:
            print(f"  [資訊] FinMind 額度已用完，融資融券查詢在 {i}/{len(stock_ids)} 檔提前中止。")
            break
        df = _finmind_get("TaiwanStockMarginPurchaseShortSale", sid, start_date, end_date)
        if not df.empty and "MarginPurchaseTodayBalance" in df.columns:
            for _, r in df.iterrows():
                rows.append({
                    "stock_id": sid,
                    "date": r["date"],
                    "margin_balance": float(r["MarginPurchaseTodayBalance"]),
                })
        if i % 20 == 0:
            print(f"  FinMind 融資餘額 進度 {i}/{len(stock_ids)}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. 三大法人買賣超（TWSE T86 全市場批次 / TPEx 見上方 FinMind）
# ----------------------------------------------------------------------
def _fetch_twse_institution_day(date_: dt.date) -> pd.DataFrame:
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    data = _get_json(url, {"date": date_.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
    if not data or data.get("stat") != "OK" or "data" not in data:
        return pd.DataFrame()
    fields = data.get("fields", [])
    df = pd.DataFrame(data["data"], columns=fields)

    col_id = _locate_column(fields, ["證券代號"])
    col_net = _locate_column(fields, ["三大法人", "買賣超"])
    if col_id is None or col_net is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "stock_id": df[col_id].astype(str).str.strip(),
        "net_buy": df[col_net].astype(str).str.replace(",", "", regex=False),
    })
    out["net_buy"] = pd.to_numeric(out["net_buy"], errors="coerce")
    out["date"] = date_.isoformat()
    return out.dropna(subset=["net_buy"])


def get_institution_trend(days: int = None, tpex_stock_ids=None) -> pd.DataFrame:
    """取得近 N 個交易日的三大法人買賣超，回傳逐日逐股資料（未彙總）。
    上市(TWSE)用全市場單日批次查詢；上櫃(TPEx)因官方無可用批次API，
    改用 FinMind 對 tpex_stock_ids 清單逐檔查詢（見檔頭說明）。
    """
    days = days or config.INSTITUTION_LOOKBACK_DAYS
    cache = _cache_path(f"institution_{days}d.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, dtype={"stock_id": str})
        cached_max_date = cached["date"].max() if not cached.empty and "date" in cached.columns else None
        if data_fetch._content_date_is_fresh(cached_max_date, label="三大法人買賣超"):
            return cached

    candidate_days = []
    d = dt.date.today()
    while len(candidate_days) < int(days * 1.6):
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            candidate_days.append(d)

    frames = []
    collected = 0
    for day in candidate_days:
        if collected >= days:
            break
        twse_df = _fetch_twse_institution_day(day)
        if not twse_df.empty:
            frames.append(twse_df)
            collected += 1
        print(f"  三大法人買賣超(上市) {day.isoformat()} ({collected}/{days})")

    if tpex_stock_ids:
        start_date = candidate_days[-1].isoformat() if candidate_days else None
        end_date = dt.date.today().isoformat()
        tpex_df = fetch_finmind_institution_batch(tpex_stock_ids, start_date, end_date)
        if not tpex_df.empty:
            frames.append(tpex_df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def get_today_institution_flow(target_date: str, tpex_stock_ids=None) -> pd.DataFrame:
    """取得「單一交易日」（通常是最新收盤日）的三大法人買賣超，用於產業/題材熱力圖
    （熱力圖要看的是「當天」籌碼流向，跟 get_institution_trend 的近N日趨勢用途不同，
    分開實作、分開快取，避免互相干擾）。
    上市(TWSE)：T86 一次請求涵蓋全市場，免費、不太消耗 FinMind 額度。
    上櫃(TPEx)：官方無批次API，仍須對 tpex_stock_ids 逐檔查 FinMind（見檔頭說明），
    這是本工具唯一會為了「熱力圖」額外消耗 FinMind 額度的地方，故呼叫端應只傳入
    候選池（例如 SECTOR_CANDIDATE_TOP_VOLUME 檔），不要傳全市場。
    回傳 DataFrame[stock_id, date, net_buy]（net_buy 單位：股）。
    """
    cache = _cache_path(f"institution_today_{target_date}.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, dtype={"stock_id": str})
        if not cached.empty:
            return cached

    try:
        target = dt.date.fromisoformat(target_date)
    except ValueError:
        return pd.DataFrame()

    frames = []
    twse_df = _fetch_twse_institution_day(target)
    if not twse_df.empty:
        frames.append(twse_df)

    if tpex_stock_ids:
        tpex_df = fetch_finmind_institution_batch(list(tpex_stock_ids), target_date, target_date)
        if not tpex_df.empty:
            frames.append(tpex_df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def summarize_institution_net(institution_df: pd.DataFrame, stock_id: str) -> dict:
    """加總指定股票近期三大法人買賣超（單位：股）。"""
    if institution_df.empty:
        return {"net_buy_sum": np.nan, "n_days": 0}
    sub = institution_df[institution_df["stock_id"] == str(stock_id)]
    if sub.empty:
        return {"net_buy_sum": np.nan, "n_days": 0}
    return {"net_buy_sum": float(sub["net_buy"].sum()), "n_days": int(sub["date"].nunique())}


# ----------------------------------------------------------------------
# 3. 融資融券餘額
# 原本 TWSE 端用 MI_MARGN 全市場批次查詢（/rwd/zh/marginTrading/MI_MARGN），但實測發現
# 大量股票回傳「融資餘額資料不足」，研判與 STOCK_DAY 是同一批「新版rwd路徑猜測未經完整
# 驗證」的端點，有相同的間歇性失敗風險。故改為統一走 FinMind（TaiwanStockMarginPurchase
# ShortSale，同時涵蓋 TWSE/TPEx），逐檔一次查詢整段區間，不再依賴 MI_MARGN 批次查詢。
# 舊的 _fetch_twse_margin_day() 保留作參考備援，預設不會被呼叫。
# ----------------------------------------------------------------------
def _fetch_twse_margin_day(date_: dt.date) -> pd.DataFrame:
    """TWSE MI_MARGN 全市場單日批次查詢（備援參考用，目前未被預設呼叫，見上方說明）。"""
    url = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
    data = _get_json(url, {"date": date_.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
    if not data or data.get("stat") != "OK" or "data" not in data:
        return pd.DataFrame()
    fields = data.get("fields", [])
    df = pd.DataFrame(data["data"], columns=fields)

    col_id = _locate_column(fields, ["股票代號"]) or _locate_column(fields, ["證券代號"])
    col_balance = _locate_column(fields, ["融資", "今日餘額"])
    if col_id is None or col_balance is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "stock_id": df[col_id].astype(str).str.strip(),
        "margin_balance": df[col_balance].astype(str).str.replace(",", "", regex=False),
    })
    out["margin_balance"] = pd.to_numeric(out["margin_balance"], errors="coerce")
    out["date"] = date_.isoformat()
    return out.dropna(subset=["margin_balance"])


def get_margin_trend(stock_ids=None, days: int = None, tpex_stock_ids=None) -> pd.DataFrame:
    """取得近 N 個交易日的融資餘額，回傳逐日逐股資料（未彙總）。
    統一透過 FinMind 逐檔查詢整段區間（TWSE/TPEx 皆適用，見上方說明）。
    stock_ids: 全部候選股票代號清單（建議傳入，TWSE+TPEx 皆可）。
    tpex_stock_ids: 相容舊呼叫方式保留的參數，若沒傳 stock_ids 則退回只查這份清單。
    """
    days = days or config.MARGIN_LOOKBACK_DAYS
    all_ids = list(stock_ids) if stock_ids else list(tpex_stock_ids or [])
    cache = _cache_path(f"margin_{days}d.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, dtype={"stock_id": str})
        cached_max_date = cached["date"].max() if not cached.empty and "date" in cached.columns else None
        if data_fetch._content_date_is_fresh(cached_max_date, label="融資餘額"):
            return cached

    if not all_ids:
        return pd.DataFrame()

    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=int(days * 1.6))).isoformat()
    out = fetch_finmind_margin_batch(all_ids, start_date, end_date)
    if out.empty:
        return pd.DataFrame()
    out.to_csv(cache, index=False)
    return out


def summarize_margin_change(margin_df: pd.DataFrame, stock_id: str) -> dict:
    """計算指定股票近期融資餘額變化（今日 vs 期初，百分比變化為負代表下降=良好訊號）。"""
    if margin_df.empty:
        return {"change_pct": np.nan, "n_days": 0}
    sub = margin_df[margin_df["stock_id"] == str(stock_id)].sort_values("date")
    if len(sub) < 2:
        return {"change_pct": np.nan, "n_days": len(sub)}
    first, last = sub["margin_balance"].iloc[0], sub["margin_balance"].iloc[-1]
    if not first:
        return {"change_pct": np.nan, "n_days": len(sub)}
    return {"change_pct": float((last - first) / first * 100), "n_days": len(sub)}
