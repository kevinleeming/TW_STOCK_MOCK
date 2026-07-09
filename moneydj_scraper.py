"""
MoneyDJ 產業分類 爬蟲 —— 用來建立「題材對照表」（theme_map.py 的自動化升級版）。

為什麼要做這個爬蟲：
FinMind 官方產業別只有粗略的大分類（如「半導體業」），MoneyDJ理財網
(https://www.moneydj.com/Z/ZH/ZHA/ZHA.djhtm) 有一份公開、免費瀏覽的產業分類，
共兩層：
  - 大分類（實測121個，例如 IC設計、面板業、被動元件、網通設備...）
  - 細分類（實測1062個，例如 IC設計底下有 網路卡IC、智慧卡IC、CPU、MCU...）
頁面的DOM結構是：每個大分類是<table class="t01">裡的一列<tr>，該列有一個
「直屬」連結(大分類本身)，以及一個巢狀<table>裝著這個大分類底下所有細分類的連結
（巢狀table的第一個連結會重複大分類自己一次，代表「未進一步歸類」的部分）。
每個細分類頁面(zh00.djhtm?a=代碼)都列出該分類內的個股清單。

這是目前找到唯一免費、且結構化程度夠高、可以拿來建立題材對照表的公開資料源。

使用限制與注意事項（很重要，請先讀過再用）：
1. 這是「非官方」爬蟲，MoneyDJ 並未提供公開 API，頁面格式未來可能調整。如果忽然
   抓不到資料，第一步請檢查 fetch_taxonomy_hierarchy() 的連結/巢狀table格式、
   fetch_category_stocks() 的 table.t01 是否仍對得上（可用瀏覽器開網址人工核對）。
2. 網頁編碼是 Big5，務必用 resp.content.decode("big5", errors="replace")，不要
   直接用 resp.text（requests 用預設方式猜編碼常常會猜成亂碼）。
3. 細分類數量超過1000個，逐一請求需要相當時間（預設每次請求間隔1秒，全部跑完約
   20~30分鐘），而且是對別人的網站增加流量負擔，故刻意設計成「獨立、非每日執行」
   的更新腳本（見 build_theme_map.py），不建議放進每天晚上的 sector_main.py 流程。
   股票的分類歸屬本身變動很慢（新股上市、公司轉型才會變），一週~一個月更新一次即可。
4. 本爬蟲只抓「分類 -> 個股清單」的歸屬關係，不使用 MoneyDJ 頁面上顯示的股價/
   漲跌幅數字，避免跟本專案主要的 TWSE/TPEx/FinMind 資料來源打架、造成兩套價格
   對不上——股價/漲跌幅仍統一由 data_fetch.get_price_change_snapshot() 提供。
5. 「資金流向」另外走 fetch_fund_flow()，資料源是 MoneyDJ 的
   https://www.moneydj.com/z/zb/zba/zba.djhtm（大盤資金流向表），這是官方公布
   的「上市/上櫃 各類股資金流向率」，用的是 TWSE/TPEx 傳統的粗分類（約30幾類，
   跟本檔案的「大分類」不是同一套系統），故用 ZHA_TO_OFFICIAL_GROUP 手動對照表
   把大分類對應到最接近的資金流向表分類，此對照是人工判斷、非精確科學對應，
   歡迎依實際狀況調整。
"""
import os
import re
import time
import json

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

TAXONOMY_URL = "https://www.moneydj.com/Z/ZH/ZHA/ZHA.djhtm"
CATEGORY_URL = "https://www.moneydj.com/z/zh/zha/zh00.djhtm"
FUND_FLOW_URL = "https://www.moneydj.com/z/zb/zba/zba.djhtm"

THEME_MAP_CACHE = os.path.join(config.CACHE_DIR, "moneydj_theme_map.json")
FUND_FLOW_CACHE = os.path.join(config.CACHE_DIR, "moneydj_fund_flow.json")


def _get_big5_soup(url: str, params: dict = None) -> BeautifulSoup:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    html = resp.content.decode("big5", errors="replace")
    return BeautifulSoup(html, "html.parser")


# ----------------------------------------------------------------------
# 1. 產業分類階層（大分類 -> 細分類）
# ----------------------------------------------------------------------
def _code_of(a_tag) -> str:
    href = a_tag.get("href", "")
    m = re.search(r"a=(C\d+)", href)
    return m.group(1) if m else None


def fetch_taxonomy_hierarchy() -> list:
    """回傳大分類階層清單：
    [{"major_name":..., "major_code":..., "children":[{"name":..., "code":...}, ...]}, ...]
    每個大分類的 children 一定包含大分類自己（代表「未進一步歸類」的細項），
    實測共121個大分類、1062個細分類（含重複的大分類自身）。
    """
    soup = _get_big5_soup(TAXONOMY_URL)
    table = soup.find("table", class_="t01")
    if table is None:
        return []

    result = []
    for tr in table.find_all("tr"):
        nested = tr.find("table")
        if nested is None:
            continue
        # 「直屬」連結 = 不在巢狀table裡的連結（就是這一列代表的大分類本身）
        nested_links = set(id(a) for a in nested.find_all("a"))
        direct_links = [a for a in tr.find_all("a") if id(a) not in nested_links]
        if not direct_links:
            continue
        major_a = direct_links[0]
        children = [
            {"name": a.get_text(strip=True), "code": _code_of(a)}
            for a in nested.find_all("a")
            if _code_of(a)
        ]
        if children:
            result.append({
                "major_name": major_a.get_text(strip=True),
                "major_code": _code_of(major_a),
                "children": children,
            })
    return result


def fetch_category_stocks(code: str) -> list:
    """回傳指定分類代碼底下的個股清單 [{"stock_id":..., "name":...}, ...]（只取歸屬關係，不取股價）。
    頁面結構：股票清單在 <table class="t01">，每列第一個 td 是「代號+名稱」黏在一起
    （例如「1560中砂」），用正規表示式拆開。
    """
    soup = _get_big5_soup(CATEGORY_URL, params={"a": code})
    table = soup.find("table", class_="t01")
    if table is None:
        return []
    stocks = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        first_cell = tds[0].get_text(strip=True)
        m = re.match(r"(\d{4,6})(.+)", first_cell)
        if not m:
            continue
        stocks.append({"stock_id": m.group(1), "name": m.group(2)})
    return stocks


def build_theme_map(sleep_sec: float = 1.0, limit: int = None, progress_every: int = 20) -> dict:
    """抓取全部（或前 limit 個大分類，方便測試）的細分類個股清單，依大分類彙總。

    回傳格式：{大分類名稱: {stock_id: [細分類名稱, ...]}}
    ——一檔股票在同一個大分類底下可能同時屬於多個細分類（例如某公司同時做
    DRAM記憶體IC 又做 FLASH記憶體IC，兩者都在IC設計底下），故用list保留全部。
    這個結構同時滿足兩個需求：(1) 大分類本身的股票清單(dict的key做union即可)
    (2) 每檔股票在該大分類下所屬的細分類標籤(給題材詳細頁顯示用)。
    """
    hierarchy = fetch_taxonomy_hierarchy()
    if limit:
        hierarchy = hierarchy[:limit]

    total_children = sum(len(m["children"]) for m in hierarchy)
    result = {}
    done = 0
    for major in hierarchy:
        major_name = major["major_name"]
        bucket = result.setdefault(major_name, {})
        for child in major["children"]:
            done += 1
            try:
                stocks = fetch_category_stocks(child["code"])
                for s in stocks:
                    sid = s["stock_id"]
                    labels = bucket.setdefault(sid, [])
                    if child["name"] not in labels:
                        labels.append(child["name"])
            except Exception as e:
                print(f"  [警告] 細分類 {child['name']}({child['code']}) 抓取失敗: {e}")
            if done % progress_every == 0:
                print(f"  進度 {done}/{total_children}")
            time.sleep(sleep_sec)
    return result


def save_theme_map(theme_map_data: dict, path: str = None):
    path = path or THEME_MAP_CACHE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(theme_map_data, f, ensure_ascii=False, indent=2)


def load_theme_map(path: str = None) -> dict:
    """讀取已抓好的完整題材對照表；還沒執行過 build_theme_map.py 的話回傳空dict
    （呼叫端會自動退回使用 theme_map.py 手動整理的起點清單，見 theme_map.py 說明）。
    格式：{大分類名稱: {stock_id: [細分類名稱, ...]}}"""
    path = path or THEME_MAP_CACHE
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 2. 資金流向（官方 上市/上櫃 類股資金流向率）
# ----------------------------------------------------------------------
def fetch_fund_flow() -> dict:
    """抓取 MoneyDJ「大盤資金流向表」，回傳當日 上市/上櫃 各類股資金流向率。
    回傳格式：{"TWSE": {類股名稱: 流向率(%)}, "TPEx": {類股名稱: 流向率(%)}}

    頁面結構（實測確認）：單一 <table class="t01">，列的排列方式是：
      第1列：「上市資金流向表」標題(單一儲存格)
      第2列：欄位標題「類股名稱｜流向率｜...」
      接下來數列：每列4組「類股名稱｜流向率」pair
      再來一列：「上櫃資金流向表」標題 —— 之後的資料列都屬於上櫃
      再來一列：欄位標題
      接下來數列：上櫃的4組pair資料列
    用「這一列是不是只有1個td」來偵測標題列，切換目前屬於上市或上櫃。
    """
    soup = _get_big5_soup(FUND_FLOW_URL)
    table = soup.find("table", class_="t01")
    if table is None:
        return {"TWSE": {}, "TPEx": {}}

    result = {"TWSE": {}, "TPEx": {}}
    current_market = None
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        if len(tds) == 1:
            text = tds[0].get_text(strip=True)
            if "上市資金流向表" in text:
                current_market = "TWSE"
            elif "上櫃資金流向表" in text:
                current_market = "TPEx"
            continue
        if current_market is None:
            continue
        cells = [td.get_text(strip=True) for td in tds]
        if cells and cells[0] == "類股名稱":
            continue  # 欄位標題列
        # 每兩個一組 (類股名稱, 流向率)
        for i in range(0, len(cells) - 1, 2):
            name, rate_str = cells[i], cells[i + 1]
            m = re.match(r"(-?\d+(?:\.\d+)?)%", rate_str)
            if name and m:
                result[current_market][name] = float(m.group(1))
    return result


def save_fund_flow(fund_flow: dict, path: str = None):
    path = path or FUND_FLOW_CACHE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fund_flow, f, ensure_ascii=False, indent=2)


def load_fund_flow(path: str = None) -> dict:
    path = path or FUND_FLOW_CACHE
    if not os.path.exists(path):
        return {"TWSE": {}, "TPEx": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 3. 大分類 -> 官方資金流向表分類 對照（人工整理，非精確科學對應，歡迎調整）
# ----------------------------------------------------------------------
# 資金流向表用的是 TWSE/TPEx 傳統粗分類（上市約34類、上櫃約22類），跟 ZHA 的121個
# 大分類是兩套不同系統，很多大分類需要「多對一」合併到同一個資金流向分類。
# 上櫃分類裡沒有「金融保險」「貿易百貨」「油電燃氣」「運動休閒」等項目（上櫃股本來
# 就沒有這些），也多了「文化創意」——若某大分類的成分股剛好是上櫃股、又對應到上市
# 才有的分類，會查不到資金流向率，sector_pipeline 會以「無資料」處理，不影響其餘欄位。
ZHA_TO_OFFICIAL_GROUP = {
    "水泥": "水泥",
    "食品加工": "食品", "大宗物資": "食品", "飲料相關": "食品",
    "塑化原料": "塑膠", "塑化製品": "塑膠",
    "化學纖維": "紡織纖維", "化纖原料": "紡織纖維", "成衣": "紡織纖維",
    "織布業": "紡織纖維", "工業紡織品": "紡織纖維", "紡紗": "紡織纖維", "紡織中游": "紡織纖維",
    "工具機": "電機機械", "機械零組件": "電機機械", "造船業": "電機機械",
    "機器人": "電機機械", "產業機械": "電機機械",
    "電力設備": "電器電纜", "家電": "電器電纜", "電線電纜": "電器電纜",
    "電池材料相關": "化工", "化學工業": "化工",
    "橡膠工業": "橡膠",
    "建材": "建材營造", "地產": "建材營造", "營造工程": "建材營造", "基礎建設營運": "建材營造",
    "家居用品": "居家生活",
    "造紙業": "造紙",
    "線材盤元": "鋼鐵", "條鋼": "鋼鐵", "不鏽鋼": "鋼鐵", "合金鋼": "鋼鐵",
    "非鐵金屬": "鋼鐵", "貴金屬": "鋼鐵", "板鋼": "鋼鐵", "金屬礦採選": "鋼鐵",
    "汽車服務相關": "汽車", "汽機車零組件": "汽車", "汽車內裝": "汽車",
    "車用金屬成型": "汽車", "車輛整車": "汽車",
    "車用電子": "電子零組件",
    "面板業": "光電", "面板零組件": "光電", "LED": "光電", "顯示器": "光電", "光學元件": "光電",
    "被動元件": "電子零組件", "電子零件元件": "電子零組件", "散熱模組": "電子零組件",
    "印刷電路板相關": "電子零組件", "電池": "電子零組件",
    "電子其他": "其他電子", "光碟片": "其他電子", "數位相機": "其他電子",
    "消費性電子產品": "其他電子", "穿戴式裝置": "其他電子", "電聲產品": "其他電子",
    "無人機": "其他電子",
    "IC設計": "半導體", "IC封裝測試": "半導體", "IC製造": "半導體", "分離式元件": "半導體",
    "半導體化學品": "半導體", "設備儀器商": "半導體", "生物辨識相關": "半導體",
    "射頻前端晶片": "半導體",
    "網通設備": "通信網路", "手機": "通信網路", "通訊服務": "通信網路",
    "光通訊": "通信網路", "手機零組件": "通信網路",
    "電子通路": "電子通路",
    "軟體業": "資訊服務", "INTERNET應用與服務": "資訊服務", "INTERNET技術與基礎設施": "資訊服務",
    "遊戲產業": "資訊服務",
    "週邊產品": "電腦及週邊設備", "工業電腦": "電腦及週邊設備",
    "電腦系統業": "電腦及週邊設備", "傳輸介面": "電腦及週邊設備",
    "封測服務與材料": "半導體",
    "運輸事業": "航運業",
    "旅宿／餐飲": "觀光餐旅", "休閒娛樂": "觀光餐旅",
    "時尚產業": "貿易百貨", "流通業": "貿易百貨", "無店舖販售": "貿易百貨",
    "金融業": "金融保險",
    "電力公共事業": "油電燃氣", "石油及天然氣": "油電燃氣", "電力": "油電燃氣", "煤": "油電燃氣",
    "太陽能": "綠能環保",
    "運動產業": "運動休閒",
    "傳播事業": "文化創意", "文化創意產業": "文化創意",
    "醫藥產業": "生技醫療", "生物科技": "生技醫療", "醫療服務": "生技醫療",
    "智慧醫療技術": "生技醫療", "醫藥流通": "生技醫療", "體外診斷用醫材": "生技醫療",
    "診斷與監測用醫材": "生技醫療", "手術與治療用醫材": "生技醫療",
    "輔助與彌補用醫材": "生技醫療", "其他醫療器材": "生技醫療", "獸醫相關": "生技醫療",
    "水資源": "其他", "其他公用事業": "其他", "控股公司": "其他",
    "農林漁牧": "其他", "礦石開採": "其他", "航天軍工": "其他",
    "綜合": "其他", "傳產其他": "其他", "製罐業": "其他", "資產股": "其他", "服務業": "其他",
}


def get_fund_flow_for_major(major_name: str, market: str, fund_flow: dict = None):
    """查詢某個大分類對應的官方資金流向率(%)。market 為 "TWSE" 或 "TPEx"。
    查不到（分類對照不到，或該市場沒有這個官方分類）回傳 None。"""
    fund_flow = fund_flow if fund_flow is not None else load_fund_flow()
    official = ZHA_TO_OFFICIAL_GROUP.get(major_name)
    if official is None:
        return None
    return fund_flow.get(market, {}).get(official)
