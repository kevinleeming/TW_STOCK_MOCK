"""
台股「產業/題材」大分類對照表。

有兩層資料來源，get_theme_membership() 會自動合併：
1. 主要來源：cache/moneydj_theme_map.json —— 由 build_theme_map.py 從 MoneyDJ
   理財網（https://www.moneydj.com/Z/ZH/ZHA/ZHA.djhtm）抓取，格式為
   {大分類名稱: {stock_id: [細分類名稱, ...]}}（實測121個大分類，例如IC設計、
   面板業、被動元件...；每個大分類底下可能有好幾個細分類，例如IC設計底下有
   網路卡IC、智慧卡IC、CPU、MCU...，這些細分類已經彙總在大分類裡，不會分開
   顯示成一堆瑣碎題材——使用者只想看大分類這一層的排序/分析，細分類只在
   點進題材詳細頁、看個股清單時，用來標示「這檔股票在這個大分類底下具體算
   哪個細項」）。第一次使用前請先手動執行一次：
       python build_theme_map.py
   （約20~30分鐘，之後不需要每天重跑，見 moneydj_scraper.py 說明——股票的分類
   歸屬變動很慢，建議一週~一個月重新執行一次即可保持更新。）

   注意：這個檔案格式在某次更新後改成大分類階層（之前是扁平的細分類清單），
   如果你的 cache/moneydj_theme_map.json 是舊版格式（value 是 list 而不是
   dict），請重新執行一次 build_theme_map.py 更新成新格式，否則會被忽略。

2. 備用/補強清單：下方 THEME_STOCKS —— 還沒執行過 build_theme_map.py 時的
   起點清單，或是你想手動補充/覆寫 MoneyDJ 抓不到的個股時使用。個股可能同時
   屬於多個題材、清單可能有遺漏，使用前請自行核對，也歡迎直接編輯增修。
   這裡面的股票沒有細分類標籤（因為是手動清單，本來就沒有這一層資訊）。

格式：{題材名稱: [stock_id, ...]}
"""

THEME_STOCKS = {
    "矽晶圓": ["6488", "5483", "6182"],
    "ABF載板/先進封裝": ["3037", "8046", "6274"],
    "AI伺服器/伺服器供應鏈": ["2382", "6669", "2317", "3231", "2356"],
    "IC設計": ["2454", "3661", "3529", "6533", "2379"],
    "記憶體": ["2408", "3260", "8299"],
    "PCB/印刷電路板": ["2367", "2313", "3037", "6213"],
    "被動元件": ["2327", "2492"],
    "航運": ["2603", "2609", "2615"],
    "重電/電網": ["1503", "1504", "1519"],
    "綠能/再生能源": ["6443", "3576"],
    "生技醫療": ["4174", "1795", "6547"],
}


def _load_moneydj_major_map() -> dict:
    """讀取 MoneyDJ 大分類對照表，並容忍舊版扁平格式（value是list）——
    偵測到舊格式就忽略（回傳空dict），避免整支程式因格式不符而壞掉，
    呼叫端會印出提示請使用者重新執行 build_theme_map.py。"""
    import moneydj_scraper
    raw = moneydj_scraper.load_theme_map()
    if not raw:
        return {}
    sample_value = next(iter(raw.values()))
    if isinstance(sample_value, list):
        print("  [警告] cache/moneydj_theme_map.json 是舊版扁平格式，"
              "本次會忽略此檔案並改用手動清單。請重新執行 python build_theme_map.py "
              "更新成新的大分類階層格式。")
        return {}
    return raw


def get_theme_membership() -> dict:
    """回傳 stock_id -> [大分類名稱, ...] 的反向對照（一檔股票可能屬於多個大分類）。
    優先合併 MoneyDJ 抓取的完整大分類（若存在），再疊加手動清單 THEME_STOCKS
    （同一檔股票若兩邊都有標記到同名題材，不會重複附加）。
    """
    major_map = _load_moneydj_major_map()

    membership = {}

    def _add(theme, stock_ids):
        for sid in stock_ids:
            sid = str(sid)
            existing = membership.setdefault(sid, [])
            if theme not in existing:
                existing.append(theme)

    for major_name, stock_dict in major_map.items():
        _add(major_name, stock_dict.keys())
    for theme, stock_ids in THEME_STOCKS.items():
        _add(theme, stock_ids)

    return membership


def get_fine_category_labels() -> dict:
    """回傳 {(大分類名稱, stock_id): "細分類A、細分類B"} 的對照，用於題材詳細頁
    在個股清單裡標示「這檔股票在這個大分類下屬於哪個較小項度的類別」。
    手動清單 THEME_STOCKS 裡的股票沒有細分類資訊，查不到時呼叫端應顯示「—」。
    """
    major_map = _load_moneydj_major_map()
    labels = {}
    for major_name, stock_dict in major_map.items():
        for sid, fine_names in stock_dict.items():
            labels[(major_name, str(sid))] = "、".join(fine_names)
    return labels
