"""
題材新聞摘要 —— 用 Google News RSS 搜尋當日/近期跟某個題材相關的新聞。

為什麼用 Google News RSS 而不是「即時上網搜尋」：
本工具設計成「每天晚上收盤後，你自己在自己的電腦上執行 python3 sector_main.py」
就能全自動產出報告，不需要我（Claude）在場即時幫你搜尋。這代表新聞來源必須是
「你的電腦可以直接用程式碼呼叫」的東西，不能是需要對話互動的網頁搜尋。
Google News RSS（https://news.google.com/rss/search）是公開、免金鑰即可用的
新聞搜尋端點，缺點是：(1) 非官方公開API，格式未來可能調整；(2) 摘要通常只有
標題(可能加上來源/時間)，沒有真正的內文摘要。若要更完整的新聞內容摘要，之後
可以考慮串接付費新聞API（如NewsAPI）或另外寫爬蟲抓內文。
"""
import time
from xml.etree import ElementTree
from urllib.parse import quote

import requests

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def fetch_theme_news(theme_name: str, limit: int = None) -> list:
    """搜尋跟某個題材相關的新聞，回傳 [{title, link, source, published}, ...]。
    查無資料或請求失敗都回傳空list，不影響報告其餘部分正常產出。
    """
    limit = limit or config.SECTOR_NEWS_PER_THEME
    # 題材名稱裡的「/」等符號拿掉，加上「台股」關鍵字提高相關性
    query = theme_name.split("/")[0].split("(")[0].strip()
    query = f"{query} 台股"
    params = {"q": query, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}

    try:
        resp = requests.get(GOOGLE_NEWS_RSS, params=params, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except Exception as e:
        print(f"  [警告] 新聞搜尋失敗（{theme_name}）: {e}")
        return []

    items = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        if title and link:
            items.append({"title": title, "link": link, "source": source, "published": pub_date})

    time.sleep(config.REQUEST_SLEEP_SEC)
    return items


def fetch_news_for_top_sectors(sector_df, top_n: int = None, per_theme: int = None) -> dict:
    """對熱度分數前 top_n 名題材各搜尋新聞，回傳 {group_name: [news, ...]}。"""
    top_n = top_n or config.SECTOR_TOP_NEWS_COUNT
    per_theme = per_theme or config.SECTOR_NEWS_PER_THEME
    result = {}
    for _, row in sector_df.head(top_n).iterrows():
        group = row["group"]
        if group == "未分類" or group.startswith("官方產業:"):
            # 官方產業別分組太籠統(例如「半導體業」)，新聞搜尋準確度較低，先只對
            # 手動整理的概念股題材抓新聞；未分類/官方產業別分組不搜尋。
            continue
        result[group] = fetch_theme_news(group, limit=per_theme)
    return result
