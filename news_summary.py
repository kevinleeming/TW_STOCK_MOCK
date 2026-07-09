"""
把抓到的新聞標題，用 Anthropic API 濃縮成幾句話的「買進理由重點概要」。

需要在 config.py 填入 ANTHROPIC_API_KEY 才會啟用；沒有填的話，
summarize_stock_news() 一律回傳 None，呼叫端（screener_pipeline.py）會直接
跳過這個欄位，不影響報告其餘部分正常產出。

注意：Google News RSS（見 news_fetch.py）只提供「標題」，沒有完整內文，所以這裡
做的其實是「多則新聞標題共同重點的歸納整理」，不是逐篇新聞的深度摘要——如果多則
標題彼此不相關或資訊量太少，模型會如實反映「近期新聞重點有限」，不會編造標題以外
沒有提到的細節。
"""
import config

_client = None
_disabled = False
_warned = False
_account_error_seen = False   # 偵測到帳號層級錯誤(額度不足/key無效等)後，同一次執行不再重試


def _get_client():
    """延遲初始化 Anthropic client，且只警告一次（沒填key是常見的正常情況，
    不需要每檔股票都印一次警告訊息）。"""
    global _client, _disabled, _warned
    if _disabled:
        return None
    if _client is not None:
        return _client
    if not config.ANTHROPIC_API_KEY:
        if not _warned:
            print("  [資訊] 未設定 config.ANTHROPIC_API_KEY，個股新聞摘要功能將跳過"
                  "（不影響其他評分/報告內容）。")
            _warned = True
        _disabled = True
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"  [警告] 無法初始化 Anthropic client（{e}），個股新聞摘要功能停用。"
              f"請確認已安裝 anthropic 套件（pip install anthropic）且 API key 正確。")
        _disabled = True
        return None
    return _client


def _is_account_level_error(e: Exception) -> bool:
    """判斷是不是「帳號層級」的錯誤（額度不足、key無效、權限問題等）——這種錯誤
    不會因為換一檔股票重試就變好，同一次執行後面的股票一定也會失敗，直接短路
    (short-circuit) 剩下的呼叫，避免每檔股票都重複印出一樣的錯誤訊息、浪費時間。
    """
    msg = str(e).lower()
    keywords = ("credit balance", "invalid_request_error", "authentication_error",
                "permission_error", "invalid x-api-key", "insufficient")
    return any(k in msg for k in keywords)


def summarize_stock_news(stock_name: str, theme_name: str, news_items: list) -> str:
    """把 news_items（[{title, source, published}, ...]）摘要成2~3句話的買進理由
    重點概要。沒有設定API key、沒有新聞、或呼叫失敗都回傳 None。
    """
    global _account_error_seen
    if not news_items:
        return None
    if _account_error_seen:
        return None
    client = _get_client()
    if client is None:
        return None

    headlines = "\n".join(
        f"- {n['title']}" + (f"（{n['source']}）" if n.get("source") else "")
        for n in news_items
    )
    prompt = f"""以下是「{stock_name}」（相關題材/產業：{theme_name}）近期的新聞標題：

{headlines}

請用繁體中文、2~3句話，整理這些新聞標題透露出的重點，當作這檔股票的「買進理由重點
概要」給投資人參考。只能根據上面列出的標題內容整理，不要編造標題以外沒有提到的細節；
如果標題彼此不太相關或資訊量有限，就如實說明「近期新聞重點有限」，不要硬湊。不需要
開頭客套話，直接給重點內容。"""

    try:
        resp = client.messages.create(
            model=config.ANTHROPIC_SUMMARY_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in resp.content).strip()
        return text or None
    except Exception as e:
        if _is_account_level_error(e):
            _account_error_seen = True
            print(f"  [警告] Anthropic API帳號層級錯誤（可能是額度不足/key無效），"
                  f"本次執行剩餘股票的新聞摘要將全部跳過，不影響其他評分/報告內容。"
                  f"請至 https://console.anthropic.com/settings/billing 檢查額度或帳單設定。"
                  f"\n  原始錯誤（{stock_name}）: {e}")
        else:
            print(f"  [警告] 新聞摘要API呼叫失敗（{stock_name}）: {e}")
        return None
