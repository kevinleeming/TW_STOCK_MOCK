"""
選股評分模型 主流程

流程：
  1. 取得全部台股清單 + 最新收盤日快照 (收盤價/成交量)
  2. 為了控制運算與 API 請求量，先用成交量篩出前 SCREENER_CANDIDATE_TOP_VOLUME 大
     （預設300檔）當候選池，而不是對全市場 1700+ 檔都做深入分析
     -- 這是效能考量的兩階段篩選（先篩大池子，再深入評分），不是最終選股結果
  3. 對候選池：抓取日K歷史、計算技術指標(KD/布林/乖離/葛蘭碧)
  4. 抓取全市場的三大法人買賣超、融資融券餘額（近N日），以及 TDCC 集保股權分散表
     （每週快照，本地累積歷史）
  5. 逐檔計算 6 個子分數，加總為 0~100 總分
  6. 依總分排序，選出前 SCREENER_PICK_TOP_N 檔（預設15檔），輸出報告
"""
import os
import datetime as dt

import numpy as np
import pandas as pd

import config
import data_fetch
import chip_data
import indicators
import scoring
import theme_map
import news_fetch
import news_summary


def get_universe(use_mock: bool = False):
    """回傳 (stock_list_df, snapshot_df)。stock_list 含 stock_id, name, market, industry；
    snapshot_df 含 stock_id, close, volume。"""
    if use_mock:
        import mock_data
        return mock_data.generate_mock_screener_universe()

    stock_list = data_fetch.get_all_stock_list()
    industry_map = data_fetch.get_industry_map()
    if not industry_map.empty:
        stock_list = stock_list.merge(industry_map, on="stock_id", how="left")
    else:
        stock_list["industry"] = None
    snapshot = data_fetch.get_latest_market_snapshot()
    return stock_list, snapshot


def select_candidates(stock_list: pd.DataFrame, snapshot: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    """從全市場快照篩出候選池。改用「成交金額」(股價 x 成交股數) 排序，而不是單純
    的成交股數——避免低價股只是因為股數換算後張數多，就排擠掉真正高流動性的股票
    （見 config.py 說明）。這裡仍是「單日」快照的粗篩，主要目的是控制候選池大小
    以節省後續籌碼資料的 API 請求量；真正的流動性把關在 run_screener() 的
    STEP4 用每檔股票近期日K歷史算出的平均成交量/金額做硬性門檻過濾。"""
    top_n = top_n or config.SCREENER_CANDIDATE_TOP_VOLUME
    merged = snapshot.merge(stock_list, on="stock_id", how="left").dropna(subset=["name"])
    merged["turnover"] = merged["close"] * merged["volume"]
    merged = merged.sort_values("turnover", ascending=False).head(top_n).reset_index(drop=True)
    return merged


def score_one_stock(stock_id: str, market: str, price: float, avg_volume: float,
                     history: pd.DataFrame, tdcc_history: pd.DataFrame,
                     institution_df: pd.DataFrame, margin_df: pd.DataFrame) -> dict:
    """對單一股票計算完整的技術指標 + 7 子分數 + 總分。history 資料不足時回傳 None。

    avg_volume 參數改由呼叫端傳入「近期真實日K歷史」算出的平均成交量（見
    run_screener STEP4），而不是單日快照——流動性門檻檢查也在呼叫端做（低於門檻
    的股票不會呼叫到這個函式），這裡收到的 avg_volume 已經是通過門檻的值。
    """
    if history is None or len(history) < config.KD_N + 10:
        return None

    ind_df = indicators.compute_all_indicators(history)
    granville = indicators.classify_granville(ind_df)
    boll = indicators.classify_bollinger(ind_df)
    left_side = indicators.analyze_left_side_setup(ind_df)

    big_holder_series = chip_data.compute_big_holder_series(tdcc_history, stock_id, price)
    inst_summary = chip_data.summarize_institution_net(institution_df, stock_id)
    margin_summary = chip_data.summarize_margin_change(margin_df, stock_id)

    net_buy_sum = inst_summary["net_buy_sum"]
    institution_net_sign = 0
    if pd.notna(net_buy_sum):
        institution_net_sign = 1 if net_buy_sum > 0 else (-1 if net_buy_sum < 0 else 0)

    sub_scores = {
        "big_holder": scoring.score_big_holder_trend(big_holder_series),
        "holder_count": scoring.score_holder_count_trend(big_holder_series),
        "institution": scoring.score_institution(net_buy_sum, inst_summary["n_days"], avg_volume),
        "margin": scoring.score_margin(margin_summary["change_pct"], margin_summary["n_days"]),
        "left_side": scoring.score_left_side_pullback(left_side),
        "bollinger": scoring.score_bollinger(boll, granville, institution_net_sign),
        "granville": scoring.score_granville(granville),
    }
    result = scoring.compute_total_score(sub_scores)
    result["stock_id"] = stock_id
    result["market"] = market
    result["price"] = price
    result["kd_k"] = ind_df["k_value"].iloc[-1]
    result["kd_d"] = ind_df["d_value"].iloc[-1]
    result["bias_60"] = ind_df[f"bias_{config.BIAS_MA_WINDOW}"].iloc[-1]
    result["boll_bandwidth"] = boll.get("bandwidth")
    result["boll_position"] = boll.get("position")
    result["granville_signal"] = granville.get("signal")
    result["left_side_applicable"] = left_side.get("applicable")
    result["left_side_pullback_pct"] = left_side.get("current_pullback_pct")
    result["left_side_typical_pullback_pct"] = left_side.get("typical_pullback_pct")
    return result


def attach_news_summaries(picks: pd.DataFrame) -> pd.DataFrame:
    """幫每檔最終入選股票找出所屬題材(或備援用官方產業別)的近期新聞，並用Claude API
    摘要成2~3句話的「買進理由重點概要」。只對最終入選的 picks（預設15檔）做，
    不對整個候選池做，控制新聞搜尋量與API成本。

    沒有設定 config.ANTHROPIC_API_KEY 時，news_summary 會自動跳過並回傳 None，
    這裡一律容錯處理（新聞/摘要失敗都不影響其餘欄位，news_summary/news_items
    會是 None/空list，報告產出端看到 None 就不顯示這個區塊）。
    """
    membership = theme_map.get_theme_membership()
    summaries, news_lists, theme_used = [], [], []
    for _, row in picks.iterrows():
        sid = str(row["stock_id"])
        themes = membership.get(sid, [])
        industry = row.get("industry")
        theme_name = themes[0] if themes else (industry if pd.notna(industry) and industry and industry != "未知" else None)
        if not theme_name:
            summaries.append(None)
            news_lists.append([])
            theme_used.append(None)
            continue
        news_items = news_fetch.fetch_theme_news(theme_name, limit=config.SCREENER_NEWS_PER_STOCK)
        summary = news_summary.summarize_stock_news(row.get("name", sid), theme_name, news_items)
        summaries.append(summary)
        news_lists.append(news_items)
        theme_used.append(theme_name)

    out = picks.copy()
    out["news_theme"] = theme_used
    out["news_summary"] = summaries
    out["news_items"] = news_lists
    return out


def run_screener(use_mock: bool = False, out_dir: str = "screener_report", fetch_news: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: 取得全部台股清單 + 最新收盤日快照")
    print("=" * 60)
    stock_list, snapshot = get_universe(use_mock=use_mock)
    if snapshot.empty:
        raise RuntimeError("無法取得最新收盤快照，請確認網路連線可存取 TWSE / TPEx 開放資料 API。")
    print(f"全市場快照日期: {snapshot['date'].iloc[0] if 'date' in snapshot.columns else 'N/A'}，"
          f"共 {len(snapshot)} 檔股票\n")

    print("=" * 60)
    print(f"STEP 2: 依成交量篩出候選池（前 {config.SCREENER_CANDIDATE_TOP_VOLUME} 大）")
    print("=" * 60)
    candidates = select_candidates(stock_list, snapshot)
    print(f"候選池共 {len(candidates)} 檔\n")

    print("=" * 60)
    print("STEP 3: 抓取籌碼面資料（TDCC集保股權分散表 / 三大法人 / 融資融券）")
    print("=" * 60)
    if use_mock:
        import mock_data
        tdcc_history, institution_df, margin_df = mock_data.generate_mock_chip_data(candidates["stock_id"].tolist())
    else:
        all_ids = candidates["stock_id"].tolist()
        tpex_ids = candidates.loc[candidates["market"] == "TPEx", "stock_id"].tolist()
        # 傳入候選股票代號，本地歷史週數不足時會自動向TDCC官網歷史查詢頁面回補
        tdcc_history = chip_data.update_tdcc_history(backfill_stock_ids=all_ids)
        institution_df = chip_data.get_institution_trend(tpex_stock_ids=tpex_ids)
        # 融資餘額統一走 FinMind，需要「全部」候選股票代號（TWSE+TPEx），不再只傳 TPEx
        margin_df = chip_data.get_margin_trend(stock_ids=all_ids)
    print(f"TDCC本地歷史週數: {tdcc_history['date'].nunique() if not tdcc_history.empty else 0}, "
          f"三大法人資料筆數: {len(institution_df)}, 融資資料筆數: {len(margin_df)}\n")

    print("=" * 60)
    print("STEP 4: 逐檔抓取日K歷史、計算技術指標與評分")
    print("=" * 60)
    min_vol_shares = config.SCREENER_MIN_AVG_VOLUME_LOTS * 1000
    liquidity_days = config.SCREENER_LIQUIDITY_LOOKBACK_DAYS
    results = []
    n_skipped_liquidity = 0
    for i, row in candidates.iterrows():
        sid, market, price = row["stock_id"], row["market"], row["close"]
        if use_mock:
            import mock_data
            history = mock_data.MOCK_HISTORIES.get(sid)
        else:
            history = data_fetch.get_stock_history(sid, market)

        # 流動性硬性門檻：用這檔股票近 liquidity_days 個交易日「真實日K歷史」算出的
        # 平均成交量/成交金額，比單日快照更能反映真實流動性（不會被單日噴量誤導）。
        # 低於門檻的股票直接跳過，不會出現在候選/最終結果，不是只有扣分而已。
        avg_vol = None
        avg_turnover = None
        if history is not None and not history.empty and "volume" in history.columns:
            recent = history.tail(liquidity_days)
            avg_vol = recent["volume"].mean()
            if "close" in recent.columns:
                avg_turnover = (recent["close"] * recent["volume"]).mean()
        if avg_vol is None or pd.isna(avg_vol):
            continue  # 沒有足夠的歷史資料判斷流動性，視同資料不足跳過
        if avg_vol < min_vol_shares or (avg_turnover is not None and avg_turnover < config.SCREENER_MIN_AVG_TURNOVER):
            n_skipped_liquidity += 1
            continue

        res = score_one_stock(sid, market, price, avg_vol, history, tdcc_history, institution_df, margin_df)
        if res is None:
            continue
        res["name"] = row.get("name", "")
        industry_val = row.get("industry", "")
        res["industry"] = industry_val if pd.notna(industry_val) and industry_val else "未知"
        res["avg_volume_lots"] = round(avg_vol / 1000, 1)
        res["avg_turnover"] = round(avg_turnover, 0) if avg_turnover is not None else None
        results.append(res)
        if (i + 1) % 20 == 0:
            print(f"  已完成評分 {i + 1}/{len(candidates)}")

    print(f"  流動性門檻過濾：{n_skipped_liquidity} 檔因近{liquidity_days}日均量/均額過低被排除"
          f"（門檻：均量>={config.SCREENER_MIN_AVG_VOLUME_LOTS}張 且 均額>={config.SCREENER_MIN_AVG_TURNOVER:,.0f}元）")

    if not results:
        raise RuntimeError("沒有任何股票成功計算出評分，請檢查歷史資料是否足夠（需要至少 KD_N+10 個交易日），"
                            "或流動性門檻(config.SCREENER_MIN_AVG_VOLUME_LOTS/SCREENER_MIN_AVG_TURNOVER)設得過高。")

    all_scores = pd.DataFrame(results).sort_values("total_score", ascending=False).reset_index(drop=True)
    picks = all_scores.head(config.SCREENER_PICK_TOP_N).reset_index(drop=True)

    all_scores.to_csv(os.path.join(out_dir, "all_candidates_scores.csv"), index=False)

    if fetch_news and not use_mock:
        print("=" * 60)
        print("STEP 5: 搜尋入選股票題材近期新聞，摘要成買進理由重點概要")
        print("=" * 60)
        picks = attach_news_summaries(picks)

    # news_items 是 list-of-dict，CSV 存不了巢狀結構，另外拆出來存；CSV 只留摘要文字
    picks_for_csv = picks.drop(columns=["news_items"], errors="ignore")
    picks_for_csv.to_csv(os.path.join(out_dir, "top_picks.csv"), index=False)

    print(f"\n完成評分，候選池 {len(all_scores)} 檔中選出前 {len(picks)} 檔。")

    _write_markdown_report(picks, out_dir, tdcc_weeks=tdcc_history['date'].nunique() if not tdcc_history.empty else 0)
    _write_html_report(picks, out_dir)

    # 每天保留一份存檔版本：out_dir/history/YYYY-MM-DD/，方便回顧歷史選股結果、
    # 比對不同天的分數變化。頂層 report.md/report.html 永遠是「最新一次」的結果。
    report_date = str(snapshot["date"].iloc[0]) if "date" in snapshot.columns and not snapshot.empty else dt.date.today().isoformat()
    _archive_daily_report(out_dir, report_date)

    print(f"\n報告已輸出至: {out_dir}/（本次資料日期: {report_date}，歷史存檔於 {out_dir}/history/{report_date}/）")
    return picks


def _archive_daily_report(out_dir: str, report_date: str):
    """把本次輸出的報告/評分檔複製一份到 out_dir/history/{report_date}/ 存檔，
    讓每天執行都留下一個版本，不會被隔天的結果覆蓋掉。"""
    import shutil
    archive_dir = os.path.join(out_dir, "history", report_date)
    os.makedirs(archive_dir, exist_ok=True)
    for fname in ("report.md", "report.html", "all_candidates_scores.csv", "top_picks.csv"):
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(archive_dir, fname))


def _write_markdown_report(picks: pd.DataFrame, out_dir: str, tdcc_weeks: int):
    lines = []
    lines.append("# 台股選股評分報告\n")
    lines.append(f"產出時間: {dt.datetime.now().isoformat()}\n")
    if tdcc_weeks < 2:
        lines.append(
            f"> **注意**：目前本地僅累積 {tdcc_weeks} 週的 TDCC 集保股權分散表快照，"
            "大戶持股/股東人數趨勢分數暫以中性值計算。建議每週執行一次本程式，"
            "累積數週資料後，籌碼趨勢分數會更準確。\n"
        )
    lines.append("## 選股結果（依總分排序）\n")
    lines.append(
        f"> 已套用流動性門檻：近{config.SCREENER_LIQUIDITY_LOOKBACK_DAYS}日平均成交量 >= "
        f"{config.SCREENER_MIN_AVG_VOLUME_LOTS}張，且平均成交金額 >= "
        f"{config.SCREENER_MIN_AVG_TURNOVER:,.0f}元，低於門檻的股票已直接排除。\n"
    )
    lines.append("| 排名 | 股票代號 | 名稱 | 產業 | 市場 | 收盤價 | 近期均量(張) | 總分 | 籌碼(50) | 左側拉回(20) | 布林(15) | 葛蘭碧(15) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in picks.iterrows():
        lines.append(
            f"| {i+1} | {row['stock_id']} | {row['name']} | {row.get('industry', '未知')} | {row['market']} | {row['price']:.2f} | "
            f"{row.get('avg_volume_lots', float('nan')):.0f} | "
            f"{row['total_score']:.1f} | {row['chip_score']:.1f} | {row['left_side_score']:.1f} | "
            f"{row['bollinger_score']:.1f} | {row['granville_score']:.1f} |"
        )

    lines.append("\n## 個股選股理由\n")
    for i, row in picks.iterrows():
        lines.append(f"### {i+1}. {row['stock_id']} {row['name']}（總分 {row['total_score']:.1f}）\n")
        news_summary_text = row.get("news_summary")
        if pd.notna(news_summary_text) and news_summary_text:
            lines.append(f"**近期新聞摘要（{row.get('news_theme', '')}）：** {news_summary_text}\n")
            news_items = row.get("news_items") or []
            for n in news_items[:3]:
                lines.append(f"  - [{n['title']}]({n['link']})" + (f"（{n['source']}）" if n.get("source") else ""))
            if news_items:
                lines.append("")
        for reason in row["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append(
        "> 免責聲明：本評分模型僅供研究與教育用途，不構成投資建議。"
        "籌碼與技術指標僅反映歷史資料，不保證未來股價表現，實際交易請自行評估風險。\n"
    )

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_html_report(picks: pd.DataFrame, out_dir: str):
    rows_html = []
    for i, row in picks.iterrows():
        reasons_html = "".join(f"<li>{r}</li>" for r in row["reasons"])
        news_summary_text = row.get("news_summary")
        news_html = ""
        if pd.notna(news_summary_text) and news_summary_text:
            news_items = row.get("news_items") or []
            link_parts = []
            for n in news_items[:3]:
                src_suffix = f" ({n['source']})" if n.get("source") else ""
                link_parts.append(
                    f'<li><a href="{n["link"]}" target="_blank" rel="noopener">{n["title"]}</a>{src_suffix}</li>'
                )
            links_html = "".join(link_parts)
            news_html = f"""
              <div class="news-summary">
                <strong>近期新聞摘要（{row.get('news_theme', '')}）：</strong> {news_summary_text}
                <ul class="news-links">{links_html}</ul>
              </div>
            """
        rows_html.append(f"""
        <tr>
          <td>{i+1}</td>
          <td>{row['stock_id']}</td>
          <td>{row['name']}</td>
          <td><span class="industry-tag">{row.get('industry', '未知')}</span></td>
          <td>{row['market']}</td>
          <td>{row['price']:.2f}</td>
          <td>{row.get('avg_volume_lots', float('nan')):.0f}</td>
          <td class="score-total">{row['total_score']:.1f}</td>
          <td>{row['chip_score']:.1f}</td>
          <td>{row['left_side_score']:.1f}</td>
          <td>{row['bollinger_score']:.1f}</td>
          <td>{row['granville_score']:.1f}</td>
          <td>{news_html}<ul>{reasons_html}</ul></td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>台股選股評分報告</title>
<style>
  :root {{
    --bg: #0a0f1a;
    --bg-gradient: radial-gradient(circle at 15% 0%, #10192e 0%, #0a0f1a 45%, #060a12 100%);
    --panel: #121b2e;
    --panel-alt: #0f1727;
    --border: #223049;
    --text: #dbe4f3;
    --text-muted: #7e93b3;
    --accent: #38bdf8;
    --accent-2: #22d3ee;
    --accent-warn: #f87171;
    --tag-bg: #16283f;
    --tag-text: #7dd3fc;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Microsoft JhengHei", "PingFang TC", sans-serif;
    margin: 0; padding: 32px;
    background: var(--bg-gradient);
    color: var(--text);
    min-height: 100vh;
  }}
  h1 {{
    font-size: 24px;
    letter-spacing: 0.5px;
    color: var(--accent-2);
    margin-bottom: 4px;
    text-shadow: 0 0 18px rgba(34, 211, 238, 0.25);
  }}
  .meta {{ color: var(--text-muted); font-size: 13px; margin-bottom: 20px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    background: var(--panel);
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    border-radius: 10px;
    overflow: hidden;
  }}
  th, td {{
    border-bottom: 1px solid var(--border);
    padding: 10px 12px;
    font-size: 13px;
    vertical-align: top;
    color: var(--text);
  }}
  th {{
    background: linear-gradient(180deg, #17273f 0%, #101c30 100%);
    color: var(--accent-2);
    font-weight: 600;
    letter-spacing: 0.3px;
    position: sticky;
    top: 0;
  }}
  tr:nth-child(even) td {{ background: var(--panel-alt); }}
  tr:hover td {{ background: #17233a; }}
  .score-total {{
    font-weight: 700;
    color: var(--accent);
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.35);
  }}
  .industry-tag {{
    display: inline-block;
    background: var(--tag-bg);
    color: var(--tag-text);
    border: 1px solid #234;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    white-space: nowrap;
  }}
  ul {{ margin: 0; padding-left: 18px; color: var(--text-muted); }}
  ul li {{ margin-bottom: 4px; }}
  .news-summary {{
    background: #0f2130;
    border: 1px solid #1e3a52;
    border-left: 3px solid var(--accent-2);
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 8px;
    font-size: 12.5px;
    color: var(--text);
  }}
  .news-links {{ margin: 6px 0 0; padding-left: 16px; }}
  .news-links li {{ margin-bottom: 2px; }}
  .news-links a {{ color: var(--accent); text-decoration: none; }}
  .news-links a:hover {{ text-decoration: underline; }}
  .disclaimer {{
    margin-top: 20px;
    font-size: 12px;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    padding-top: 12px;
  }}
</style>
</head>
<body>
  <h1>台股選股評分報告</h1>
  <p class="meta">產出時間: {dt.datetime.now().isoformat()}</p>
  <p class="meta">流動性門檻：近{config.SCREENER_LIQUIDITY_LOOKBACK_DAYS}日平均成交量 >= {config.SCREENER_MIN_AVG_VOLUME_LOTS}張，且平均成交金額 >= {config.SCREENER_MIN_AVG_TURNOVER:,.0f}元（低於門檻已排除）</p>
  <table>
    <thead>
      <tr>
        <th>排名</th><th>代號</th><th>名稱</th><th>產業</th><th>市場</th><th>收盤價</th><th>近期均量(張)</th>
        <th>總分</th><th>籌碼(50)</th><th>左側拉回(20)</th><th>布林(15)</th><th>葛蘭碧(15)</th><th>選股理由</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
  <p class="disclaimer">免責聲明：本評分模型僅供研究與教育用途，不構成投資建議。歷史籌碼與技術指標不保證未來股價表現。</p>
</body>
</html>
"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
