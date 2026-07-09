"""
產業/題材熱力圖 主流程

流程：
  1. 取得全部台股清單 + 官方產業別 + 當日(最新交易日 vs 前一交易日)漲跌幅快照(全市場)
  2. 建立「全市場」題材/產業分組（build_full_universe + attach_groups），刻意不像
     screener 那樣先篩出前N大候選池才分組——漲跌幅快照本來就是全市場一次抓齊、不消耗
     FinMind額度，先篩池子會讓大部分題材的真實成員被濾掉，點進題材詳細頁只剩一兩檔股票。
     每檔股票同時掛上：(a) 題材(theme_map.py，MoneyDJ細分類+手動清單合併，可能屬於0~多個)
     (b) 官方產業別(FinMind，一定會有，當題材對照表沒收錄到的股票的備援分類)
  3. 抓取「當日」三大法人買賣超：上市(TWSE)全市場一次請求就有、免費，故全抓；
     上櫃(TPEx)才會用到需要逐檔查詢的 FinMind，才受 SECTOR_CANDIDATE_TOP_VOLUME 限制
     （見 attach_institution_flow 說明）
  4. 依題材/產業分組，計算：成交值加權平均漲跌幅、上漲家數比例(廣度)、
     法人買賣超金額佔成交值比例 -> 綜合成「熱度分數」(0~100)
  5. 依熱度分數排名，產出「產業分析師視角」的簡短延續性判斷文字
  6. 對熱度分數最高的前幾名題材，搜尋當日相關新聞摘要
  7. 輸出 dark、暖色調 HTML 熱力圖報告，每個題材可點擊查看完整個股清單

注意：題材分類主要來自 MoneyDJ 細產業分類爬蟲（見 theme_map.py / moneydj_scraper.py
說明），並非保證完整或正確，使用前請自行核對。
"""
import os
import json
import datetime as dt

import numpy as np
import pandas as pd

import config
import data_fetch
import chip_data
import theme_map
import moneydj_scraper


def get_sector_universe(use_mock: bool = False):
    """回傳 (stock_list_df, price_df)。
    stock_list 含 stock_id, name, market, industry(官方產業別)。
    price_df 含 stock_id, date, close, prev_close, pct_change, volume。
    """
    if use_mock:
        import mock_data
        return mock_data.generate_mock_sector_universe()

    stock_list = data_fetch.get_all_stock_list()
    industry_map = data_fetch.get_industry_map()
    if not industry_map.empty:
        stock_list = stock_list.merge(industry_map, on="stock_id", how="left")
    else:
        stock_list["industry"] = None

    price_df = data_fetch.get_price_change_snapshot()
    return stock_list, price_df


def build_full_universe(stock_list: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """合併「全市場」股票清單 + 當日漲跌幅快照，不做任何前置篩選/截斷。

    這裡刻意不像 screener 那樣先用成交值篩出前N大候選池，是因為題材分組要看的是
    「這個題材底下所有真實成員的漲跌幅」——漲跌幅快照本來就是全市場一次抓齊、
    不消耗 FinMind 額度，如果先篩出前200大才分組，大部分題材(尤其是非熱門題材)
    的真實成員大多不在前200大成交值裡，會被整批濾掉，導致點進去只剩1、2檔股票
    （這是使用者實際回報、也已經在真實資料上重現的bug）。真正需要控制額度/請求量
    的只有「上櫃股當日三大法人查詢」這一項，見 attach_institution_flow()。
    """
    merged = price_df.merge(stock_list, on="stock_id", how="left").dropna(subset=["name"])
    merged["turnover"] = merged["close"] * merged["volume"]
    return merged.reset_index(drop=True)


def attach_groups(universe: pd.DataFrame) -> pd.DataFrame:
    """幫每檔股票標上「分組清單」：概念股題材(可能多個) + 官方產業別(保底一定有)。
    回傳新增一欄 groups（list of str）的 DataFrame。
    """
    membership = theme_map.get_theme_membership()
    out = universe.copy()

    def _groups_for(row):
        groups = list(membership.get(str(row["stock_id"]), []))
        industry = row.get("industry")
        if pd.notna(industry) and industry:
            groups.append(f"官方產業:{industry}")
        if not groups:
            groups.append("未分類")
        return groups

    out["groups"] = out.apply(_groups_for, axis=1)
    return out


def attach_institution_flow(universe: pd.DataFrame, use_mock: bool = False, tpex_top_n: int = None) -> pd.DataFrame:
    """抓取「當日」三大法人買賣超，合併進 DataFrame（新增 inst_net_buy 欄，單位：股）。

    上市(TWSE)：T86 一次請求就涵蓋全市場，免費、不消耗 FinMind 額度，故對「全部」
    TWSE 股票都抓，不受候選池大小限制。
    上櫃(TPEx)：官方無批次API，仍須對每一檔個股呼叫一次 FinMind，才會消耗額度，
    因此只對「成交值前 tpex_top_n 大」的上櫃股抓（見 config.SECTOR_CANDIDATE_TOP_VOLUME），
    其餘上櫃股的法人買賣超會是 NaN（顯示為「—」），但漲跌幅、是否列入題材清單不受影響。
    """
    if universe.empty:
        universe["inst_net_buy"] = np.nan
        return universe

    tpex_top_n = tpex_top_n or config.SECTOR_CANDIDATE_TOP_VOLUME

    if use_mock:
        import mock_data
        inst_df = mock_data.generate_mock_institution_flow(universe)
    else:
        target_date = str(universe["date"].iloc[0])
        tpex_rows = universe.loc[universe["market"] == "TPEx"].sort_values("turnover", ascending=False)
        tpex_ids = tpex_rows["stock_id"].head(tpex_top_n).tolist()
        inst_df = chip_data.get_today_institution_flow(target_date, tpex_stock_ids=tpex_ids)

    if inst_df.empty:
        universe["inst_net_buy"] = np.nan
        return universe
    out = universe.merge(
        inst_df[["stock_id", "net_buy"]].rename(columns={"net_buy": "inst_net_buy"}),
        on="stock_id", how="left",
    )
    return out


def aggregate_by_group(candidates: pd.DataFrame, fund_flow: dict = None) -> pd.DataFrame:
    """把候選池「展開」成每列一個(股票, 分組)的組合，再依分組聚合出兩種漲跌幅、
    上漲家數比例(廣度)、法人買賣超金額(資金流向估算)、MoneyDJ官方資金流向率。

    漲跌幅提供兩種算法：
    - price_change_avg：該題材「全部成員股」漲跌幅的單純平均數——這是題材頁預設
      呈現、排序用的主要指標，直覺對應「這個題材整體表現」。缺點是若題材內有一檔
      小型股當天暴漲/暴跌，單純平均會被它拉動較多，不反映資金規模差異。
    - price_change_weighted：用成交值(收盤價x成交量)加權的平均漲跌幅，資金規模
      越大的個股影響力越大，較能反映「這個題材今天的資金主流表現」，但不是使用者
      要求的「平均漲跌幅」，故只作為輔助參考，顯示在題材詳細頁裡。

    資金流向提供兩種來源：
    - inst_net_amount/inst_ratio_pct：本專案自己用三大法人買賣超估算的資金流向。
    - fund_flow_rate：MoneyDJ「大盤資金流向表」官方公布的類股資金流向率，經
      moneydj_scraper.ZHA_TO_OFFICIAL_GROUP 對照到最接近的官方分類。因為一個
      題材大分類可能同時有上市+上櫃成分股，而官方資金流向率是「上市」「上櫃」
      分開公布，這裡用該題材上市/上櫃各自的成交值佔比去加權合併成單一數字；
      若對照不到、或該市場沒有這個官方分類，缺的部分就不計入(不是當作0)。
    """
    exploded = candidates.explode("groups").rename(columns={"groups": "group"})
    exploded = exploded.dropna(subset=["group"])

    # 法人買賣超金額（估算）= 買賣超股數 x 當日收盤價，跟成交值一樣以「元」為單位；
    # 這裡就是「資金流入/流出」的量化數字：>0 代表三大法人合計對這個題材是淨買超
    # (資金流入)，<0 代表淨賣超(資金流出)。
    exploded["inst_net_amount"] = exploded["inst_net_buy"].fillna(0) * exploded["close"]
    exploded["is_up"] = exploded["pct_change"] > 0

    fine_labels = theme_map.get_fine_category_labels()

    rows = []
    for group, g in exploded.groupby("group"):
        turnover_sum = g["turnover"].sum()
        if turnover_sum <= 0:
            continue
        price_change_avg = g["pct_change"].mean()
        price_change_weighted = (g["pct_change"] * g["turnover"]).sum() / turnover_sum
        breadth = g["is_up"].mean() * 100
        inst_amount_sum = g["inst_net_amount"].sum()
        inst_ratio = inst_amount_sum / turnover_sum * 100  # 佔成交值百分比

        fund_flow_rate = None
        if fund_flow:
            official = moneydj_scraper.ZHA_TO_OFFICIAL_GROUP.get(group)
            if official:
                market_turnover = g.groupby("market")["turnover"].sum()
                total_rate, total_weight = 0.0, 0.0
                for mkt in ("TWSE", "TPEx"):
                    rate = fund_flow.get(mkt, {}).get(official)
                    w = market_turnover.get(mkt, 0)
                    if rate is not None and w > 0:
                        total_rate += rate * w
                        total_weight += w
                if total_weight > 0:
                    fund_flow_rate = total_rate / total_weight

        g_sorted = g.sort_values("pct_change", ascending=False)
        members = g_sorted.assign(
            inst_net_lots=lambda d: d["inst_net_buy"] / 1000,  # 保留NaN(無資料) 跟 0(真的無買賣超) 的區別
            fine_category=lambda d: d["stock_id"].apply(lambda sid: fine_labels.get((group, str(sid)), "")),
        )[["stock_id", "name", "market", "close", "pct_change", "inst_net_lots", "inst_net_amount", "fine_category"]].to_dict("records")

        rows.append({
            "group": group,
            "stock_count": len(g),
            "turnover": turnover_sum,
            "price_change_avg": price_change_avg,
            "price_change_weighted": price_change_weighted,
            "breadth_pct": breadth,
            "inst_net_amount": inst_amount_sum,
            "inst_ratio_pct": inst_ratio,
            "fund_flow_rate": fund_flow_rate,
            "top_movers": g_sorted[["stock_id", "name", "pct_change"]].head(3).to_dict("records"),
            "members": members,
        })
    return pd.DataFrame(rows)


def _normalize_0_100(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo) * 100


def compute_heat_score(sector_df: pd.DataFrame) -> pd.DataFrame:
    """綜合「價格動能 + 籌碼跟隨 + 上漲廣度」三項，算出 0~100 的熱度分數，
    權重見 config.SECTOR_HEAT_WEIGHT_*。分數越高代表當天話題性/資金關注度越高。
    """
    if sector_df.empty:
        return sector_df
    out = sector_df.copy()
    out["_n_price"] = _normalize_0_100(out["price_change_weighted"])
    out["_n_inst"] = _normalize_0_100(out["inst_ratio_pct"])
    out["_n_breadth"] = _normalize_0_100(out["breadth_pct"])
    out["heat_score"] = (
        out["_n_price"] * config.SECTOR_HEAT_WEIGHT_PRICE
        + out["_n_inst"] * config.SECTOR_HEAT_WEIGHT_INSTITUTION
        + out["_n_breadth"] * config.SECTOR_HEAT_WEIGHT_BREADTH
    )
    out = out.drop(columns=["_n_price", "_n_inst", "_n_breadth"])
    # 預設排序改用「題材總體漲跌幅(單純平均)」，這是使用者要求的主要排序依據；
    # 熱度分數/資金流向仍然算好放在資料裡，網頁上可以切換排序依據查看。
    return out.sort_values("price_change_avg", ascending=False).reset_index(drop=True)


def generate_commentary(row: pd.Series) -> str:
    """以「產業分析師」的角度，給出簡短的延續性判斷文字（規則式，非AI生成）。"""
    price = row["price_change_avg"]
    inst = row["inst_ratio_pct"]
    breadth = row["breadth_pct"]

    price_up = price > 0.5
    price_down = price < -0.5
    inst_support = inst > 0.3
    inst_against = inst < -0.3
    broad = breadth >= 60
    narrow = breadth < 40

    if price_up and inst_support and broad:
        return "全面性上漲，三大法人買盤同步進駐，籌碼與價格同步，具備延續性看點。"
    if price_up and inst_against:
        return "股價上漲但三大法人偏賣超，籌碼未跟上價格，需留意追高風險、注意是否為短線出貨。"
    if price_up and narrow:
        return "漲幅集中在少數個股，非全面性表態，話題延續性仍待觀察，非產業全面性行情。"
    if price_down and inst_support:
        return "股價拉回但三大法人逆勢買超，籌碼相對強勢，可能是逢低布局訊號，後續可留意是否止跌。"
    if price_down and inst_against and broad:
        return "普遍性下跌且三大法人同步賣超，籌碼與價格同步走弱，短線偏空、追蹤是否持續流出。"
    if abs(price) <= 0.5:
        return "今日漲跌幅有限，尚未形成明確趨勢，暫屬盤整格局。"
    return "價格與籌碼訊號不完全一致，建議搭配後續幾日資料觀察是否延續。"


def run_sector_report(use_mock: bool = False, out_dir: str = "sector_report", fetch_news: bool = True):
    print("=" * 60)
    print("STEP 1: 取得全部台股清單 + 當日漲跌幅快照")
    print("=" * 60)
    stock_list, price_df = get_sector_universe(use_mock=use_mock)
    if price_df.empty:
        raise RuntimeError("無法取得漲跌幅快照，請確認網路連線可存取 TWSE / TPEx 開放資料 API。")
    report_date = str(price_df["date"].iloc[0]) if "date" in price_df.columns else dt.date.today().isoformat()
    print(f"資料日期: {report_date}，共 {len(price_df)} 檔股票\n")

    print("=" * 60)
    print("STEP 2: 建立全市場題材/產業分組（不截斷候選池，見說明）")
    print("=" * 60)
    universe = build_full_universe(stock_list, price_df)
    universe = attach_groups(universe)
    print(f"全市場共 {len(universe)} 檔股票納入分組\n")

    print("=" * 60)
    print(f"STEP 3: 抓取當日三大法人買賣超（上市全市場免費全抓；上櫃僅前 {config.SECTOR_CANDIDATE_TOP_VOLUME} 大受FinMind額度限制）")
    print("=" * 60)
    universe = attach_institution_flow(universe, use_mock=use_mock)
    print(f"已合併籌碼資料，缺值 {universe['inst_net_buy'].isna().sum()} 檔\n")

    fund_flow = {}
    if not use_mock:
        print("=" * 60)
        print("STEP 3b: 抓取 MoneyDJ 當日大盤資金流向表（官方上市/上櫃類股資金流向率）")
        print("=" * 60)
        try:
            fund_flow = moneydj_scraper.fetch_fund_flow()
            moneydj_scraper.save_fund_flow(fund_flow)
            print(f"上市 {len(fund_flow.get('TWSE', {}))} 類、上櫃 {len(fund_flow.get('TPEx', {}))} 類\n")
        except Exception as e:
            print(f"  [警告] 即時抓取失敗({e})，改用上次快取的資金流向表...")
            fund_flow = moneydj_scraper.load_fund_flow()

    print("=" * 60)
    print("STEP 4: 依題材/產業分組彙總 + 計算熱度分數")
    print("=" * 60)
    sector_df = aggregate_by_group(universe, fund_flow=fund_flow)
    sector_df = compute_heat_score(sector_df)
    sector_df["commentary"] = sector_df.apply(generate_commentary, axis=1)
    print(f"共 {len(sector_df)} 個分組\n")

    news_map = {}
    if fetch_news and not use_mock:
        print("=" * 60)
        print(f"STEP 5: 搜尋熱度前 {config.SECTOR_TOP_NEWS_COUNT} 名題材的相關新聞")
        print("=" * 60)
        import news_fetch
        news_map = news_fetch.fetch_news_for_top_sectors(sector_df)
        print(f"已取得 {sum(len(v) for v in news_map.values())} 則新聞\n")

    os.makedirs(out_dir, exist_ok=True)
    sector_df.to_csv(os.path.join(out_dir, "sector_scores.csv"), index=False)
    _write_sector_html(sector_df, news_map, out_dir, report_date)

    archive_dir = os.path.join(out_dir, "history", report_date)
    os.makedirs(archive_dir, exist_ok=True)
    import shutil
    for fname in ("sector_report.html", "sector_scores.csv"):
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(archive_dir, fname))

    print(f"\n產業熱力圖已輸出至: {out_dir}/sector_report.html（資料日期: {report_date}）")
    return sector_df


def _heat_color(pct_change: float) -> str:
    """依漲跌幅回傳暖色調配色（台股慣例：紅漲、綠跌），幅度越大顏色越飽和。"""
    if pd.isna(pct_change):
        return "#3a352c"
    clamp = max(-5.0, min(5.0, pct_change)) / 5.0  # -1 ~ 1
    if clamp >= 0:
        # 漲：從暗紅到亮橘紅
        r = int(122 + clamp * (224 - 122))
        g = int(58 + clamp * (74 - 58))
        b = int(42 + clamp * (36 - 42))
    else:
        t = -clamp
        # 跌：從暗橄欖綠到較亮的橄欖綠(刻意不用鮮綠，維持整體暖色調氛圍)
        r = int(58 - t * (58 - 40))
        g = int(92 + t * (128 - 92))
        b = int(58 - t * (58 - 46))
    return f"rgb({r},{g},{b})"


def _write_sector_html(sector_df: pd.DataFrame, news_map: dict, out_dir: str, report_date: str):
    tiles_html = []
    detail_records = []
    for idx, row in enumerate(sector_df.to_dict("records")):
        color = _heat_color(row["price_change_avg"])
        arrow = "▲" if row["price_change_avg"] >= 0 else "▼"
        inst_arrow = "買超" if row["inst_ratio_pct"] >= 0 else "賣超"
        movers = "、".join(f"{m['name']}({m['pct_change']:+.1f}%)" for m in row["top_movers"])

        news_items = news_map.get(row["group"], [])
        news_html = ""
        if news_items:
            news_lis = "".join(
                f'<li><a href="{n["link"]}" target="_blank" rel="noopener">{n["title"]}</a>'
                f'<span class="news-src">{n.get("source","")}</span></li>'
                for n in news_items
            )
            news_html = f'<div class="news-box"><div class="news-title">相關新聞</div><ul>{news_lis}</ul></div>'

        fund_flow_rate = row.get("fund_flow_rate")
        has_flow = fund_flow_rate is not None and not pd.isna(fund_flow_rate)
        flow_badge = f'<span>官方資金流向率 {fund_flow_rate:+.2f}%</span>' if has_flow else ""

        tiles_html.append(f"""
        <div class="tile" tabindex="0" onclick="showDetail({idx})" data-idx="{idx}"
             data-price="{row['price_change_avg']:.4f}" data-inst="{row['inst_net_amount']:.0f}" data-heat="{row['heat_score']:.4f}"
             data-flow="{fund_flow_rate if has_flow else -999999:.4f}"
             style="background: linear-gradient(160deg, {color}22 0%, var(--panel) 55%); border-left: 4px solid {color};">
          <div class="tile-head">
            <span class="tile-name">{row['group']}</span>
            <span class="tile-heat">熱度 {row['heat_score']:.0f}</span>
          </div>
          <div class="tile-price" style="color: {color};">{arrow} {row['price_change_avg']:+.2f}%</div>
          <div class="tile-meta">
            <span>{row['stock_count']} 檔</span>
            <span>上漲家數比 {row['breadth_pct']:.0f}%</span>
            <span>法人{inst_arrow} 佔成交值 {abs(row['inst_ratio_pct']):.2f}%</span>
            {flow_badge}
          </div>
          <div class="tile-movers">領漲/跌: {movers}</div>
          <div class="tile-commentary">{row['commentary']}</div>
          {news_html}
          <div class="tile-hint">點擊查看個股清單 →</div>
        </div>
        """)

        members_clean = []
        for m in row.get("members", []):
            members_clean.append({
                "stock_id": m.get("stock_id"),
                "name": m.get("name"),
                "market": m.get("market"),
                "close": None if pd.isna(m.get("close")) else round(float(m.get("close")), 2),
                "pct_change": None if pd.isna(m.get("pct_change")) else round(float(m.get("pct_change")), 2),
                "inst_net_lots": None if pd.isna(m.get("inst_net_lots")) else round(float(m.get("inst_net_lots")), 1),
                "fine_category": m.get("fine_category") or "",
            })

        detail_records.append({
            "group": row["group"],
            "heat_score": round(float(row["heat_score"]), 1),
            "price_change_avg": round(float(row["price_change_avg"]), 2),
            "price_change_weighted": round(float(row["price_change_weighted"]), 2),
            "breadth_pct": round(float(row["breadth_pct"]), 1),
            "inst_ratio_pct": round(float(row["inst_ratio_pct"]), 2),
            "inst_net_amount": round(float(row["inst_net_amount"]), 0),
            "fund_flow_rate": round(float(fund_flow_rate), 2) if has_flow else None,
            "stock_count": int(row["stock_count"]),
            "commentary": row["commentary"],
            "color": color,
            "news": news_items,
            "members": members_clean,
        })

    detail_json = json.dumps(detail_records, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>台股產業/題材熱力圖</title>
<style>
  :root {{
    --bg: #17110c;
    --bg-gradient: radial-gradient(circle at 20% 0%, #241a10 0%, #17110c 45%, #100b08 100%);
    --panel: #2a2015;
    --panel-alt: #221909;
    --border: #4a3a26;
    --text: #f2e6d3;
    --text-muted: #b8a68b;
    --accent: #d9a441;
    --accent-2: #c97b4a;
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
    color: var(--accent);
    margin-bottom: 4px;
    text-shadow: 0 0 18px rgba(217, 164, 65, 0.25);
  }}
  .meta {{ color: var(--text-muted); font-size: 13px; margin-bottom: 14px; }}
  .sort-bar {{
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin-bottom: 20px;
  }}
  .sort-label {{ font-size: 13px; color: var(--text-muted); margin-right: 4px; }}
  .sort-btn {{
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 12px;
    cursor: pointer;
    font-family: inherit;
  }}
  .sort-btn:hover {{ color: var(--text); border-color: var(--accent-2); }}
  .sort-btn.active {{
    background: var(--accent);
    color: #1a1108;
    border-color: var(--accent);
    font-weight: 700;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }}
  .tile {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    cursor: pointer;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
  }}
  .tile:hover, .tile:focus {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.5);
    outline: none;
    border-color: var(--accent);
  }}
  .tile-hint {{
    font-size: 11px; color: var(--accent-2); text-align: right;
    margin-top: 6px; opacity: 0.8;
  }}
  .tile-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 6px;
  }}
  .tile-name {{ font-size: 16px; font-weight: 700; color: var(--text); }}
  .tile-heat {{ font-size: 12px; color: var(--accent); font-weight: 600; }}
  .tile-price {{ font-size: 22px; font-weight: 700; margin-bottom: 8px; }}
  .tile-meta {{
    display: flex; flex-wrap: wrap; gap: 10px;
    font-size: 12px; color: var(--text-muted); margin-bottom: 8px;
  }}
  .tile-movers {{ font-size: 12px; color: var(--text-muted); margin-bottom: 8px; }}
  .tile-commentary {{
    font-size: 13px; line-height: 1.5; color: var(--text);
    border-top: 1px solid var(--border); padding-top: 8px; margin-bottom: 4px;
  }}
  .news-box {{ margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 8px; }}
  .news-title {{ font-size: 12px; color: var(--accent-2); font-weight: 600; margin-bottom: 4px; }}
  .news-box ul {{ margin: 0; padding-left: 16px; }}
  .news-box li {{ font-size: 12px; margin-bottom: 4px; }}
  .news-box a {{ color: var(--text); text-decoration: none; }}
  .news-box a:hover {{ color: var(--accent); text-decoration: underline; }}
  .news-src {{ color: var(--text-muted); font-size: 11px; margin-left: 4px; }}
  .disclaimer {{
    margin-top: 24px; font-size: 12px; color: var(--text-muted);
    border-top: 1px solid var(--border); padding-top: 12px;
  }}

  /* ---- 題材詳細頁 modal ---- */
  .overlay {{
    display: none;
    position: fixed; inset: 0;
    background: rgba(10, 7, 4, 0.72);
    z-index: 100;
    padding: 40px 20px;
    overflow-y: auto;
  }}
  .overlay.open {{ display: block; }}
  .modal {{
    max-width: 860px;
    margin: 0 auto;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    padding: 24px 28px;
  }}
  .modal-close {{
    float: right;
    background: none; border: 1px solid var(--border); color: var(--text-muted);
    border-radius: 8px; padding: 4px 12px; cursor: pointer; font-size: 13px;
  }}
  .modal-close:hover {{ color: var(--accent); border-color: var(--accent); }}
  .modal-title {{ font-size: 20px; font-weight: 700; color: var(--accent); margin: 0 0 4px 0; }}
  .modal-summary {{
    display: flex; flex-wrap: wrap; gap: 18px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0 10px 0;
  }}
  .modal-summary .stat {{ min-width: 110px; }}
  .modal-summary .stat-label {{ font-size: 11px; color: var(--text-muted); }}
  .modal-summary .stat-value {{ font-size: 18px; font-weight: 700; }}
  .modal-commentary {{
    font-size: 13px; line-height: 1.6; color: var(--text); margin: 10px 0 18px 0;
  }}
  .modal-section-title {{
    font-size: 13px; color: var(--accent-2); font-weight: 700; margin: 18px 0 8px 0;
  }}
  .modal table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .modal th, .modal td {{
    text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border);
  }}
  .modal th {{ color: var(--text-muted); font-weight: 600; font-size: 12px; }}
  .modal tr:hover td {{ background: rgba(217,164,65,0.06); }}
  .modal .news-box {{ margin-top: 4px; }}
</style>
</head>
<body>
  <h1>台股產業/題材熱力圖</h1>
  <p class="meta">資料日期: {report_date}　產出時間: {dt.datetime.now().isoformat()}　（點擊題材色塊可查看該題材個股清單）</p>
  <div class="sort-bar">
    <span class="sort-label">排序依據：</span>
    <button class="sort-btn active" data-key="price" onclick="sortTiles('price', this)">題材平均漲跌幅</button>
    <button class="sort-btn" data-key="inst" onclick="sortTiles('inst', this)">資金流向（三大法人買賣超估算）</button>
    <button class="sort-btn" data-key="flow" onclick="sortTiles('flow', this)">官方資金流向率（MoneyDJ）</button>
    <button class="sort-btn" data-key="heat" onclick="sortTiles('heat', this)">熱度分數（綜合指標）</button>
  </div>
  <div class="grid" id="tile-grid">
    {''.join(tiles_html)}
  </div>
  <p class="disclaimer">
    免責聲明：本報告的「概念股題材」分類為人工整理之起點清單，可能不完整或有誤；
    熱度分數、法人買賣超金額為估算值，僅供研究與教育用途，不構成投資建議，
    實際交易請自行查證並評估風險。
  </p>

  <div class="overlay" id="detail-overlay" onclick="if(event.target===this) closeDetail();">
    <div class="modal">
      <button class="modal-close" onclick="closeDetail()">關閉 ✕</button>
      <div id="modal-body"></div>
    </div>
  </div>

  <script>
    const SECTOR_DATA = {detail_json};

    function fmtPct(v) {{
      if (v === null || v === undefined) return '—';
      const sign = v > 0 ? '+' : '';
      return sign + v.toFixed(2) + '%';
    }}
    function pctColor(v) {{
      if (v === null || v === undefined) return 'var(--text-muted)';
      return v >= 0 ? '#e0483d' : '#4a8035';
    }}

    function showDetail(idx) {{
      const d = SECTOR_DATA[idx];
      const arrow = d.price_change_avg >= 0 ? '▲' : '▼';
      const instArrow = d.inst_ratio_pct >= 0 ? '買超' : '賣超';

      let membersRows = d.members.map(m => `
        <tr>
          <td>${{m.stock_id}}</td>
          <td>${{m.name || ''}}</td>
          <td>${{m.market || ''}}</td>
          <td>${{m.close !== null ? m.close.toFixed(2) : '—'}}</td>
          <td style="color:${{pctColor(m.pct_change)}}; font-weight:600;">${{fmtPct(m.pct_change)}}</td>
          <td>${{m.inst_net_lots !== null ? m.inst_net_lots.toLocaleString(undefined, {{maximumFractionDigits:1}}) + ' 張' : '—'}}</td>
          <td style="color:var(--text-muted); font-size:12px;">${{m.fine_category || '—'}}</td>
        </tr>
      `).join('');

      let newsHtml = '';
      if (d.news && d.news.length) {{
        const items = d.news.map(n => `<li><a href="${{n.link}}" target="_blank" rel="noopener">${{n.title}}</a><span class="news-src">${{n.source || ''}}</span></li>`).join('');
        newsHtml = `<div class="news-box"><div class="news-title">相關新聞</div><ul>${{items}}</ul></div>`;
      }}

      const flowHtml = d.fund_flow_rate !== null
        ? `<div class="stat"><div class="stat-label">官方資金流向率(MoneyDJ)</div><div class="stat-value" style="color:${{d.fund_flow_rate>=0?'#e0483d':'#4a8035'}};">${{fmtPct(d.fund_flow_rate)}}</div></div>`
        : `<div class="stat"><div class="stat-label">官方資金流向率(MoneyDJ)</div><div class="stat-value" style="color:var(--text-muted);">無對應資料</div></div>`;

      document.getElementById('modal-body').innerHTML = `
        <h2 class="modal-title">${{d.group}}</h2>
        <div class="modal-summary">
          <div class="stat"><div class="stat-label">題材平均漲跌幅</div><div class="stat-value" style="color:${{d.color}};">${{arrow}} ${{fmtPct(d.price_change_avg)}}</div></div>
          <div class="stat"><div class="stat-label">三大法人買賣超金額(資金流向估算)</div><div class="stat-value" style="color:${{d.inst_net_amount>=0?'#e0483d':'#4a8035'}};">${{d.inst_net_amount>=0?'流入':'流出'}} ${{Math.abs(d.inst_net_amount/100000000).toFixed(2)}} 億</div></div>
          ${{flowHtml}}
          <div class="stat"><div class="stat-label">熱度分數(綜合指標)</div><div class="stat-value" style="color:var(--accent);">${{d.heat_score.toFixed(0)}}</div></div>
          <div class="stat"><div class="stat-label">成交值加權漲跌幅</div><div class="stat-value">${{fmtPct(d.price_change_weighted)}}</div></div>
          <div class="stat"><div class="stat-label">上漲家數比</div><div class="stat-value">${{d.breadth_pct.toFixed(0)}}%</div></div>
          <div class="stat"><div class="stat-label">三大法人${{instArrow}}佔成交值</div><div class="stat-value">${{Math.abs(d.inst_ratio_pct).toFixed(2)}}%</div></div>
          <div class="stat"><div class="stat-label">股票檔數</div><div class="stat-value">${{d.stock_count}}</div></div>
        </div>
        <div class="modal-commentary">${{d.commentary}}</div>
        ${{newsHtml}}
        <div class="modal-section-title">個股清單（依漲跌幅排序，「細分類」標示這檔股票在此大分類下的較小項度類別）</div>
        <table>
          <thead><tr><th>代號</th><th>名稱</th><th>市場</th><th>收盤價</th><th>漲跌幅</th><th>法人買賣超</th><th>細分類</th></tr></thead>
          <tbody>${{membersRows}}</tbody>
        </table>
      `;
      document.getElementById('detail-overlay').classList.add('open');
    }}

    function closeDetail() {{
      document.getElementById('detail-overlay').classList.remove('open');
    }}

    const SORT_ATTR = {{price: 'data-price', inst: 'data-inst', flow: 'data-flow', heat: 'data-heat'}};
    function sortTiles(key, btn) {{
      const grid = document.getElementById('tile-grid');
      const tiles = Array.from(grid.querySelectorAll('.tile'));
      const attr = SORT_ATTR[key];
      tiles.sort((a, b) => parseFloat(b.getAttribute(attr)) - parseFloat(a.getAttribute(attr)));
      tiles.forEach(t => grid.appendChild(t));
      document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }}

    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') closeDetail();
    }});
  </script>
</body>
</html>"""

    with open(os.path.join(out_dir, "sector_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
