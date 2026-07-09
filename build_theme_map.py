"""
更新「題材對照表」—— 從 MoneyDJ 抓取完整產業分類階層(大分類+細分類) + 個股清單，
以及當日「大盤資金流向表」(官方上市/上櫃類股資金流向率)。

這是「偶爾手動執行」的腳本，不是每天晚上都要跑的東西（見 moneydj_scraper.py 開頭
說明：細分類數量超過1000個，且股票的分類歸屬本身變動很慢，一週~一個月跑一次即可，
不需要放進每天執行的 sector_main.py 流程）。資金流向表則是「每天」會變的資料，
但這裡先跟著題材對照表一起抓一次存檔即可，之後如果想要更即時的資金流向可以自行
更頻繁地單獨呼叫 moneydj_scraper.fetch_fund_flow()。

用法:
  python build_theme_map.py              # 抓全部約121個大分類、1000+個細分類
                                          # （約20~30分鐘，會對MoneyDJ發送大量請求，
                                          # 請勿縮短 --sleep）
  python build_theme_map.py --limit 10   # 只抓前10個大分類，快速測試流程用
"""
import argparse

import moneydj_scraper


def main():
    parser = argparse.ArgumentParser(description="更新 MoneyDJ 題材對照表 + 資金流向表")
    parser.add_argument("--limit", type=int, default=None, help="只抓前N個大分類（測試用）")
    parser.add_argument("--sleep", type=float, default=1.0, help="每次請求間隔秒數（請勿設太小，避免對MoneyDJ造成負擔）")
    args = parser.parse_args()

    print("開始抓取 MoneyDJ 產業分類階層...")
    hierarchy = moneydj_scraper.fetch_taxonomy_hierarchy()
    n_major = len(hierarchy) if not args.limit else min(args.limit, len(hierarchy))
    n_children = sum(len(m["children"]) for m in (hierarchy[:args.limit] if args.limit else hierarchy))
    print(f"共 {len(hierarchy)} 個大分類，本次將抓取 {n_major} 個大分類、{n_children} 個細分類...")

    theme_map_data = moneydj_scraper.build_theme_map(sleep_sec=args.sleep, limit=args.limit)
    moneydj_scraper.save_theme_map(theme_map_data)

    total_stocks = sum(len(v) for v in theme_map_data.values())
    print(f"\n完成！共 {len(theme_map_data)} 個大分類，總計 {total_stocks} 檔次個股歸屬"
          f"（一檔股票可能同時屬於多個大分類），已存至 {moneydj_scraper.THEME_MAP_CACHE}")

    print("\n開始抓取當日大盤資金流向表...")
    try:
        fund_flow = moneydj_scraper.fetch_fund_flow()
        moneydj_scraper.save_fund_flow(fund_flow)
        print(f"完成！上市 {len(fund_flow.get('TWSE', {}))} 類、上櫃 {len(fund_flow.get('TPEx', {}))} 類，"
              f"已存至 {moneydj_scraper.FUND_FLOW_CACHE}")
    except Exception as e:
        print(f"  [警告] 資金流向表抓取失敗: {e}（不影響題材對照表，之後可重新單獨執行本腳本補抓）")

    print("\nsector_main.py 之後執行時會自動讀取這份完整分類與資金流向，不需要額外設定。")


if __name__ == "__main__":
    main()
