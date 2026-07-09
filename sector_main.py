"""
產業/題材熱力圖 —— 主程式

用法:
  python sector_main.py                # 正式模式：抓取真實資料 + 搜尋新聞
  python sector_main.py --mock         # 測試模式：用模擬資料驗證流程（不需連網）
  python sector_main.py --no-news      # 正式模式但跳過新聞搜尋（較快、較省）

建議收盤後（約下午2點後，三大法人/融資資料通常在傍晚才會公布齊全，晚一點跑
籌碼資料會更完整）執行，輸出在 sector_report/sector_report.html，用瀏覽器開啟即可。
每天執行都會額外存一份到 sector_report/history/<資料日期>/。
"""
import argparse

from sector_pipeline import run_sector_report


def main():
    parser = argparse.ArgumentParser(description="台股產業/題材熱力圖")
    parser.add_argument("--mock", action="store_true", help="使用模擬資料測試流程（不連網）")
    parser.add_argument("--no-news", action="store_true", help="跳過新聞搜尋步驟")
    parser.add_argument("--out", default="sector_report", help="報告輸出資料夾")
    args = parser.parse_args()

    sector_df = run_sector_report(use_mock=args.mock, out_dir=args.out, fetch_news=not args.no_news)
    print("\n題材平均漲跌幅排行前幾名（網頁上可切換依資金流向/熱度分數排序）：")
    print(sector_df[["group", "price_change_avg", "inst_net_amount", "heat_score", "breadth_pct"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
