"""
選股評分模型 —— 主程式

用法:
  python screener_main.py            # 正式模式：抓取真實資料 + 個股新聞摘要
  python screener_main.py --mock     # 測試模式：用模擬資料驗證流程（不需連網）
  python screener_main.py --no-news  # 正式模式但跳過新聞搜尋/摘要步驟（較快）

個股新聞摘要需要在 config.py 填入 ANTHROPIC_API_KEY 才會產生摘要文字，
沒有填的話會自動跳過，不影響其餘評分/報告內容。
"""
import argparse

from screener_pipeline import run_screener


def main():
    parser = argparse.ArgumentParser(description="台股選股評分模型")
    parser.add_argument("--mock", action="store_true", help="使用模擬資料測試流程（不連網）")
    parser.add_argument("--no-news", action="store_true", help="跳過個股新聞搜尋/摘要步驟")
    parser.add_argument("--out", default="screener_report", help="報告輸出資料夾")
    args = parser.parse_args()

    picks = run_screener(use_mock=args.mock, out_dir=args.out, fetch_news=not args.no_news)
    print("\n前幾名：")
    print(picks[["stock_id", "name", "total_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
