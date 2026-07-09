"""
台股量能前50檔 模型分析 / 回測 / 自動迭代驗證 —— 主程式

用法:
  python main.py            # 正式模式：抓取真實 TWSE / TPEx 資料
  python main.py --mock     # 測試模式：用模擬資料驗證整條流程是否正常運作
"""
import argparse

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="台股量能前50檔 模型分析與回測")
    parser.add_argument("--mock", action="store_true", help="使用模擬資料測試流程（不連網）")
    parser.add_argument("--out", default="report", help="報告輸出資料夾")
    args = parser.parse_args()

    run_pipeline(use_mock=args.mock, out_dir=args.out)


if __name__ == "__main__":
    main()
