"""
全域參數設定
"""
import os

# ------- 選股參數 (ML回測模組 main.py/pipeline.py 使用) -------
TOP_N = 50                 # 取成交量前幾大股票
VOLUME_LOOKBACK_DAYS = 22   # 「近一個月」約 22 個交易日

# ------- 歷史資料 -------
HISTORY_YEARS = 2          # 建模用歷史資料長度（年）

# ------- 特徵 / 標籤 -------
PREDICTION_HORIZON = 5      # 預測未來 N 個交易日後的漲跌方向
MA_WINDOWS = [5, 10, 20, 60]
RSI_WINDOW = 14
VOLUME_MA_WINDOW = 5

# ------- 資料切分（依時間先後，避免 look-ahead）-------
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ------- 交易 / 回測 -------
INITIAL_CAPITAL = 1_000_000
SIGNAL_THRESHOLD = 0.5       # 模型預測機率 > 此值 -> 做多
TRANSACTION_COST_RATE = 0.004425  # 約略：手續費0.1425%*2 + 證交稅0.3%（賣出）

# ------- 驗證門檻（決定「正確」與否）-------
MIN_TEST_ACCURACY = 0.55        # 測試集方向準確率門檻
MIN_EXCESS_ANNUAL_RETURN = 0.0  # 策略年化報酬 需優於 benchmark 的幅度（0 = 打平即可）
MIN_SHARPE = 0.3

# ------- 自動迭代 -------
MAX_ITERATIONS = 5

# ------- 快取 -------
CACHE_DIR = "cache"

# ------- 網路請求 -------
REQUEST_TIMEOUT = 15
REQUEST_SLEEP_SEC = 0.6   # 避免打太快被官方網站擋
MAX_RETRY = 3

# ------- FinMind API（個股歷史日K / 上櫃三大法人 / 融資融券，見 data_fetch.py, chip_data.py 說明）-------
# 原因：TWSE 官方 STOCK_DAY 逐檔逐月查詢在候選池較大（如300檔 x 24個月）時，容易觸發官網
# 反爬蟲/流量限制（回傳空內容或逾時）；TPEx 則本身沒有可用的官方個股歷史API。
# 兩者統一改走 FinMind（單一股票單次查詢即可涵蓋整段區間，請求量大幅降低）。
# 免費註冊 https://finmindtrade.com 取得 token 填入下方，可將請求上限從匿名 300次/小時
# 提高到 600次/小時；候選池較大時建議申請，避免同一輪跑下來超過額度。
# 優先讀環境變數 FINMIND_API_TOKEN（部署到 GitHub Actions 時用 repository secret
# 注入，不用把 token 明文寫進程式碼/推上 GitHub）；本機執行也可以直接填在這裡。
FINMIND_API_TOKEN = os.environ.get("FINMIND_API_TOKEN", "")
FINMIND_REQUEST_SLEEP_SEC = 0.3

# ========================================================================
# 以下為「選股評分模型」(screener_*.py) 專用參數
# ========================================================================

# ------- 選股候選範圍 -------
# 個股歷史K線 + 融資餘額現在都改走 FinMind（見上方 FinMind 說明），每個候選股大約要
# 2次 FinMind 請求（K線1次+融資1次，融資已改成全市場都查，不只上櫃），上櫃股再多1次
# 三大法人查詢。候選池120檔粗估：120x2 + 上櫃股數(約占4成，約48次) + 1(產業別) ≈ 289次，
# 已經很接近匿名帳號「300次/小時」的額度上限——只要同一小時內重跑第二次（例如測試時），
# 或上櫃股比例偏高，就很容易超過額度，導致後段股票收到 HTTP 402（額度用完，見
# data_fetch.py 的 _mark_finmind_quota_exhausted 說明）而變成「資料不足」。
# 故調低到100，估算約 100x2+40+1=241次，保留較多緩衝空間；若已申請
# FINMIND_API_TOKEN（免費，額度提高到600次/小時），可以放心調高到 250~300。
SCREENER_CANDIDATE_TOP_VOLUME = 100   # 從全市場先取成交量前N大做候選（效能考量，避免對全部1700+檔都抓籌碼資料）
SCREENER_PICK_TOP_N = 15              # 最終選出幾檔

# ------- 流動性篩選（使用者反應：選股結果前幾名會出現成交量很低、乏人問津的股票）-------
# 根本原因：STEP2 的候選池篩選只用「單一天」的成交股數做排序，容易被單日噴出的
# 異常量（例如消息面帶量一日拉高、隔天就打回原形）誤導，且用「股數」而非「成交金額」
# 排序時，低價股即使成交金額很小，也可能因為股數換算後張數高而排進候選池。
# 因此改成：(1) 候選池改用「成交金額」(股價x股數)排序，避免低價股的股數優勢；
# (2) 進一步評分時，改用每檔股票近 SCREENER_LIQUIDITY_LOOKBACK_DAYS 個交易日「日K
# 歷史」算出的近期平均成交量/成交金額（比單日快照更能反映真實流動性），並設立
# 硬性門檻——低於門檻的股票直接排除、不會出現在最終選股結果，不只是扣分而已。
SCREENER_LIQUIDITY_LOOKBACK_DAYS = 20      # 計算「近期平均成交量/成交金額」取近幾個交易日（約1個月）
SCREENER_MIN_AVG_VOLUME_LOTS = 300         # 近期平均成交量至少要有幾「張」(1張=1000股)，低於此值直接排除
SCREENER_MIN_AVG_TURNOVER = 20_000_000     # 近期平均每日成交金額至少要有多少元，雙重把關（低價股張數夠但成交值太小一樣排除）

# ------- 個股新聞摘要（買進理由重點概要，見 news_fetch.py / news_summary.py）-------
# 新聞搜尋本身（Google News RSS）不需要金鑰；但要把抓到的新聞「標題」濃縮成幾句話的
# 摘要文字，需要呼叫 Anthropic API 做語言模型摘要。請自行到
# https://console.anthropic.com/settings/keys 申請一組 API key 填在下面
# ANTHROPIC_API_KEY——請直接在你自己電腦上編輯這個檔案填入，不要把key貼給任何人
# （包含貼在跟Claude的對話裡）。留空的話這個功能會自動跳過，不影響其他部分正常運作。
# 優先讀環境變數 ANTHROPIC_API_KEY（部署到 GitHub Actions 時用 repository secret
# 注入，不要把key明文寫進程式碼、更不要推上GitHub——見這次對話的提醒說明）；
# 本機執行也可以在你自己電腦上把 key 直接填在這裡的雙引號中間。
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # 用便宜快速的小模型即可，不需要大模型
SCREENER_NEWS_PER_STOCK = 4   # 每檔股票搜尋幾則相關新聞（其題材或所屬產業）拿來做摘要

# ------- 大戶/ 散戶 判斷門檻 -------
BIG_HOLDER_PRICE_THRESHOLD = 1000     # 股價 >= 1000 元，用「百張大戶」；否則用「千張大戶」
# TDCC 集保股權分散表 持股分級代碼所對應的「張數下限」（股數 / 1000）
# 15 = 1,000,001股以上 (>1000張)；10 = 100,001股以上 (>100張)。實際代碼請以 TDCC 開放資料欄位為準，
# 若欄位改版請調整 chip_data.py 中的 BRACKET_* 設定。
BIG_HOLDER_BRACKET_1000 = "15"   # 對應 > 1,000 張
BIG_HOLDER_BRACKET_100 = "10"    # 對應 > 100 張（含100~1000之間，用於百元以上股票的百張大戶）

# ------- 籌碼趨勢觀察期間 -------
CHIP_TREND_WEEKS = 4          # 大戶/股東人數 本地累積/回補的週數（需要比比較窗口多留一點緩衝，見 chip_data.py）
CHIP_TREND_COMPARE_WEEKS = 2  # 大戶持股比例/股東人數「變化」只看最近幾週（使用者指定：只看近2週變化）
INSTITUTION_LOOKBACK_DAYS = 14  # 三大法人買賣超 觀察近幾個交易日
MARGIN_LOOKBACK_DAYS = 22       # 融資餘額變化 觀察近幾個交易日（22個交易日約當1個月）

# ------- 技術指標參數（以日K線近似使用者要求的60分K，見README說明）-------
KD_N = 60
KD_K_SMOOTH = 3
KD_D_SMOOTH = 3
BOLL_WINDOW = 20
BOLL_STD_MULT = 2
BIAS_MA_WINDOW = 60   # 乖離率參考的均線（以60日線近似「季線」，左側交易拉回條件也用這條均線）
GRANVILLE_MA_WINDOW = 20  # 葛蘭碧法則參考的「月線」

# 布林帶寬 分類門檻（帶寬 = (上軌-下軌)/中軌）
BOLL_BANDWIDTH_NARROW = 0.05   # <5% 視為窄
BOLL_BANDWIDTH_NORMAL = 0.10   # 約10%正常
BOLL_BANDWIDTH_WIDE = 0.20     # >=20% 視為寬

# ------- 左側交易「出量創高後拉回季線」條件 -------
# 找出近期曾經量增創高、目前拉回到季線(BIAS_MA_WINDOW=60日線)附近或以下的股票，
# 並計算回檔幅度、以及這檔股票過去的「慣性拉回深度」(歷史拉回幅度中位數)作為參考基準。
LEFT_SIDE_LOOKBACK_DAYS = 120       # 往前找「近期高點」的範圍（約半年）
LEFT_SIDE_VOLUME_SPIKE_MULT = 2.0   # 高點附近的量 需達當時20日均量的幾倍，才算「出量」
LEFT_SIDE_VOLUME_MA_WINDOW = 20     # 判斷「出量」用的均量天數
LEFT_SIDE_NEAR_MA_BAND = 0.05       # 現價 <= 季線 x (1+此比例) 視為「拉回到季線附近」
LEFT_SIDE_MIN_DAYS_SINCE_HIGH = 5   # 創高後至少要經過幾個交易日才算「已經開始拉回」
LEFT_SIDE_DRAWDOWN_MIN_PCT = 3.0    # 計算「歷史慣性拉回深度」時，只採計拉回幅度超過此值的拉回（排除雜訊）

# ------- 評分權重（總分100）-------
# 使用者重新分配：籌碼50 / 左側交易拉回20 / 布林15 / 葛蘭碧15
SCORE_CHIP_MAX = 50
SCORE_LEFT_SIDE_MAX = 20
SCORE_BOLLINGER_MAX = 15
SCORE_GRANVILLE_MAX = 15

# 籌碼面 50分 內部拆分
SCORE_BIG_HOLDER_MAX = 20     # 大戶持股比例趨勢
SCORE_HOLDER_COUNT_MAX = 10   # 股東人數(散戶)趨勢
SCORE_INSTITUTION_MAX = 10    # 三大法人買賣超
SCORE_MARGIN_MAX = 10         # 融資餘額變化

# ========================================================================
# 以下為「產業/題材熱力圖」(sector_*.py) 專用參數
# ========================================================================
# 注意：這個數字現在「只」用來限制上櫃(TPEx)股票逐檔查詢FinMind三大法人買賣超的
# 數量，不會再用來篩選題材分組要涵蓋哪些股票（題材分組現在是全市場都納入，見
# sector_pipeline.build_full_universe 說明——先前用這個數字篩「候選池」再分組，
# 會導致大部分題材的真實成員被濾掉，點進題材詳細頁常常只剩1、2檔股票）。
# 上市(TWSE)三大法人資料是全市場一次請求就有、免費，不受此限制。若兩個工具
# (screener/sector)常同一晚一起跑，建議申請免費 FINMIND_API_TOKEN（見上方說明）。
SECTOR_CANDIDATE_TOP_VOLUME = 200   # 上櫃股當日三大法人查詢，只取成交值前N大（FinMind額度考量）
SECTOR_TOP_NEWS_COUNT = 5           # 熱度分數前幾名題材要抓新聞摘要
SECTOR_NEWS_PER_THEME = 3           # 每個題材抓幾則新聞

# 熱度分數權重（0~100，三個子項加權，見 sector_pipeline.compute_heat_score）
SECTOR_HEAT_WEIGHT_PRICE = 0.40        # 成交值加權平均漲跌幅（價格動能）
SECTOR_HEAT_WEIGHT_INSTITUTION = 0.35  # 三大法人買賣超金額佔該產業成交值比例（籌碼是否跟上）
SECTOR_HEAT_WEIGHT_BREADTH = 0.25      # 上漲家數比例（是否為普遍性上漲，判斷延續性）
