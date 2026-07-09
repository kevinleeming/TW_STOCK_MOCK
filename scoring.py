"""
選股評分邏輯（總分100）
  籌碼面 50分 = 大戶持股趨勢20 + 股東人數(散戶)趨勢10 + 三大法人買賣超10 + 融資餘額變化10
  左側交易拉回 20分 = 出量創高後拉回季線的型態評分
  布林評級 15分 = 依帶寬(窄/正常/寬) x 位置(上/中/下軌) x 是否開喇叭 x 三大法人買賣方向 綜合判斷
  葛蘭碧法則 15分 = 依葛蘭碧5條規則判斷買賣點

這一版是「簡單、透明、可重現」的規則式評分，方便使用者之後直接拿分數去做回測；
所有門檻與級距都集中在 config.py，之後要微調非常方便。
"""
import numpy as np
import pandas as pd

import config


def score_big_holder_trend(big_holder_series: pd.DataFrame) -> dict:
    """大戶持股比例趨勢（20分）。只看最近 config.CHIP_TREND_COMPARE_WEEKS 週的變化
    （預設2週）：這段期間大戶持股比例明顯增加，視為大戶進場的好徵兆。
    大戶門檻依股價高低分「千張大戶」(股價<1000元) / 「百張大戶」(股價>=1000元)，
    這個判斷已經在 chip_data.compute_big_holder_series() 依股價自動切換，此處只需
    處理「變化幅度」的評分。
    """
    max_pts = config.SCORE_BIG_HOLDER_MAX
    if big_holder_series is None or len(big_holder_series) < 2:
        return {"points": max_pts * 0.5, "reason": "籌碼歷史快照不足（少於2週），大戶趨勢暫給中性分數，建議持續每週執行以累積歷史資料"}

    series = big_holder_series.dropna(subset=["big_holder_pct"])
    if len(series) < 2:
        return {"points": max_pts * 0.5, "reason": "大戶持股比例資料不足，暫給中性分數"}

    compare_weeks = min(config.CHIP_TREND_COMPARE_WEEKS, len(series) - 1)
    change = series["big_holder_pct"].iloc[-1] - series["big_holder_pct"].iloc[-1 - compare_weeks]
    label = "百張大戶" if series.attrs.get("use_100", False) else "千張大戶"

    if change >= 2:
        pts = max_pts
    elif change > 0:
        pts = max_pts * 0.7
    elif change == 0:
        pts = max_pts * 0.5
    elif change > -2:
        pts = max_pts * 0.25
    else:
        pts = 0.0

    reason = f"{label}持股比例近{compare_weeks}週變化 {change:+.2f} 個百分點"
    return {"points": pts, "reason": reason, "change": change}


def score_holder_count_trend(big_holder_series: pd.DataFrame) -> dict:
    """股東人數(散戶)趨勢（10分），只看最近 config.CHIP_TREND_COMPARE_WEEKS 週的變化
    （預設2週）。人數下降視為籌碼集中、散戶減少的良好訊號。"""
    max_pts = config.SCORE_HOLDER_COUNT_MAX
    if big_holder_series is None or len(big_holder_series) < 2:
        return {"points": max_pts * 0.5, "reason": "股東人數歷史快照不足，暫給中性分數"}

    series = big_holder_series.dropna(subset=["total_holders"])
    if len(series) < 2:
        return {"points": max_pts * 0.5, "reason": "股東人數資料不足，暫給中性分數"}

    compare_weeks = min(config.CHIP_TREND_COMPARE_WEEKS, len(series) - 1)
    base = series["total_holders"].iloc[-1 - compare_weeks]
    if base == 0:
        return {"points": max_pts * 0.5, "reason": "股東人數資料不足，暫給中性分數"}
    change_pct = (series["total_holders"].iloc[-1] - base) / base * 100

    if change_pct <= -2:
        pts = max_pts
    elif change_pct < 0:
        pts = max_pts * 0.7
    elif change_pct == 0:
        pts = max_pts * 0.5
    elif change_pct < 2:
        pts = max_pts * 0.25
    else:
        pts = 0.0

    reason = f"股東人數近{compare_weeks}週變化 {change_pct:+.2f}%（下降代表散戶籌碼流出、籌碼集中）"
    return {"points": pts, "reason": reason, "change_pct": change_pct}


def score_institution(net_buy_sum: float, n_days: int, avg_volume: float) -> dict:
    """三大法人買賣超（10分）。以近期淨買超股數占同期平均成交量的比例來評分，避免不同股票規模不可比。
    顯示的買賣超數字改用「張」為單位（1張=1000股），比較符合台股習慣用語。"""
    max_pts = config.SCORE_INSTITUTION_MAX
    if n_days == 0 or np.isnan(net_buy_sum) or not avg_volume:
        return {"points": max_pts * 0.5, "reason": "三大法人買賣超資料不足，暫給中性分數"}

    ratio = net_buy_sum / (avg_volume * n_days)  # 淨買超占期間總成交量比例（內部運算仍用股數，避免單位混淆）

    if ratio > 0.03:
        pts = max_pts
    elif ratio > 0:
        pts = max_pts * 0.7
    elif ratio == 0:
        pts = max_pts * 0.5
    elif ratio > -0.03:
        pts = max_pts * 0.25
    else:
        pts = 0.0

    direction = "買超" if net_buy_sum > 0 else ("賣超" if net_buy_sum < 0 else "買賣平衡")
    net_buy_lots = abs(net_buy_sum) / 1000
    reason = f"近{n_days}個交易日三大法人合計{direction} {net_buy_lots:,.0f} 張（約當期成交量 {ratio*100:.1f}%）"
    return {"points": pts, "reason": reason, "ratio": ratio, "net_buy_sum": net_buy_sum}


def score_margin(change_pct: float, n_days: int) -> dict:
    """融資餘額變化（10分），下降視為籌碼變乾淨的良好訊號。"""
    max_pts = config.SCORE_MARGIN_MAX
    if n_days < 2 or np.isnan(change_pct):
        return {"points": max_pts * 0.5, "reason": "融資餘額資料不足，暫給中性分數"}

    if change_pct <= -10:
        pts = max_pts
    elif change_pct < 0:
        pts = max_pts * 0.7
    elif change_pct == 0:
        pts = max_pts * 0.5
    elif change_pct < 10:
        pts = max_pts * 0.25
    else:
        pts = 0.0

    reason = f"近{n_days}個交易日融資餘額變化 {change_pct:+.2f}%（下降代表籌碼較乾淨）"
    return {"points": pts, "reason": reason, "change_pct": change_pct}


# ----------------------------------------------------------------------
# 左側交易「出量創高後拉回季線」（20分）
# ----------------------------------------------------------------------
def score_left_side_pullback(setup: dict) -> dict:
    """左側交易拉回買點（20分）。找出「近期量增創高、目前拉回到季線附近或以下」的
    股票，並依下列三個面向給分：
      - 40%：型態本身成立（量增創高 + 已開始拉回，這是 applicable=True 的前提，固定給分）
      - 30%：是否已經拉回到季線(60日線)附近或以下（左側交易員傾向在這個位置分批承接）
      - 30%：目前拉回幅度 對比 這檔股票「歷史慣性拉回深度」的比例——越接近或超過
             歷史慣性拉回深度，統計上代表這次拉回已經跌到這檔股票「過去通常會反彈」
             的深度，向上反彈的機率相對較高。
    不符合「量增創高後拉回」型態的股票，此項給0分（不是每檔股票都適用這個特定策略）。
    """
    max_pts = config.SCORE_LEFT_SIDE_MAX
    if not setup or not setup.get("applicable"):
        reason = (setup or {}).get("reason", "近期未出現「量增創高後拉回」型態，此項不適用")
        return {"points": 0.0, "reason": reason}

    pts = max_pts * 0.4  # 型態成立的基礎分

    if setup.get("near_or_below_ma"):
        pts += max_pts * 0.3

    cur = setup.get("current_pullback_pct", np.nan)
    typical = setup.get("typical_pullback_pct", np.nan)
    if pd.notna(cur) and pd.notna(typical) and typical > 0:
        ratio = cur / typical
        if ratio >= 1.0:
            pts += max_pts * 0.3
        elif ratio >= 0.7:
            pts += max_pts * 0.2
        elif ratio >= 0.4:
            pts += max_pts * 0.1

    return {"points": round(pts, 2), "reason": setup.get("reason", "")}


# ----------------------------------------------------------------------
# 布林評級（15分）—— 依使用者提供的準則 a~h 與範例 1~5 綜合判斷
# ----------------------------------------------------------------------
def score_bollinger(boll: dict, granville: dict, institution_net_sign: int) -> dict:
    max_pts = config.SCORE_BOLLINGER_MAX
    band_state = boll.get("band_state", "unknown")
    position = boll.get("position", "unknown")
    is_expanding = boll.get("is_expanding", False)
    was_squeezed = boll.get("was_squeezed", False)
    bandwidth = boll.get("bandwidth", np.nan)

    if band_state == "unknown" or position == "unknown":
        return {"points": max_pts * 0.4, "reason": "布林通道資料不足，暫給中性偏低分數", "band_state": band_state, "position": position}

    pts, reason = max_pts * 0.4, "布林型態中性，無明顯訊號"

    # 範例 1/2：帶寬窄、開喇叭、沿上軌
    if band_state == "narrow" and was_squeezed and is_expanding and position == "upper":
        if institution_net_sign > 0:
            pts, reason = max_pts, "帶寬窄後開喇叭，股價沿上軌上攻，且三大法人同步買超 -> 最強買點型態(範例2)"
        else:
            pts, reason = max_pts * 0.87, "帶寬窄後開喇叭，股價沿上軌上攻 -> 強勢買點型態(範例1)"
    # 範例4：大帶寬、法人買、股價在下軌 -> 超跌轉機
    elif band_state == "wide" and institution_net_sign > 0 and position == "lower":
        pts, reason = max_pts * 0.8, "帶寬已大，三大法人買超，股價貼近下軌 -> 超跌轉機買點型態(範例4)"
    # 範例3：股價在上通道與月線之間震盪、多頭排列
    elif position in ("upper", "middle") and band_state in ("normal", "wide") and granville.get("signal") in ("buy_breakout", "buy_support"):
        pts, reason = max_pts * 0.67, "股價於布林上通道與月線之間震盪，符合多頭排列格局 -> 良好趨勢型態(範例3)"
    # 範例5：大帶寬、法人賣、股價在上軌 -> 賣點示警
    elif band_state == "wide" and institution_net_sign < 0 and position == "upper":
        pts, reason = max_pts * 0.13, "帶寬已大，三大法人賣超，股價卻在上軌，留意反轉賣壓 -> 賣點示警型態(範例5)"
    # 帶寬窄、尚未開喇叭 -> 盤整待變
    elif band_state == "narrow" and not is_expanding:
        pts, reason = max_pts * 0.5, "布林帶寬收縮、尚未開喇叭，屬盤整待變階段"

    # g/h 一般性風控準則：開喇叭但主力賣超，屬於應避開的型態，強制降分
    if is_expanding and institution_net_sign < 0:
        pts = min(pts, max_pts * 0.17)
        reason = "布林剛開喇叭，但三大法人為賣超方向，屬於應避開的型態（準則h）"

    bw_txt = f"{bandwidth*100:.1f}%" if pd.notna(bandwidth) else "N/A"
    reason_full = f"{reason}（帶寬{bw_txt}, 型態={band_state}, 位置={position}, 開喇叭={'是' if is_expanding else '否'}）"
    return {"points": pts, "reason": reason_full, "band_state": band_state, "position": position}


# ----------------------------------------------------------------------
# 葛蘭碧法則（15分）
# ----------------------------------------------------------------------
GRANVILLE_SCORE_MAP = {
    "buy_breakout": 1.0,
    "buy_support": 0.8,
    "neutral": 0.5,
    "sell_overbought": 0.2,
    "sell_breakdown": 0.1,
    "sell_rebound": 0.1,
}


def score_granville(granville: dict) -> dict:
    max_pts = config.SCORE_GRANVILLE_MAX
    ratio = GRANVILLE_SCORE_MAP.get(granville.get("signal", "neutral"), 0.5)
    pts = max_pts * ratio
    return {"points": pts, "reason": granville.get("reason", "")}


# ----------------------------------------------------------------------
# 彙總
# ----------------------------------------------------------------------
def compute_total_score(sub_scores: dict) -> dict:
    """sub_scores 需含 big_holder, holder_count, institution, margin, left_side, bollinger, granville
    七個子分數 dict。總分100 = 籌碼50(big_holder20+holder_count10+institution10+margin10)
    + 左側交易拉回20 + 布林15 + 葛蘭碧15。"""
    chip_pts = (
        sub_scores["big_holder"]["points"]
        + sub_scores["holder_count"]["points"]
        + sub_scores["institution"]["points"]
        + sub_scores["margin"]["points"]
    )
    left_side_pts = sub_scores["left_side"]["points"]
    boll_pts = sub_scores["bollinger"]["points"]
    granville_pts = sub_scores["granville"]["points"]
    total = chip_pts + left_side_pts + boll_pts + granville_pts

    reasons = [
        sub_scores["big_holder"]["reason"],
        sub_scores["holder_count"]["reason"],
        sub_scores["institution"]["reason"],
        sub_scores["margin"]["reason"],
        sub_scores["left_side"]["reason"],
        sub_scores["bollinger"]["reason"],
        sub_scores["granville"]["reason"],
    ]
    return {
        "total_score": round(total, 1),
        "chip_score": round(chip_pts, 1),
        "left_side_score": round(left_side_pts, 1),
        "bollinger_score": round(boll_pts, 1),
        "granville_score": round(granville_pts, 1),
        "reasons": reasons,
    }
