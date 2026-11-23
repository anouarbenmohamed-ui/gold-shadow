# -*- coding: utf-8 -*-
"""
GOLD ELITE v3.2 - SHADOW MODE
- نفس منطق v3.2 (4 نماذج ML + Golden Range + شمعة M30 مغلقة + سعر Quote حديث).
- لا يرسل أي رسالة Telegram.
- يسجل كل إشارة ويتابعها حتى TP أو SL.
- يكتب فقط داخل مجلد shadow_data/ ولا يلمس أي ملف آخر.
- تم إصلاح خطأ المتغير quote الذي كان يمنع الإشارات.
"""

import os
import re
import csv
import json
import datetime
import requests
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")

SYMBOL = "XAU/USD"
INTERVAL = "30min"
MAX_CANDLE_AGE_MINUTES = 45
MAX_QUOTE_AGE_SECONDS = 180

DATA_DIR = "shadow_data"
TRADES_FILE = os.path.join(DATA_DIR, "shadow_v32_trades.json")
CSV_FILE = os.path.join(DATA_DIR, "shadow_v32_trades.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "shadow_v32_summary.json")

UTC = datetime.timezone.utc


# ---------------------------------------------------------------------
# Helpers

def utc_now():
    return datetime.datetime.now(UTC)


def to_utc_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_trades():
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"⚠️ Cannot read {TRADES_FILE}: {exc}")
        return []


def save_trades(trades):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = TRADES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)
    os.replace(tmp, TRADES_FILE)


PRE_FILTER_DELAY_MIN = 15      # إشارات متأخرة جداً (تشغيل يدوي) تُستبعد من تحليل الفلاتر


def compute_reversal_flag(trades, trade_type, now_str):
    """True إذا كانت الإشارة السابقة بعكس الاتجاه قبل ساعة أو أقل."""
    if not trades:
        return False
    prev = max(trades, key=lambda t: t.get("open_time") or "")
    try:
        gap_min = (pd.Timestamp(now_str) - pd.Timestamp(prev["open_time"])).total_seconds() / 60
    except Exception:
        return False
    return bool(prev.get("type") != trade_type and gap_min <= 66)


def enrich(trades):
    """يكمل الحقول المشتقة للصفقات القديمة (في الذاكرة فقط، لا يغيّر ملف JSON)."""
    ordered = sorted(trades, key=lambda t: t.get("open_time") or "")
    prev = None
    for t in ordered:
        if t.get("max_conf") is None:
            vals = re.findall(r":(\d+(?:\.\d+)?)%", t.get("models") or "")
            t["max_conf"] = max(float(v) for v in vals) if vals else None
            t["n_models"] = len(vals)
        if t.get("risk_pts") is None:
            try:
                t["risk_pts"] = round(abs(float(t["entry"]) - float(t["sl"])), 2)
            except Exception:
                t["risk_pts"] = None
        if "reversal_1h" not in t:
            flag = False
            if prev is not None:
                try:
                    gap = (pd.Timestamp(t["open_time"]) - pd.Timestamp(prev["open_time"])).total_seconds() / 60
                    flag = bool(prev.get("type") != t.get("type") and gap <= 66)
                except Exception:
                    flag = False
            t["reversal_1h"] = flag
        # نسبة المسافة التي قطعها السعر نحو الهدف (mfe)، ونحو الوقف (mae)
        try:
            if t.get("mfe") is not None:
                reward = abs(float(t["tp"]) - float(t["entry"]))
                t["mfe_pct_tp"] = round(float(t["mfe"]) / reward * 100, 1) if reward else None
            if t.get("mae") is not None and t.get("risk_pts"):
                t["mae_pct_risk"] = round(float(t["mae"]) / float(t["risk_pts"]) * 100, 1)
        except Exception:
            pass
        prev = t


def _conf70(t):
    return (t.get("max_conf") or 0) > 70


def _risk20(t):
    return t.get("risk_pts") is not None and t["risk_pts"] <= 20


def _no_rev(t):
    return t.get("reversal_1h") is False


def _trend(t):
    return t.get("with_trend") is True


# الفلاتر المرشحة: تُحسب نتائجها افتراضياً على نفس الصفقات (لا تغيّر الصفقات المسجلة)
RULES = {
    "baseline": lambda t: True,
    "conf_gt_70": _conf70,
    "risk_le_20": _risk20,
    "no_reversal_1h": _no_rev,
    "with_trend_sma200": _trend,
    "conf70_and_risk20": lambda t: _conf70(t) and _risk20(t),
    "conf70_and_trend": lambda t: _conf70(t) and _trend(t),
    "conf70_risk20_trend": lambda t: _conf70(t) and _risk20(t) and _trend(t),
}


def rule_stats(closed, rule):
    sub = [t for t in closed if rule(t)]
    wins = [t for t in sub if t["status"] == "WIN"]
    r_total = sum((t.get("points") or 0) / t["risk_pts"] for t in sub if t.get("risk_pts"))
    return {
        "closed": len(sub),
        "wins": len(wins),
        "win_rate_pct": round(len(wins) / len(sub) * 100, 1) if sub else None,
        "R": round(r_total, 2),
        "points": round(sum(t.get("points") or 0 for t in sub), 2),
    }


def export_csv_and_summary(trades):
    """يكتب CSV يفتح في Excel + ملخص الأداء + نتائج الفلاتر المرشحة."""
    os.makedirs(DATA_DIR, exist_ok=True)
    enrich(trades)
    fields = [
        "id", "type", "strategy", "entry", "sl", "tp", "status",
        "open_time", "close_time", "close_price", "points",
        "both_touched", "signal_candle_time", "delay_minutes",
        "quote_age_seconds", "adx", "models",
        "with_trend", "signal_close", "sma50", "sma200",
        "max_conf", "n_models", "reversal_1h", "risk_pts", "mfe", "mae",
        "mfe_pct_tp", "mae_pct_risk", "new_schema",
    ]
    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            writer.writerow(t)

    closed = [t for t in trades if t.get("status") in ("WIN", "LOSS")]
    wins = [t for t in closed if t["status"] == "WIN"]
    losses = [t for t in closed if t["status"] == "LOSS"]
    summary = {
        "updated_utc": utc_now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_signals": len(trades),
        "open": sum(1 for t in trades if t.get("status") == "OPEN"),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "total_points": round(sum(t.get("points") or 0 for t in closed), 2),
        "avg_delay_minutes": round(
            float(np.mean([t["delay_minutes"] for t in trades if t.get("delay_minutes") is not None])), 1
        ) if any(t.get("delay_minutes") is not None for t in trades) else None,
    }
    for label, flag in (("with_trend", True), ("against_trend", False)):
        sub = [t for t in closed if t.get("with_trend") is flag]
        sub_wins = [t for t in sub if t["status"] == "WIN"]
        summary[f"{label}_closed"] = len(sub)
        summary[f"{label}_win_rate_pct"] = round(len(sub_wins) / len(sub) * 100, 1) if sub else None
        summary[f"{label}_points"] = round(sum(t.get("points") or 0 for t in sub), 2)

    analysis = [
        t for t in closed
        if not (t.get("delay_minutes") is not None and t["delay_minutes"] > PRE_FILTER_DELAY_MIN)
    ]
    new_only = [t for t in analysis if t.get("new_schema")]
    # filters_all: كل الصفقات (فيها صفقات استُخرجت منها الفلاتر، فهي متفائلة)
    # filters_new_only: الصفقات المسجلة بعد إضافة الفلاتر فقط (اختبار حقيقي)
    summary["filters_all"] = {name: rule_stats(analysis, r) for name, r in RULES.items()}
    summary["filters_new_only"] = {name: rule_stats(new_only, r) for name, r in RULES.items()}
    # نتيجة كل استراتيجية (TREND اتجاهي / RANGE عرضي) وكل اتجاه (BUY / SELL)
    def _group(rows, key_fn):
        keys = sorted({key_fn(t) for t in rows})
        return {k: rule_stats(rows, lambda t, k=k: key_fn(t) == k) for k in keys}

    def _strategy(t):
        return t.get("strategy") or "?"

    def _type(t):
        return t.get("type") or "?"

    def _strategy_type(t):
        return f"{_strategy(t)}_{_type(t)}"

    summary["by_strategy"] = _group(analysis, _strategy)
    summary["by_type"] = _group(analysis, _type)
    summary["by_strategy_type"] = _group(analysis, _strategy_type)
    summary["by_strategy_new_only"] = _group(new_only, _strategy)
    summary["by_strategy_type_new_only"] = _group(new_only, _strategy_type)
    summary["open_by_strategy"] = {
        k: sum(1 for t in trades if t.get("status") == "OPEN" and _strategy(t) == k)
        for k in sorted({_strategy(t) for t in trades})
    }

    ex = [t for t in analysis if t.get("mfe_pct_tp") is not None]
    ex_loss = [t for t in ex if t["status"] == "LOSS"]
    ex_win = [t for t in ex if t["status"] == "WIN" and t.get("mae_pct_risk") is not None]
    summary["excursions"] = {
        "trades_with_data": len(ex),
        "losses": len(ex_loss),
        "losses_reached_50pct_of_tp": sum(1 for t in ex_loss if t["mfe_pct_tp"] >= 50),
        "losses_reached_70pct_of_tp": sum(1 for t in ex_loss if t["mfe_pct_tp"] >= 70),
        "avg_mfe_pct_of_tp_in_losses": round(float(np.mean([t["mfe_pct_tp"] for t in ex_loss])), 1) if ex_loss else None,
        "wins": len(ex_win),
        "avg_mae_pct_of_risk_in_wins": round(float(np.mean([t["mae_pct_risk"] for t in ex_win])), 1) if ex_win else None,
        "wins_within_20pct_of_stop": sum(1 for t in ex_win if t["mae_pct_risk"] >= 80),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📈 Summary: {json.dumps({k: v for k, v in summary.items() if not k.startswith('filters')}, ensure_ascii=False)}")


# ---------------------------------------------------------------------
# Twelve Data

def fetch_data():
    if not TWELVE_DATA_KEY:
        print("❌ TWELVE_DATA_KEY is missing.")
        return None

    params = {
        "symbol": SYMBOL, "interval": INTERVAL, "outputsize": 1500,
        "apikey": TWELVE_DATA_KEY, "order": "ASC", "timezone": "UTC",
    }
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "ok" or "values" not in payload:
            print(f"❌ Twelve Data response: {str(payload)[:500]}")
            return None

        frame = pd.DataFrame(payload["values"])
        required = ["datetime", "open", "high", "low", "close"]
        if any(c not in frame.columns for c in required):
            print("❌ Missing OHLC fields.")
            return None

        for c in ["open", "high", "low", "close"]:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=required)
        frame = frame.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

        now = pd.Timestamp(utc_now())
        frame = frame[frame["datetime"] < now.floor("30min")].reset_index(drop=True)
        if frame.empty:
            print("❌ No closed candles.")
            return None

        last_end = frame["datetime"].iloc[-1] + pd.Timedelta(minutes=30)
        age = (now - last_end).total_seconds() / 60
        print(f"🕒 Last closed candle: {frame['datetime'].iloc[-1]} | age after close: {age:.1f} min")
        if age > MAX_CANDLE_AGE_MINUTES:
            print("🛑 Data too old; no signal.")
            return None
        return frame
    except Exception as exc:
        print(f"❌ Data fetch error: {exc}")
        return None


def fetch_live_quote():
    params = {"symbol": SYMBOL, "apikey": TWELVE_DATA_KEY, "timezone": "UTC"}
    try:
        r = requests.get("https://api.twelvedata.com/quote", params=params, timeout=20)
        r.raise_for_status()
        p = r.json()
        if p.get("status") == "error":
            print(f"⚠️ Quote error: {p.get('message')}")
            return None

        qt = p.get("last_quote_at")
        if not qt or not str(qt).isdigit():
            print("⚠️ Quote has no last_quote_at.")
            return None
        quote_ts = pd.Timestamp(int(qt), unit="s", tz="UTC")

        age = (pd.Timestamp(utc_now()) - quote_ts).total_seconds()
        if age < -30 or age > MAX_QUOTE_AGE_SECONDS:
            print(f"🛑 Quote stale/invalid ({age:.0f}s).")
            return None

        cv = p.get("close") or p.get("price")
        if cv is None:
            print("⚠️ Quote has no price.")
            return None
        close_price = float(cv)
        bid = float(p["bid"]) if p.get("bid") not in (None, "") else close_price
        ask = float(p["ask"]) if p.get("ask") not in (None, "") else close_price
        if bid <= 0 or ask <= 0 or ask < bid:
            print("⚠️ Invalid bid/ask.")
            return None

        return {
            "bid": bid, "ask": ask, "close": close_price,
            "time": quote_ts, "age_seconds": age,
            "has_bid_ask": p.get("bid") not in (None, "") and p.get("ask") not in (None, ""),
        }
    except Exception as exc:
        print(f"⚠️ Quote fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------
# Indicators (نفس v3.2)

def calculate_indicators(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / 14.0
    df["atr"] = tr.ewm(alpha=alpha, adjust=False, min_periods=14).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    plus_s = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    minus_s = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_s / (df["atr"] + 1e-10)
    minus_di = 100 * minus_s / (df["atr"] + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(alpha=alpha, adjust=False, min_periods=14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()

    df["sma20"] = close.rolling(20).mean()
    df["std20"] = close.rolling(20).std()
    df["upper_band"] = df["sma20"] + 2 * df["std20"]
    df["lower_band"] = df["sma20"] - 2 * df["std20"]

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-10)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


# ---------------------------------------------------------------------
# Trades

def can_open_new_trade(trade_type, entry, signal_candle_time):
    trades = load_trades()
    for t in trades:
        if t.get("signal_candle_time") == signal_candle_time:
            print("🚫 Duplicate signal candle.")
            return False
    for t in trades:
        if t.get("status") != "OPEN":
            continue
        if t.get("type") == trade_type:
            print(f"🚫 Direction lock: open {trade_type} (id={t.get('id')}).")
            return False
        try:
            if abs(float(entry) - float(t["entry"])) < 3.0:
                print(f"🚫 Open trade near {t['entry']}.")
                return False
        except (KeyError, TypeError, ValueError):
            continue
    return True


def manage_active_trades(df):
    """يتابع الصفقات المفتوحة. يبدأ من أول شمعة كاملة بعد الدخول.
    mfe = أبعد حركة لصالح الصفقة، mae = أبعد حركة ضدها (بالنقاط).
    شمعة الإغلاق: ترتيب الحركة داخلها مجهول، فلا نحتسب منها إلا ما هو مؤكد:
    - خسارة: mae = مسافة الوقف، و mfe من الشموع السابقة فقط (حد أدنى).
    - ربح:   mfe = مسافة الهدف، و mae من الشموع السابقة فقط (حد أدنى).
    """
    trades = load_trades()
    changed = False

    for t in trades:
        if t.get("status") != "OPEN":
            continue
        try:
            open_time = to_utc_timestamp(t["open_time"])
            first_safe = open_time.ceil("30min")
        except Exception:
            print(f"⚠️ Bad open_time for id={t.get('id')}")
            continue

        tp, sl, entry = float(t["tp"]), float(t["sl"]), float(t["entry"])
        risk, reward = abs(entry - sl), abs(tp - entry)
        mfe, mae = 0.0, 0.0
        for _, c in df[df["datetime"] >= first_safe].iterrows():
            if t["type"] == "BUY":
                hit_tp, hit_sl = c["high"] >= tp, c["low"] <= sl
                fav, adv = float(c["high"]) - entry, entry - float(c["low"])
            else:
                hit_tp, hit_sl = c["low"] <= tp, c["high"] >= sl
                fav, adv = entry - float(c["low"]), float(c["high"]) - entry
            if not (hit_tp or hit_sl):
                mfe, mae = max(mfe, fav), max(mae, adv)
                continue

            # لمس الهدف والوقف في نفس الشمعة => خسارة تحفظياً
            result = "LOSS" if hit_sl else "WIN"
            exit_price = sl if result == "LOSS" else tp
            if result == "LOSS":
                mae = risk
            else:
                mfe = reward
            t["status"] = result
            t["close_price"] = exit_price
            t["close_time"] = c["datetime"].strftime("%Y-%m-%d %H:%M UTC")
            t["both_touched"] = bool(hit_tp and hit_sl)
            t["points"] = round(
                (exit_price - entry) if t["type"] == "BUY" else (entry - exit_price), 2
            )
            changed = True
            print(f"{'🎯' if result == 'WIN' else '🛑'} Trade id={t['id']} {t['type']} -> {result}")
            break

        new_mfe, new_mae = round(mfe, 2), round(mae, 2)
        if t.get("mfe") != new_mfe or t.get("mae") != new_mae:
            t["mfe"], t["mae"] = new_mfe, new_mae
            changed = True

    if changed:
        save_trades(trades)


# ---------------------------------------------------------------------
# Market / news filters

def is_market_truly_open():
    now = utc_now()
    wd, h = now.weekday(), now.hour
    if wd == 5:
        print("⏳ Market closed (Saturday).")
        return False
    if wd == 6 and h < 23:
        print("⏳ Market closed (before Sunday open).")
        return False
    if wd == 4 and h >= 22:
        print("⏳ Market closed (after Friday close).")
        return False
    return True


def check_economic_news():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        if r.status_code != 200:
            return False
        now = utc_now()
        for ev in r.json():
            if ev.get("country") != "USD" or ev.get("impact") != "High":
                continue
            dv = ev.get("date")
            if not dv:
                continue
            et = datetime.datetime.fromisoformat(dv.replace("Z", "+00:00"))
            if et.tzinfo is None:
                et = et.replace(tzinfo=UTC)
            if -30 <= (et - now).total_seconds() / 60 <= 30:
                print(f"⚠️ News filter: {ev.get('title')}")
                return True
    except Exception as exc:
        print(f"⚠️ News calendar unavailable: {exc}")
    return False


# ---------------------------------------------------------------------
# Signal detection (نفس منطق v3.2)

def resolve_tie_break(model_results, df):
    buy_p = [v["prob"] for v in model_results.values() if v["pred"] == 1]
    sell_p = [v["prob"] for v in model_results.values() if v["pred"] == 0]
    avg_buy = float(np.mean(buy_p)) if buy_p else 0.0
    avg_sell = float(np.mean(sell_p)) if sell_p else 0.0
    close, sma50 = float(df["close"].iloc[-1]), float(df["sma50"].iloc[-1])
    if avg_buy > avg_sell and close > sma50:
        return 1
    if avg_sell > avg_buy and close < sma50:
        return 0
    return 1 if avg_buy >= avg_sell else 0


def detect_signal(df, adx):
    """يرجع (final, strategy, confidences) أو (None, None, {})."""
    if adx >= 25:
        print("📈 Trend mode: 4 ML models.")
        d = df.copy()
        d["target"] = np.where(d["close"].shift(-1) > d["close"], 1, 0)
        feats = ["atr", "rsi", "macd", "macd_signal", "sma50", "sma200", "adx"]
        train_x, train_y = d[feats].iloc[:-1], d["target"].iloc[:-1]
        latest_x = d[feats].iloc[[-1]]
        if train_y.nunique() < 2:
            print("⚠️ Single target class; skipping.")
            return None, None, {}

        scaler = StandardScaler()
        train_s = scaler.fit_transform(train_x)
        latest_s = scaler.transform(latest_x)

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "SVM": SVC(probability=True, random_state=42),
        }
        results, votes, confs = {}, [], {}
        for name, m in models.items():
            if name in ("LogisticRegression", "SVM"):
                m.fit(train_s, train_y)
                pred = int(m.predict(latest_s)[0])
                prob = float(m.predict_proba(latest_s)[0][pred])
            else:
                m.fit(train_x, train_y)
                pred = int(m.predict(latest_x)[0])
                prob = float(m.predict_proba(latest_x)[0][pred])
            print(f"🤖 {name} | {'BUY' if pred == 1 else 'SELL'} | {prob*100:.1f}%")
            results[name] = {"pred": pred, "prob": prob * 100}
            if prob >= 0.60:
                votes.append(pred)
                confs[name] = round(prob * 100, 1)

        b, s = votes.count(1), votes.count(0)
        print(f"🗳️ Votes BUY={b} SELL={s}")
        final = None
        if b >= 3:
            final = 1
        elif s >= 3:
            final = 0
        elif b == 2 and s == 2:
            final = resolve_tie_break(results, df)
        elif b > s:
            final = 1
        elif s > b:
            final = 0
        return (final, "TREND", confs) if final is not None else (None, None, {})

    if adx < 20:
        print("⚖️ Range mode.")
        row = df.iloc[-1]
        price, lower, upper, stoch = (
            float(row["close"]), float(row["lower_band"]),
            float(row["upper_band"]), float(row["stoch_k"]),
        )
        if price <= lower and stoch < 15:
            return 1, "RANGE", {}
        if price >= upper and stoch > 85:
            return 0, "RANGE", {}
        return None, None, {}

    print("ℹ️ ADX grey zone (20-25); no signal.")
    return None, None, {}


# ---------------------------------------------------------------------
# Main

def main():
    if not TWELVE_DATA_KEY:
        print("❌ TWELVE_DATA_KEY is missing.")
        return

    now = utc_now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    market_open = is_market_truly_open()

    raw = fetch_data()
    if raw is None or len(raw) < 250:
        print("❌ Insufficient or stale data.")
        return

    df = calculate_indicators(raw)
    if len(df) < 200:
        print("❌ Insufficient rows after indicators.")
        return

    # متابعة الصفقات المفتوحة دائماً
    manage_active_trades(df)
    export_csv_and_summary(load_trades())

    if not market_open:
        return
    if check_economic_news():
        print("🛑 High-impact USD news; no scan.")
        return

    candle_time = df["datetime"].iloc[-1]
    signal_candle_time = candle_time.isoformat()
    adx = float(df["adx"].iloc[-1])
    atr = float(df["atr"].iloc[-1])
    print(f"📊 ADX={adx:.2f} | closed candle={signal_candle_time}")

    final, strategy, confs = detect_signal(df, adx)
    if final is None:
        print("ℹ️ No qualifying signal.")
        return

    quote = fetch_live_quote()
    if quote is None:
        print("🛑 No fresh quote; signal skipped.")
        return

    entry = quote["ask"] if final == 1 else quote["bid"]
    tp_mult = 2.0 if strategy == "TREND" else 1.2
    if final == 1:
        sl, tp = entry - 1.2 * atr, entry + tp_mult * atr
    else:
        sl, tp = entry + 1.2 * atr, entry - tp_mult * atr

    trade_type = "BUY" if final == 1 else "SELL"
    if not can_open_new_trade(trade_type, entry, signal_candle_time):
        return

    delay_min = (pd.Timestamp(now) - (candle_time + pd.Timedelta(minutes=30))).total_seconds() / 60

    # فلتر الاتجاه (للتحليل فقط): شراء فوق SMA200، وبيع تحتها
    last = df.iloc[-1]
    sig_close, sma50, sma200 = float(last["close"]), float(last["sma50"]), float(last["sma200"])
    with_trend = bool((final == 1 and sig_close > sma200) or (final == 0 and sig_close < sma200))

    trades = load_trades()
    reversal_1h = compute_reversal_flag(trades, trade_type, now_str)
    trades.append({
        "id": max([int(t.get("id", 0)) for t in trades], default=0) + 1,
        "type": trade_type,
        "strategy": strategy,
        "entry": round(float(entry), 2),
        "sl": round(float(sl), 2),
        "tp": round(float(tp), 2),
        "status": "OPEN",
        "open_time": now_str,
        "close_time": None,
        "close_price": None,
        "points": None,
        "both_touched": False,
        "signal_candle_time": signal_candle_time,
        "delay_minutes": round(delay_min, 1),
        "quote_age_seconds": round(quote["age_seconds"]),
        "adx": round(adx, 2),
        "models": ", ".join(f"{k}:{v}%" for k, v in confs.items()),
        "with_trend": with_trend,
        "signal_close": round(sig_close, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "max_conf": max(confs.values()) if confs else None,
        "n_models": len(confs),
        "reversal_1h": reversal_1h,
        "risk_pts": round(abs(float(entry) - float(sl)), 2),
        "mfe": 0.0,
        "mae": 0.0,
        "new_schema": True,
    })
    save_trades(trades)
    export_csv_and_summary(trades)
    print(f"✅ Shadow signal recorded: {trade_type} @ {entry:.2f} | SL {sl:.2f} | TP {tp:.2f} | delay {delay_min:.1f} min | with_trend={with_trend}")


if __name__ == "__main__":
    main()
