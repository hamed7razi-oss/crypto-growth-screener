"""
Crypto Growth Screener
------------------------
شناسایی کوین‌های میان‌رده (رتبه ۵۰-۵۰۰ مارکت‌کپ) با پتانسیل رشد
بر اساس ترکیب چهار معیار: مومنتوم قیمتی، Breakout، RSI، MACD

منابع داده:
- CoinGecko API  -> لیست کوین‌ها، رتبه مارکت‌کپ، و کندل‌های OHLC (رایگان، بدون کلید)

توجه: در نسخه‌های قبلی از Binance API برای کندل استفاده می‌شد، اما چون سرورهای
GitHub Actions معمولاً توسط Binance به‌خاطر محدودیت جغرافیایی مسدود می‌شوند،
این نسخه به‌طور کامل از CoinGecko OHLC API استفاده می‌کند.

خروجی: ارسال لیست رتبه‌بندی‌شده به تلگرام
"""

import os
import json
import time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# تنظیمات قابل تغییر
# ---------------------------------------------------------------------------
MIN_MARKET_CAP_RANK = 50
MAX_MARKET_CAP_RANK = 500
OHLC_DAYS = 30                 # بازه کندل (30 روز -> کندل‌های ~4 ساعته توسط CoinGecko)
TOP_N_RESULTS = 10             # چند تا کوین برتر در گزارش نهایی نشون داده بشه
MIN_SCORE_TO_REPORT = 50       # حداقل امتیاز از 100 برای اینکه در لیست باشه
REQUEST_DELAY = 1.5            # تاخیر بین درخواست‌ها به CoinGecko (جلوگیری از rate limit)
MAX_RETRIES = 3                # تعداد تلاش مجدد در صورت خطای 429 (rate limit)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# .strip() برای جلوگیری از خطای احتمالی به‌خاطر فاصله/خط اضافه هنگام کپی توکن در گیت‌هاب
TELEGRAM_BOT_TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()


# ---------------------------------------------------------------------------
# ۱) گرفتن کوین‌های رتبه ۵۰-۵۰۰ از CoinGecko
# ---------------------------------------------------------------------------
def get_market_universe():
    """
    لیست کوین‌های میان‌رده رو از CoinGecko می‌گیره.
    خروجی: لیستی از دیکشنری با symbol و coingecko id
    """
    coins = []
    # هر صفحه 250 تا کوین داره، صفحه 1 و 2 رو می‌گیریم تا رتبه 1-500 پوشش داده بشه
    for page in [1, 2]:
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false",
        }
        resp = requests.get(f"{COINGECKO_BASE}/coins/markets", params=params, timeout=15)
        resp.raise_for_status()
        coins.extend(resp.json())
        time.sleep(REQUEST_DELAY)

    filtered = [
        c for c in coins
        if c.get("market_cap_rank")
        and MIN_MARKET_CAP_RANK <= c["market_cap_rank"] <= MAX_MARKET_CAP_RANK
    ]
    return filtered


# ---------------------------------------------------------------------------
# ۲) گرفتن کندل قیمتی (OHLC) از CoinGecko
# ---------------------------------------------------------------------------
def get_ohlc(coin_id):
    """
    کندل‌های OHLC رو از CoinGecko می‌گیره.
    خروجی: DataFrame با ستون‌های high, low, close
    اگه داده کافی نبود، None برمی‌گردونه.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": OHLC_DAYS}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                # rate limit -> کمی صبر کن و دوباره امتحان کن
                time.sleep(5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                return None

            data = resp.json()
            if not data or len(data) < 20:
                return None

            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            return df
        except requests.RequestException:
            time.sleep(2)
            continue

    return None


# ---------------------------------------------------------------------------
# ۳) محاسبه اندیکاتورها
# ---------------------------------------------------------------------------
def calculate_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_macd(closes, fast=12, slow=26, signal=9):
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    # کراس صعودی: هیستوگرام از منفی به مثبت رفته
    bullish_cross = histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0
    return {
        "macd": macd_line.iloc[-1],
        "signal": signal_line.iloc[-1],
        "histogram": histogram.iloc[-1],
        "bullish_cross": bullish_cross,
    }


def detect_momentum(df):
    """
    مومنتوم قیمتی: تغییرات کندل‌های اخیر (کوتاه‌مدت و میان‌مدت)
    توجه: چون CoinGecko OHLC رایگان داده حجم نمی‌ده، این نسخه فقط بر پایه قیمته.
    """
    closes = df["close"]

    change_short = (closes.iloc[-1] / closes.iloc[-4] - 1) * 100 if len(closes) > 4 else 0
    change_long = (closes.iloc[-1] / closes.iloc[-8] - 1) * 100 if len(closes) > 8 else 0

    return {
        "change_short_pct": change_short,
        "change_long_pct": change_long,
    }


def detect_breakout(df, lookback=20):
    """
    تشخیص شکست مقاومت: آیا قیمت فعلی از سقف N کندل قبلی (به‌جز کندل آخر) عبور کرده.
    """
    if len(df) < lookback + 1:
        lookback = len(df) - 1

    recent_high = df["high"].iloc[-(lookback + 1):-1].max()
    current_close = df["close"].iloc[-1]

    is_breakout = current_close > recent_high

    return {
        "is_breakout": is_breakout,
        "resistance_level": recent_high,
    }


# ---------------------------------------------------------------------------
# ۴) امتیازدهی ترکیبی (جمعاً از 100)
# ---------------------------------------------------------------------------
def score_coin(df):
    closes = df["close"]

    momentum = detect_momentum(df)
    breakout = detect_breakout(df)
    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)

    score = 0
    reasons = []

    # --- مومنتوم قیمتی (35 امتیاز) ---
    if momentum["change_short_pct"] > 3 and momentum["change_long_pct"] > 5:
        score += 35
        reasons.append("مومنتوم صعودی پیوسته")
    elif momentum["change_short_pct"] > 0 and momentum["change_long_pct"] > 0:
        score += 18
        reasons.append("مومنتوم صعودی ضعیف")

    # --- Breakout (25 امتیاز) ---
    if breakout["is_breakout"]:
        score += 25
        reasons.append("شکست مقاومت اخیر")

    # --- RSI (20 امتیاز) ---
    if rsi is not None:
        if 50 <= rsi <= 70:
            score += 20
            reasons.append(f"RSI در محدوده سالم صعودی ({rsi:.0f})")
        elif 40 <= rsi < 50:
            score += 10
            reasons.append(f"RSI در حال خروج از منطقه خنثی ({rsi:.0f})")
        elif rsi > 70:
            score += 5
            reasons.append(f"RSI اشباع خرید - احتیاط ({rsi:.0f})")

    # --- MACD (20 امتیاز) ---
    if macd["bullish_cross"]:
        score += 20
        reasons.append("کراس صعودی MACD")
    elif macd["histogram"] > 0:
        score += 10
        reasons.append("هیستوگرام MACD مثبت")

    return {
        "score": round(score, 1),
        "rsi": round(rsi, 1) if rsi is not None else None,
        "macd_histogram": round(macd["histogram"], 5),
        "change_7candles_pct": round(momentum["change_long_pct"], 2),
        "breakout": breakout["is_breakout"],
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# ۵) ارسال نتیجه به تلگرام
# ---------------------------------------------------------------------------
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ توکن یا Chat ID تلگرام تنظیم نشده. پیام ارسال نشد.")
        print(text)
        return

    # بررسی صحت فرمت توکن قبل از ارسال (باید شبیه 123456:ABC... باشه)
    if ":" not in TELEGRAM_BOT_TOKEN:
        print(f"⚠️ فرمت TELEGRAM_BOT_TOKEN نامعتبر به‌نظر می‌رسد (طول={len(TELEGRAM_BOT_TOKEN)}). "
              f"لطفاً مقدار Secret را در گیت‌هاب دوباره بررسی کنید.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # تلگرام محدودیت طول پیام داره (4096 کاراکتر)، در صورت نیاز تقسیم می‌کنیم
    max_len = 4000
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"خطا در ارسال تلگرام: {resp.text}")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# ۵.۵) ذخیره اسنپ‌شات سیگنال‌ها (برای خواندن خودکار توسط پروژه ردیاب عملکرد)
# ---------------------------------------------------------------------------
def save_signal_snapshot(results):
    """
    نتایج این اجرا رو (فقط نماد، امتیاز، قیمت) تو پوشه signals/ ذخیره می‌کنه
    تا پروژه جداگانه‌ی ردیاب عملکرد بتونه خودکار بخونتش.
    """
    if not results:
        return
    os.makedirs("signals", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    payload = [
        {"symbol": r["symbol"], "coin_id": r["coin_id"], "score": r["score"], "price": r["price"]}
        for r in results[:TOP_N_RESULTS]
    ]
    with open(f"signals/{ts}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"سیگنال‌ها ذخیره شد: signals/{ts}.json")


def format_report(results):
    if not results:
        return "🔍 <b>گزارش اسکرینر کریپتو</b>\n\nهیچ کوینی امتیاز کافی کسب نکرد."

    lines = [f"🔍 <b>گزارش اسکرینر کریپتو</b>", f"کوین‌های میان‌رده با بالاترین پتانسیل رشد:\n"]

    for i, r in enumerate(results[:TOP_N_RESULTS], 1):
        lines.append(
            f"{i}. <b>{r['symbol']}</b> — امتیاز: {r['score']}/100\n"
            f"   قیمت: ${r['price']:.4f} | تغییر اخیر: {r['change_7candles_pct']}%\n"
            f"   RSI: {r['rsi']} | شکست مقاومت: {'بله' if r['breakout'] else 'خیر'}\n"
            f"   دلایل: {', '.join(r['reasons'])}\n"
        )

    lines.append("\n⚠️ این لیست صرفاً برای بررسی بیشتر است، نه توصیه سرمایه‌گذاری.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ۶.۵) ذخیره گزارش در signals/all_signals.json (برای خواندن خودکار توسط پروژه ردیاب)
# ---------------------------------------------------------------------------
def save_signal_record(results):
    if not results:
        return
    signal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coins": [
            {"symbol": r["symbol"], "score": r["score"], "price": r["price"]}
            for r in results[:TOP_N_RESULTS]
        ],
    }
    os.makedirs("signals", exist_ok=True)
    path = "signals/all_signals.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(signal)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ۷) اجرای اصلی
# ---------------------------------------------------------------------------
def main():
    print("در حال دریافت لیست کوین‌های میان‌رده از CoinGecko...")
    universe = get_market_universe()
    print(f"{len(universe)} کوین در محدوده رتبه {MIN_MARKET_CAP_RANK}-{MAX_MARKET_CAP_RANK} یافت شد.")

    results = []
    checked = 0
    skipped = 0

    total = len(universe)
    for i, coin in enumerate(universe, 1):
        symbol = coin["symbol"].upper()
        coin_id = coin["id"]

        df = get_ohlc(coin_id)
        time.sleep(REQUEST_DELAY)

        if i % 50 == 0:
            print(f"پیشرفت: {i}/{total} کوین بررسی شد...")

        if df is None:
            skipped += 1
            continue

        checked += 1
        try:
            analysis = score_coin(df)
        except Exception as e:
            print(f"خطا در تحلیل {symbol}: {e}")
            continue

        if analysis["score"] >= MIN_SCORE_TO_REPORT:
            results.append({
                "symbol": symbol,
                "coin_id": coin_id,
                "name": coin.get("name"),
                "price": coin.get("current_price", 0),
                "market_cap_rank": coin.get("market_cap_rank"),
                **analysis,
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nبررسی شد: {checked} کوین (داده کندل موجود بود)")
    print(f"رد شد: {skipped} کوین (داده کندل کافی از CoinGecko دریافت نشد)")
    print(f"واجد شرایط (امتیاز >= {MIN_SCORE_TO_REPORT}): {len(results)} کوین\n")

    report = format_report(results)
    print(report)
    send_telegram_message(report)
    save_signal_snapshot(results)
    save_signal_record(results)


if __name__ == "__main__":
    main()
