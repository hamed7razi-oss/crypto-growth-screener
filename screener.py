"""
Crypto Growth Screener
------------------------
شناسایی کوین‌های میان‌رده (رتبه ۵۰-۵۰۰ مارکت‌کپ) با پتانسیل رشد
بر اساس ترکیب چهار معیار: مومنتوم قیمتی، Breakout، RSI، MACD

منابع داده:
- CoinGecko API  -> لیست کوین‌ها و رتبه مارکت‌کپ (رایگان، بدون کلید)
- Binance API    -> کندل‌های قیمتی برای محاسبه اندیکاتورها (رایگان، بدون کلید)

خروجی: ارسال لیست رتبه‌بندی‌شده به تلگرام
"""

import os
import time
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# تنظیمات قابل تغییر
# ---------------------------------------------------------------------------
MIN_MARKET_CAP_RANK = 50
MAX_MARKET_CAP_RANK = 500
KLINE_INTERVAL = "4h"          # تایم‌فریم کندل (مناسب افق میان‌مدت)
KLINE_LIMIT = 100              # تعداد کندل برای تحلیل (~16 روز در تایم‌فریم 4h)
TOP_N_RESULTS = 10             # چند تا کوین برتر در گزارش نهایی نشون داده بشه
MIN_SCORE_TO_REPORT = 50       # حداقل امتیاز از 100 برای اینکه در لیست باشه
REQUEST_DELAY = 0.3            # تاخیر بین درخواست‌ها به Binance (جلوگیری از rate limit)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BINANCE_BASE = "https://api.binance.com/api/v3"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


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
# ۲) گرفتن کندل قیمتی از Binance
# ---------------------------------------------------------------------------
def get_klines(binance_symbol):
    """
    کندل‌های قیمتی رو از Binance می‌گیره.
    خروجی: DataFrame با ستون‌های open, high, low, close, volume
    اگه جفت‌ارز روی Binance وجود نداشته باشه، None برمی‌گردونه.
    """
    params = {
        "symbol": binance_symbol,
        "interval": KLINE_INTERVAL,
        "limit": KLINE_LIMIT,
    }
    try:
        resp = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or len(data) < 30:
            return None

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except requests.RequestException:
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
    مومنتوم قیمتی: تغییرات ۳ و ۷ کندل اخیر + همراهی حجم
    """
    closes = df["close"]
    volumes = df["volume"]

    change_short = (closes.iloc[-1] / closes.iloc[-4] - 1) * 100 if len(closes) > 4 else 0
    change_long = (closes.iloc[-1] / closes.iloc[-8] - 1) * 100 if len(closes) > 8 else 0

    avg_volume_prev = volumes.iloc[-15:-5].mean()
    recent_volume = volumes.iloc[-5:].mean()
    volume_increase = (recent_volume / avg_volume_prev) if avg_volume_prev > 0 else 1

    return {
        "change_short_pct": change_short,
        "change_long_pct": change_long,
        "volume_increase_ratio": volume_increase,
    }


def detect_breakout(df, lookback=30):
    """
    تشخیص شکست مقاومت: آیا قیمت فعلی از سقف N کندل قبلی (به‌جز کندل آخر) عبور کرده،
    و آیا حجم کندل شکست بالاتر از میانگین بوده.
    """
    if len(df) < lookback + 1:
        lookback = len(df) - 1

    recent_high = df["high"].iloc[-(lookback + 1):-1].max()
    current_close = df["close"].iloc[-1]
    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].iloc[-(lookback + 1):-1].mean()

    is_breakout = current_close > recent_high
    volume_confirmed = current_volume > avg_volume * 1.3

    return {
        "is_breakout": is_breakout,
        "volume_confirmed": volume_confirmed,
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

    # --- مومنتوم (30 امتیاز) ---
    if momentum["change_short_pct"] > 3 and momentum["change_long_pct"] > 5:
        score += 15
        reasons.append("مومنتوم صعودی پیوسته")
    elif momentum["change_short_pct"] > 0 and momentum["change_long_pct"] > 0:
        score += 7
        reasons.append("مومنتوم صعودی ضعیف")

    if momentum["volume_increase_ratio"] > 1.5:
        score += 15
        reasons.append("افزایش قابل‌توجه حجم معاملات")
    elif momentum["volume_increase_ratio"] > 1.1:
        score += 7
        reasons.append("افزایش جزئی حجم")

    # --- Breakout (30 امتیاز) ---
    if breakout["is_breakout"] and breakout["volume_confirmed"]:
        score += 30
        reasons.append("شکست مقاومت با تایید حجم")
    elif breakout["is_breakout"]:
        score += 15
        reasons.append("شکست مقاومت بدون تایید حجم قوی")

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
        "volume_ratio": round(momentum["volume_increase_ratio"], 2),
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


def format_report(results):
    if not results:
        return "🔍 <b>گزارش اسکرینر کریپتو</b>\n\nهیچ کوینی امتیاز کافی کسب نکرد."

    lines = [f"🔍 <b>گزارش اسکرینر کریپتو</b>", f"کوین‌های میان‌رده با بالاترین پتانسیل رشد:\n"]

    for i, r in enumerate(results[:TOP_N_RESULTS], 1):
        lines.append(
            f"{i}. <b>{r['symbol']}</b> — امتیاز: {r['score']}/100\n"
            f"   قیمت: ${r['price']:.4f} | تغییر {KLINE_INTERVAL}×7: {r['change_7candles_pct']}%\n"
            f"   RSI: {r['rsi']} | حجم نسبت به میانگین: {r['volume_ratio']}x\n"
            f"   دلایل: {', '.join(r['reasons'])}\n"
        )

    lines.append("\n⚠️ این لیست صرفاً برای بررسی بیشتر است، نه توصیه سرمایه‌گذاری.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ۶) اجرای اصلی
# ---------------------------------------------------------------------------
def main():
    print("در حال دریافت لیست کوین‌های میان‌رده از CoinGecko...")
    universe = get_market_universe()
    print(f"{len(universe)} کوین در محدوده رتبه {MIN_MARKET_CAP_RANK}-{MAX_MARKET_CAP_RANK} یافت شد.")

    results = []
    checked = 0
    skipped = 0

    for coin in universe:
        symbol = coin["symbol"].upper()
        binance_symbol = f"{symbol}USDT"

        df = get_klines(binance_symbol)
        time.sleep(REQUEST_DELAY)

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
                "name": coin.get("name"),
                "price": coin.get("current_price", 0),
                "market_cap_rank": coin.get("market_cap_rank"),
                **analysis,
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nبررسی شد: {checked} کوین (روی Binance موجود بودن)")
    print(f"رد شد: {skipped} کوین (جفت‌ارز روی Binance موجود نبود)")
    print(f"واجد شرایط (امتیاز >= {MIN_SCORE_TO_REPORT}): {len(results)} کوین\n")

    report = format_report(results)
    print(report)
    send_telegram_message(report)


if __name__ == "__main__":
    main()
