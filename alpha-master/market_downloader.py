
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Market Downloader
- A-shares (Tushare -> fallback to AkShare)
- US stocks (Yahoo Finance with retry -> fallback to Stooq via pandas-datareader)
- Crypto (CCXT: try Binance -> OKX -> Bybit -> Gate -> Kraken)

Date range: 2024-01-01 to 2025-08-31
Output: ./data/{a_share|us|crypto}/*.csv
Schema: date, open, high, low, close, volume, source, symbol

Install (choose what you need):
  pip install pandas yfinance ccxt tushare akshare pandas-datareader

A股（Tushare）需要设置 Token：
  export TUSHARE_TOKEN="your_token_here"   # Linux/macOS
  $env:TUSHARE_TOKEN="your_token_here"     # Windows PowerShell
"""
import os

import time
from typing import List, Optional, Any, Iterable
import pandas as pd

# Optional imports
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

try:
    import ccxt
except Exception:
    ccxt = None

try:
    import tushare as ts
except Exception:
    ts = None

try:
    import akshare as ak
except Exception:
    ak = None

# ---------------------- CONFIG ----------------------
START_DATE = "2024-01-01"
END_DATE   = "2025-08-31"

OUTPUT_DIR = "data"
A_SHARE_DIR = os.path.join(OUTPUT_DIR, "a_share")
US_DIR      = os.path.join(OUTPUT_DIR, "us")
CRYPTO_DIR  = os.path.join(OUTPUT_DIR, "crypto")

# Provider preferences
A_SHARE_PROVIDER = "auto"  # "tushare" | "akshare" | "auto"
CRYPTO_EXCHANGES: List[str] = ["binance", "okx", "bybit", "gate", "kraken"]

# Retry settings for Yahoo
YF_MAX_RETRIES = 6
YF_BACKOFF_SEC = 3

# 预填：A股（ts_code）
A_SHARES: List[str] = [
    "600519.SH",  # 贵州茅台
    "600036.SH",  # 招商银行
    "601318.SH",  # 中国平安
    "600000.SH",  # 浦发银行
    "601398.SH",  # 工商银行
    "601988.SH",  # 中国银行
    "601857.SH",  # 中国石油
    "000001.SZ",  # 平安银行
    "000333.SZ",  # 美的集团
    "300750.SZ",  # 宁德时代
    "000002.SZ",  # 万科A
    "600104.SH",  # 上汽集团
]

# 预填：美股/ETF
US_TICKERS: List[str] = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","SPY","QQQ","BRK-B"
]

# 预填：Crypto 交易对
CRYPTO_SYMBOLS: List[str] = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
    "ADA/USDT","DOGE/USDT","LTC/USDT","DOT/USDT","AVAX/USDT",
]

# ----------------------------------------------------
def ensure_dirs():
    for d in [OUTPUT_DIR, A_SHARE_DIR, US_DIR, CRYPTO_DIR]:
        os.makedirs(d, exist_ok=True)

def normalize_df(df: pd.DataFrame, symbol: str, source: str, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns and not (df.index.name and df.index.name.lower() == date_col.lower()):
        raise ValueError(f"date column '{date_col}' not found for {symbol} ({source})")
    out = df.copy()
    out = out.rename(columns={
        # common
        "Date": "date", "date": "date", "trade_date": "date",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "vol": "volume", "volume": "volume",
        # akshare zh columns
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
        "成交量": "volume",
    })
    if "date" not in out.columns and out.index.name and out.index.name.lower() == date_col.lower():
        out = out.reset_index().rename(columns={date_col: "date"})
    keep = [c for c in ["date","open","high","low","close","volume"] if c in out.columns]
    out = out[keep]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    out["source"] = source
    out["symbol"] = symbol
    out = out.sort_values("date").reset_index(drop=True)
    return out

# ---------------------- A-SHARES ----------------------
def get_tushare_client() -> Optional[Any]:
    if ts is None:
        return None
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        return None
    ts.set_token(token)
    return ts.pro_api()

def fetch_a_share_tushare(pro, ts_code: str, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start).strftime("%Y%m%d")
    end_ts   = pd.Timestamp(end).strftime("%Y%m%d")
    df = pro.daily(ts_code=ts_code, start_date=start_ts, end_date=end_ts)
    # If permission denied, Tushare returns empty df or raises; let caller handle
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("trade_date").rename(columns={"trade_date": "date"})
    for col in ["open","high","low","close","vol"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return normalize_df(df, ts_code, "tushare", "date")

def fetch_a_share_akshare(symbol: str, start: str, end: str, adjust: str="") -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare not installed")
    # akshare uses numeric code without suffix, e.g. 600519 or 000001
    code = symbol.replace(".SH","").replace(".SZ","")
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e, adjust=adjust)
    if df is None or df.empty:
        return pd.DataFrame()
    return normalize_df(df, symbol, f"akshare{('/'+adjust) if adjust else ''}", "日期")

def fetch_a_share(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Try Tushare first (if available & token set), else fall back to AkShare
    if A_SHARE_PROVIDER in ("tushare", "auto"):
        pro = get_tushare_client()
        if pro is not None:
            try:
                df = fetch_a_share_tushare(pro, symbol, start, end)
                if not df.empty:
                    return df
            except Exception as e:
                # permission or other errors -> fall back
                pass
    if A_SHARE_PROVIDER in ("akshare", "auto"):
        try:
            return fetch_a_share_akshare(symbol, start, end, adjust="")  # 原始
        except Exception:
            # try qfq (前复权) as alternative
            try:
                return fetch_a_share_akshare(symbol, start, end, adjust="qfq")
            except Exception as e:
                raise e
    return pd.DataFrame()

# ---------------------- US STOCKS ----------------------
def fetch_us_stock_yf_with_retry(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    delay = YF_BACKOFF_SEC
    last_err = None
    for attempt in range(1, YF_MAX_RETRIES+1):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}).reset_index()
                return normalize_df(df, ticker, "yahoo", "Date")
        except Exception as e:
            last_err = e
            msg = str(e)
            if "Too Many Requests" in msg or "Rate limited" in msg or "429" in msg:
                time.sleep(delay)
                delay *= 2
                continue
            else:
                break
        # handle empty df (rate-limited w/o exception)
        time.sleep(delay); delay *= 2
    # Fallback returns empty; caller can try stooq
    return pd.DataFrame()

def fetch_us_stock_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    if pdr is None:
        return pd.DataFrame()
    try:
        df = pdr.DataReader(ticker, "stooq", start=pd.Timestamp(start), end=pd.Timestamp(end))
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}).sort_index()
        df = df.reset_index().rename(columns={"Date":"Date"})
        return normalize_df(df, ticker, "stooq", "Date")
    except Exception:
        return pd.DataFrame()

def fetch_us_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = fetch_us_stock_yf_with_retry(ticker, start, end)
    if df is not None and not df.empty:
        return df
    # Fallback to Stooq (daily data, may differ slightly)
    df = fetch_us_stock_stooq(ticker, start, end)
    return df

# ---------------------- CRYPTO (CCXT) ----------------------
def fetch_crypto_ccxt(symbol: str, start: str, end: str, timeframe: str, ex_id: str) -> pd.DataFrame:
    ex_class = getattr(ccxt, ex_id)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    if symbol not in ex.symbols:
        # try common alt formats
        candidates = [symbol, symbol.replace("-", "/"), symbol.replace("/", "-")]
        found = None
        for c in candidates:
            if c in ex.symbols:
                found = c; break
        if found is None:
            raise ValueError(f"{symbol} not listed on {ex_id}")
        symbol = found

    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").replace(hour=23, minute=59, second=59).timestamp() * 1000)

    all_rows = []
    limit = 1000
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + 1
        if next_since >= end_ms:
            break
        since = next_since
        time.sleep(ex.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame()

    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(None)
    df = df.drop(columns=["timestamp"])
    return normalize_df(df, symbol, ex_id, "date")

def fetch_crypto(symbol: str, start: str, end: str, timeframe: str="1d") -> pd.DataFrame:
    if ccxt is None:
        return pd.DataFrame()
    last_err = None
    for ex_id in CRYPTO_EXCHANGES:
        try:
            return fetch_crypto_ccxt(symbol, start, end, timeframe, ex_id)
        except Exception as e:
            last_err = e
            continue
    # All failed
    return pd.DataFrame()

# ---------------------- MAIN ----------------------
def main():
    ensure_dirs()

    # A-shares
    for code in A_SHARES:
        try:
            print(f"[A-share] Downloading {code}...")
            df = fetch_a_share(code, START_DATE, END_DATE)
            if df is None or df.empty:
                print(f"  !! Failed {code}: no data (check Tushare权限或安装 akshare)")
                continue
            out = os.path.join(A_SHARE_DIR, f"{code.replace('.', '_')}.csv")
            df.to_csv(out, index=False)
            print(f"  -> {out} ({len(df)} rows)")
        except Exception as e:
            print(f"  !! Failed {code}: {e}")

    # US stocks
    for t in US_TICKERS:
        try:
            print(f"[US] Downloading {t}...")
            df = fetch_us_stock(t, START_DATE, END_DATE)
            if df is None or df.empty:
                print(f"  !! Failed {t}: no data (Yahoo限流或Stooq无该代码)")
                continue
            out = os.path.join(US_DIR, f"{t}.csv")
            df.to_csv(out, index=False)
            print(f"  -> {out} ({len(df)} rows)")
        except Exception as e:
            print(f"  !! Failed {t}: {e}")

    # Crypto
    for s in CRYPTO_SYMBOLS:
        try:
            print(f"[Crypto] Downloading {s}...")
            df = fetch_crypto(s, START_DATE, END_DATE, timeframe="1d")
            if df is None or df.empty:
                print(f"  !! Failed {s}: no data from {', '.join(CRYPTO_EXCHANGES)}")
                continue
            safe = s.replace("/", "-")
            out = os.path.join(CRYPTO_DIR, f"{safe}.csv")
            df.to_csv(out, index=False)
            print(f"  -> {out} ({len(df)} rows)")
        except Exception as e:
            print(f"  !! Failed {s}: {e}")

    print("\nDone. Files saved under ./data/")

if __name__ == "__main__":
    main()
