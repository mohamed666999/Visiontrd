#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""جلب بيانات بينانس USDⓈ-M الآجلة — لا يحتاج مفاتيح (نقاط عامة)."""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
               "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    num = ["open", "high", "low", "close", "volume", "quote_volume",
           "taker_buy_base", "taker_buy_quote"]
    df[num] = df[num].astype(float)
    df["trades"] = df["trades"].astype(int)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[num + ["trades"]]


def fetch_klines(symbol, interval, limit=8000, base_url="https://fapi.binance.com"):
    if requests is None:
        raise RuntimeError("requests غير متوفرة")
    out, end, step = [], None, 1500
    while len(out) < limit:
        n = min(step, limit - len(out))
        params = {"symbol": symbol, "interval": interval, "limit": n}
        if end is not None:
            params["endTime"] = end
        r = requests.get(f"{base_url}/fapi/v1/klines", params=params, timeout=25)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out = rows + out
        end = rows[0][0] - 1
        time.sleep(0.12)
    df = _to_df(out)
    return df[~df.index.duplicated(keep="first")].sort_index().tail(limit)


def fetch_funding(symbol, base_url="https://fapi.binance.com") -> pd.Series:
    if requests is None:
        return pd.Series(dtype=float)
    r = requests.get(f"{base_url}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": 1000}, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([float(x["fundingRate"]) for x in rows],
                     index=pd.to_datetime([x["fundingTime"] for x in rows], unit="ms", utc=True)).sort_index()


def fetch_open_interest(symbol, period="15m", base_url="https://fapi.binance.com") -> pd.Series:
    if requests is None:
        return pd.Series(dtype=float)
    r = requests.get(f"{base_url}/futures/data/openInterestHist",
                     params={"symbol": symbol, "period": period, "limit": 500}, timeout=25)
    if r.status_code != 200:
        return pd.Series(dtype=float)
    rows = r.json()
    return pd.Series([float(x["sumOpenInterest"]) for x in rows],
                     index=pd.to_datetime([x["timestamp"] for x in rows], unit="ms", utc=True)).sort_index()


def load_market(symbol, interval, limit, base_url="https://fapi.binance.com",
                cache_dir="data_cache", with_derivs=True, max_age=600) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{interval}_{limit}.csv")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch_klines(symbol, interval, limit, base_url)
    if with_derivs:
        try:
            fr = fetch_funding(symbol, base_url)
            if len(fr):
                df["funding_rate"] = fr.reindex(df.index, method="ffill").fillna(0.0)
        except Exception:
            pass
        try:
            oi = fetch_open_interest(symbol, interval, base_url)
            if len(oi):
                df["open_interest"] = oi.reindex(df.index, method="ffill")
        except Exception:
            pass
    df.to_csv(path)
    return df


def make_synthetic_data(n=8000, seed=7, start="2023-01-01", interval="15m") -> pd.DataFrame:
    """بيانات صناعية بأنظمة (لاختبار الكود بلا إنترنت). ليست للاستنتاج الاستثماري."""
    rng = np.random.default_rng(seed)
    drift = np.array([0.00004, 0.0009, -0.0008, 0.0])
    vol = np.array([0.004, 0.007, 0.009, 0.02])
    trans = np.array([[0.96, 0.02, 0.01, 0.01], [0.03, 0.93, 0.02, 0.02],
                      [0.03, 0.02, 0.93, 0.02], [0.05, 0.05, 0.05, 0.85]])
    st, states = 0, []
    for _ in range(n):
        states.append(st); st = rng.choice(4, p=trans[st])
    states = np.array(states)
    r = drift[states] + vol[states] * rng.standard_normal(n)
    for i in range(1, n):
        r[i] += 0.22 * r[i - 1]
    price = 20000.0 * np.exp(np.cumsum(r))
    hi = price * (1 + np.abs(rng.standard_normal(n)) * vol[states])
    lo = price * (1 - np.abs(rng.standard_normal(n)) * vol[states])
    op = np.concatenate([[price[0]], price[:-1]])
    volume = 500 * (1 + 2 * np.abs(r)) * (1 + 0.5 * rng.standard_normal(n)).clip(0.2)
    taker = volume * (0.5 + 0.25 * np.tanh(r * 300) + 0.05 * rng.standard_normal(n))
    freq = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1h","4h":"4h","1d":"1D"}.get(interval, interval)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({"open": op, "high": np.maximum.reduce([op, price, hi]),
                       "low": np.minimum.reduce([op, price, lo]), "close": price,
                       "volume": volume, "quote_volume": volume * price,
                       "trades": (volume / 3).astype(int),
                       "taker_buy_base": taker, "taker_buy_quote": taker * price}, index=idx)
    df["funding_rate"] = 0.0001 * np.tanh(r * 200)
    df["open_interest"] = (pd.Series(volume, index=idx).rolling(48).sum() / pd.Series(price, index=idx)).bfill()
    df.index.name = "open_time"
    return df
