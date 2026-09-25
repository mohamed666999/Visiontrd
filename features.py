#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ميزات سببية (Causal) — لا تستخدم أي معلومة من المستقبل."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(s, span): return s.ewm(span=span, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rolling_z(s, win):
    m = s.rolling(win, min_periods=max(20, win // 3)).mean()
    v = s.rolling(win, min_periods=max(20, win // 3)).std()
    return (s - m) / v.replace(0, np.nan)


def build_features(df, cfg) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    ret1 = c.pct_change()
    f["ret_1"], f["ret_3"] = ret1, c.pct_change(3)
    f["ret_12"], f["ret_48"] = c.pct_change(12), c.pct_change(48)
    for w in (12, 48, 96):
        f[f"mom_z_{w}"] = rolling_z(c.pct_change(w), cfg.zscore_window)
    f["mom_accel"] = f["ret_3"] - f["ret_12"]
    vol_ew = ret1.ewm(halflife=cfg.vol_halflife, adjust=False).std()
    f["vol_ew"], f["vol_z"] = vol_ew, rolling_z(vol_ew, cfg.zscore_window)
    f["vol_ratio"] = vol_ew / vol_ew.rolling(96).mean()
    f["atr_norm"] = atr(df, 14) / c
    f["dist_ema20"] = (c - ema(c, 20)) / (atr(df, 14) + 1e-9)
    f["dist_ema50"] = (c - ema(c, 50)) / (atr(df, 14) + 1e-9)
    f["dist_ema200"] = (c - ema(c, 200)) / c
    f["ema_slope"] = ema(c, 50).diff(12) / c
    f["adx_proxy"] = c.diff(12).abs() / (atr(df, 14) * 12 + 1e-9)
    f["rsi14"] = (rsi(c, 14) - 50) / 50
    f["rsi2"] = (rsi(c, 2) - 50) / 50
    f["bb_pos"] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std() + 1e-9)
    f["zscore_close"] = rolling_z(c, 96)
    rng_ = (h - l).replace(0, np.nan)
    f["body_ratio"] = (c - o) / rng_
    f["upper_wick"] = (h - np.maximum(c, o)) / rng_
    f["lower_wick"] = (np.minimum(c, o) - l) / rng_
    f["close_loc"] = (c - l) / rng_
    if "taker_buy_base" in df:
        tbr = df["taker_buy_base"] / v.replace(0, np.nan)
        f["taker_imb"] = (tbr - 0.5) * 2
        f["taker_imb_z"] = rolling_z(f["taker_imb"], 96)
    f["vol_z2"] = rolling_z(np.log1p(v), cfg.zscore_window)
    f["trade_size"] = np.log1p(df["quote_volume"] / df["trades"].replace(0, np.nan))
    if "funding_rate" in df:
        f["funding"] = df["funding_rate"]
        f["funding_z"] = rolling_z(df["funding_rate"], 96)
    if "open_interest" in df:
        f["oi_chg"] = df["open_interest"].pct_change()
        f["oi_z"] = rolling_z(f["oi_chg"], 96)
    f["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = {
    "momentum": ["mom_z_12", "mom_z_48", "mom_z_96", "mom_accel", "ret_1", "ret_3", "ret_12"],
    "volatility": ["vol_z", "vol_ratio", "atr_norm", "vol_ew"],
    "trend": ["dist_ema20", "dist_ema50", "dist_ema200", "ema_slope", "adx_proxy"],
    "reversion": ["rsi14", "rsi2", "bb_pos", "zscore_close"],
    "flow": ["taker_imb", "taker_imb_z", "vol_z2", "trade_size", "body_ratio",
             "close_loc", "upper_wick", "lower_wick"],
    "derivs": ["funding", "funding_z", "oi_chg", "oi_z"],
    "time": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
}
GROUP_SIGN = {"momentum": +1.0, "trend": +1.0, "flow": +1.0,
              "reversion": -1.0, "volatility": -1.0, "derivs": 0.0, "time": 0.0}


def build_expert_inputs(f: pd.DataFrame) -> pd.DataFrame:
    """اتجاه كل مجموعة أوزان → إشارة خبير في [-1,+1]، بمعايرة دائرة في الزمن."""
    out = pd.DataFrame(index=f.index)
    for name, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in f.columns]
        if not cols:
            continue
        sub = f[cols]
        sub = sub.sub(sub.rolling(250, min_periods=60).mean()).div(
            sub.rolling(250, min_periods=60).std().replace(0, np.nan))
        raw = np.tanh(sub).mean(axis=1)
        out[f"dir_{name}"] = (GROUP_SIGN[name] * raw).clip(-1, 1) if GROUP_SIGN[name] else raw.clip(-1, 1)
    return out
