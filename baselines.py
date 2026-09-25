#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""مقارنة FUGU-MAX بنسخ مرجعية (Baselines) على نفس البيانات ونفس التكاليف ونفس قواعد التنفيذ.

A  = Random entry
B  = Fixed ensemble (أوزان ثابتة، بلا تعلّم)
C  = Adaptive ensemble (تعلّم + نظام سوق، بلا ميتا)
D  = Adaptive + Regime + Meta   ← FUGU-MAX الكامل
E  = Buy & Hold (للسياق)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtest import run, metrics, walk_forward

SCHEMES = [
    ("A_random", "random"),
    ("B_fixed", "fixed"),
    ("C_adaptive", "regime_free"),
    ("D_fugu_max", "adaptive"),
    ("E_buy_hold", "buy_hold"),
]


def compare_on_dataframe(df, cfg, warmup=1000, seed=0) -> pd.DataFrame:
    rows = []
    for name, mode in SCHEMES:
        r = run(df, cfg, mode=mode, warmup=warmup, seed=seed)
        m = metrics(r["equity"], cfg, len(r["trades"]))
        m.update({"scheme": name, "mode": mode,
                  "expert_weights": r["engine"]._last.get("weights", {})})
        rows.append(m)
    return pd.DataFrame(rows).set_index("scheme")


def capital_curves(df, cfg, warmup=1000, seed=0) -> pd.DataFrame:
    out = {}
    for name, mode in SCHEMES:
        out[name] = run(df, cfg, mode=mode, warmup=warmup, seed=seed)["equity"]
    return pd.DataFrame(out)


def compare_symbols(loader, symbols, cfg, warmup=1000, out_of_sample=False,
                    split=0.7) -> pd.DataFrame:
    """يقارن الأنظمة عبر عدة أزواج. loader(symbol) → DataFrame."""
    rows = []
    for sym in symbols:
        try:
            df = loader(sym)
        except Exception as e:
            rows.append({"symbol": sym, "scheme": "ERROR", "total_return": np.nan,
                         "sharpe": np.nan, "max_drawdown": np.nan, "note": str(e)[:60]})
            continue
        if out_of_sample:
            cut = int(len(df) * split)
            df = df.iloc[cut:]
        for mode in ("fixed", "regime_free", "adaptive"):
            r = run(df, cfg, mode=mode, warmup=min(warmup, max(300, len(df) // 3)))
            m = metrics(r["equity"], cfg, len(r["trades"]))
            m["symbol"] = sym
            m["mode"] = mode
            rows.append(m)
    return pd.DataFrame(rows)


def pretty(dfres: pd.DataFrame) -> str:
    cols = ["total_return", "sharpe", "max_drawdown", "n_trades"]
    d = dfres[cols].copy()
    d["total_return"] = d["total_return"].map(lambda x: f"{x:+.2%}")
    d["sharpe"] = d["sharpe"].map(lambda x: f"{x:.2f}")
    d["max_drawdown"] = d["max_drawdown"].map(lambda x: f"{x:.2%}")
    return d.to_string()
