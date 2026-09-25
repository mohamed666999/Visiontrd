#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUGU-MAX — إشارة واحدة مفصّلة (بلا تنفيذ).

    python fugu_indicator.py --symbol BTCUSDT --interval 15m
    python fugu_indicator.py --symbol ETHUSDT --demo --json

يُطبع: القرار، درجة المؤشر، إشارة كل خبير ووزنه، النظام، الإجماع، الانجراف،
واحتمال الميتا — أي "لماذا" هذا القرار، لا مجرد رقم.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

from config import Config
from data import load_market, make_synthetic_data
from features import build_features, build_expert_inputs, atr
from labels import compute_labels
from models import Ensemble, DriftMonitor, RegimeDetector, REGIME_NAMES, EXPERT_NAMES
from strategy import MetaFilter


def build(df, cfg, warmup=800, learn=True):
    f = build_features(df, cfg)
    dirs = build_expert_inputs(f)
    F = f.fillna(0.0).values
    lab = compute_labels(df, cfg)
    eng = Ensemble(cfg, F.shape[1]); eng.attach_drift(DriftMonitor(cfg.drift_delta, cfg.drift_threshold))
    reg_det = RegimeDetector(cfg.regime_lookback)
    meta = MetaFilter()
    n = len(df)
    h_max = max(cfg.h_signal, cfg.h_barrier) + 1
    scores = np.full(n, np.nan); regs = np.zeros(n, dtype=int); mps = np.full(n, np.nan)
    for t in range(warmup, n):
        x = F[t]
        d = {k: float(dirs.iloc[t][k]) for k in dirs.columns}
        reg = reg_det.classify(float(f["vol_ew"].iloc[t] or 0.2), float(f["dist_ema50"].iloc[t] or 0))
        s = eng.score(x, d, reg)
        scores[t] = s; regs[t] = reg
        mps[t] = meta.proba(s, eng._last["agreement"], reg, float(f["vol_z"].iloc[t] or 0),
                            eng.drift.hit_rate, 0.0)
        if learn and t - h_max >= warmup:
            j = t - h_max
            rr = lab["fwd_ret"].iloc[j]
            if np.isfinite(rr):
                yb = lab["y_barrier"].iloc[j]; mt = lab["meta"].iloc[j]
                eng.learn(F[j], {k: float(dirs.iloc[j][k]) for k in dirs.columns}, float(rr),
                          regs[j], float(yb) if np.isfinite(yb) else 0.0,
                          float(mt) if np.isfinite(mt) else 0.0)
                meta.update(scores[j], 0.5, regs[j], float(f["vol_z"].iloc[j] or 0),
                            eng.drift.hit_rate, 0.0, float(mt > 0) if np.isfinite(mt) else 0.0)
                eng.update_drift(1.0 if np.sign(rr) == np.sign(scores[j]) else 0.0)
    return f, eng, meta, scores, regs, mps


def snapshot(cfg, demo=False, warmup=800):
    if demo:
        data = make_synthetic_data(max(cfg.history_bars, 3000))
    else:
        data = load_market(cfg.symbols[0], cfg.interval, cfg.history_bars, cfg.market_base,
                           cfg.path(cfg.cache_dir))
    data = data.copy()
    f, eng, meta, scores, regs, mps = build(data, cfg, warmup=warmup)
    t = len(data) - 1
    s = float(scores[t]) if np.isfinite(scores[t]) else 0.0
    return data, f, eng, meta, scores, regs, mps, s


def decide(s, cfg):
    if s >= cfg.score_entry:
        return "🟢 شراء (LONG)"
    if s <= -cfg.score_entry:
        return "🔴 بيع (SHORT)"
    if abs(s) <= cfg.score_exit:
        return "⚪ انتظار / إلغاء"
    return "🟡 مراقبة"


def payload(data, f, eng, meta, scores, regs, mps, s, cfg, symbol):
    t = len(data) - 1
    reg = int(regs[t])
    return {
        "symbol": symbol, "timeframe": cfg.interval,
        "timestamp": str(data.index[t]),
        "last_price": float(data["close"].iloc[t]),
        "score": round(s, 2), "score_entry": cfg.score_entry,
        "action": decide(s, cfg),
        "regime": reg, "regime_name": REGIME_NAMES[reg],
        "agreement": round(float(eng._last.get("agreement", 0.0)), 3),
        "drift": bool(eng.drift.drift),
        "master_hit_rate": round(float(eng.drift.hit_rate), 3),
        "meta_probability": round(float(mps[t]) if np.isfinite(mps[t]) else 0.0, 3),
        "expert_signals": {k: round(float(v), 3) for k, v in eng._last.get("signals", {}).items()},
        "expert_weights": {k: round(float(v), 3) for k, v in eng._last.get("weights", {}).items()},
        "expert_hit_rate": {k: round(v, 3) for k, v in eng.snapshot()["expert_hit_rate"].items()},
        "learning_steps": eng.steps,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="FUGU-MAX — مؤشر ذاتي التعلّم لعقود بينانس USDⓈ-M")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--bars", type=int, default=8000)
    ap.add_argument("--demo", action="store_true", help="بيانات صناعية (بلا إنترنت)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config(); cfg.symbols = [a.symbol]; cfg.interval = a.interval; cfg.history_bars = a.bars
    try:
        data, f, eng, meta, scores, regs, mps, s = snapshot(cfg=cfg, demo=a.demo)
    except Exception as e:
        print(f"⚠️ تعذّر الجلب ({e}) — بيانات صناعية.")
        data, f, eng, meta, scores, regs, mps, s = snapshot(cfg=cfg, demo=True)
    out = payload(data, f, eng, meta, scores, regs, mps, s, cfg, a.symbol)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    W = 64
    print("=" * W)
    print(f"  FUGU-MAX  ●  {out['symbol']}  {out['timeframe']}  (عقود USDⓈ-M الآجلة)")
    print("=" * W)
    print(f"  السعر الأخير        : {out['last_price']:,.4f}")
    print(f"  درجة المؤشر         : {out['score']:+.1f}   (المدى -100..100، حد الدخول ±{out['score_entry']:.0f})")
    print(f"  القرار              : {out['action']}")
    print(f"  النَّظام             : {out['regime_name']}  |  إجماع الخبراء: {out['agreement']:.0%}")
    print(f"  الانجراف            : {'⚠️ نعم' if out['drift'] else 'لا'}  |  إصابة المحرّك: {out['master_hit_rate']:.0%}")
    print(f"  بوابة الميتا        : {out['meta_probability']:.2f}  (الحد {cfg.meta_threshold})")
    print(f"  خطوات التعلّم       : {out['learning_steps']}")
    print("-" * W)
    print("  مجلس الخبراء        (الإشارة / الوزن / الإصابة)")
    for k in EXPERT_NAMES:
        sg = out["expert_signals"].get(k, 0.0)
        w = out["expert_weights"].get(k, 0.0)
        hr = out["expert_hit_rate"].get(k, 0.0)
        bar = "█" * int(round(abs(sg) * 10))
        print(f"    {k:11s} {sg:+.2f}  {w:6.1%}  {hr:5.0%}  {bar}")
    print("=" * W)
    return out


if __name__ == "__main__":
    main()
