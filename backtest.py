#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الاختبار الخلفي — بلا أي تطلّع للمستقبل.

قواعد التنفيذ الصارمة:
    1. الإشارة تُحسب من بيانات الشمعة t المكتملة فقط.
    2. القرار يُنفَّذ عند **فتح** الشمعة t+1 (أمر معلّق).
    3. SL/TP يُفحصان داخل الشمعة بشمعة-بشمعة، وعند الالتباس (ضرب الاثنين) يُعتبر SL أولًا.
    4. الرسوم والانزلاق تُطبّق عند التبديل، والفائدة التمويلية على كل شمعة احتفاظ.
    5. التعلّم يحدث فقط من نتيجة اكتملت سابقًا.

الأنماط (للمقارنة):
    adaptive     : النظام الكامل (تعلّم + نظام سوق + ميتا)
    no_meta      : تعلّم + نظام بلا بوابة ميتا
    regime_free  : تعلّم + ميتا بلا نظام سوق
    fixed        : أوزان ثابتة (بلا تعلّم) — Baseline B
    random       : دخول عشوائي — Baseline A
    buy_hold     : شراء واحتفاظ — Baseline
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from features import build_features, build_expert_inputs, atr
from labels import compute_labels
from models import Ensemble, DriftMonitor, RegimeDetector, EXPERT_NAMES
from strategy import MetaFilter, RiskManager, Position, make_sltp, trailing_update, target_leverage


def bars_per_year(interval: str) -> float:
    m = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    return 365.0 * 24 * 60 / m.get(interval, 15)


def _cost_frac(cfg) -> float:
    return (cfg.fee_bps + cfg.slip_bps) / 10_000.0


def run(df, cfg, mode="adaptive", engine=None, warmup=800, learn=True, seed=0):
    f = build_features(df, cfg)
    dirs = build_expert_inputs(f)
    lab = compute_labels(df, cfg)
    F = f.fillna(0.0).values
    Dv = dirs.fillna(0.0)
    Fv = f.values                       # للقيم الخام (vol_z ...)
    n = len(df)
    bpy = bars_per_year(cfg.interval)

    eng = engine or Ensemble(cfg, F.shape[1])
    if not hasattr(eng, "drift"):
        eng.attach_drift(DriftMonitor(cfg.drift_delta, cfg.drift_threshold))
    reg_det = RegimeDetector(cfg.regime_lookback)
    meta = MetaFilter()
    rm = RiskManager(cfg, equity0=10000.0)
    a = atr(df, 14).values
    op = df["open"].values; hi = df["high"].values
    lo = df["low"].values; cl = df["close"].values
    ret1 = df["close"].pct_change().fillna(0.0).values
    vol_ann = (pd.Series(ret1).rolling(cfg.vol_halflife).std().fillna(0.4) * np.sqrt(bpy)
               ).clip(0.05, 3.0).values
    fr = df["funding_rate"].values if "funding_rate" in df else np.zeros(n)

    rng = np.random.default_rng(seed)
    h_max = max(cfg.h_signal, cfg.h_barrier) + 1
    cost = _cost_frac(cfg)
    use_learn = learn and mode in ("adaptive", "no_meta", "regime_free")
    use_meta = mode in ("adaptive", "regime_free")
    use_regime = mode in ("adaptive", "no_meta")

    equity = 10000.0
    eq = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    regimes = np.zeros(n, dtype=int)
    meta_p = np.full(n, np.nan)
    pos = Position()
    pending = None                       # القرار المعلّق للشمعة التالية
    trades = []

    for t in range(warmup, n):
        # ---------- 1) تنفيذ الأمر المعلّق عند فتح هذه الشمعة ----------
        if pending is not None:
            if pos.is_open and pending["dir"] != pos.side:
                gross = pos.side * (op[t] / pos.entry - 1.0) * pos.qty
                pnl = (gross - 2 * cost * pos.qty) * equity
                trades.append({"side": pos.side, "entry": pos.entry, "exit": op[t],
                               "qty": pos.qty, "pnl_pct": pnl / equity, "bars": pos.bars,
                               "reason": "REVERSE"})
                equity += pnl
                pos = Position()
            if not pos.is_open and pending["dir"] != 0:
                side = pending["dir"]; ent = op[t]
                sl, tp = make_sltp(side, ent, a[t - 1] if a[t - 1] > 0 else a[t], cfg)
                pos = Position(side=side, qty=pending["lev"], entry=ent, sl=sl, tp=tp,
                               bars=0, reg=pending["reg"])
            pending = None

        # ---------- 2) فحص SL/TP داخل الشمعة ----------
        if pos.is_open:
            pos.bars += 1
            trailing_update(pos, cl[t], a[t], cfg)
            hit_sl = (lo[t] <= pos.sl) if pos.side > 0 else (hi[t] >= pos.sl)
            hit_tp = (hi[t] >= pos.tp) if pos.side > 0 else (lo[t] <= pos.tp)
            exit_px = None; reason = None
            if hit_sl:
                exit_px, reason = pos.sl, "SL"       # SL أولًا عند الالتباس (محافظ)
            elif hit_tp:
                exit_px, reason = pos.tp, "TP"
            elif pos.bars >= cfg.time_stop_bars:
                exit_px, reason = cl[t], "TIME"
            if exit_px is not None:
                gross = pos.side * (exit_px / pos.entry - 1.0) * pos.qty
                pnl = (gross - 2 * cost * pos.qty) * equity
                trades.append({"side": pos.side, "entry": pos.entry, "exit": exit_px,
                               "qty": pos.qty, "pnl_pct": pnl / equity, "bars": pos.bars,
                               "reason": reason})
                equity += pnl
                pos = Position()

        # ---------- 3) العائد الحالي (Mark-to-market) ----------
        if t > warmup:
            if pos.is_open and cl[t - 1] > 0:
                r = pos.side * pos.qty * (cl[t] / cl[t - 1] - 1.0)
                r -= pos.side * fr[t] * (1.0 / cfg.funding_bars) * pos.qty   # الفائدة التمويلية
                equity *= (1.0 + r)
            rm.equity = equity
        eq[t] = equity
        rm.peak = max(rm.peak, equity)

        # ---------- 4) القرار عند إغلاق الشمعة t ----------
        t_last = (t == n - 1)
        if not t_last:
            x = F[t]
            d = {k: float(Dv.iloc[t][k]) for k in Dv.columns}
            if use_regime:
                reg = reg_det.classify(float(f["vol_ew"].iloc[t] or 0.2),
                                       float(f["dist_ema50"].iloc[t] or 0))
            else:
                reg = 0
            regimes[t] = reg
            s = eng.score(x, d, reg)
            scores[t] = s
            agree = eng._last["agreement"]
            hr = getattr(eng, "drift", DriftMonitor()).hit_rate
            mp = meta.proba(s, agree, reg, float(f["vol_z"].iloc[t] or 0), hr,
                            equity / 10000.0 - 1.0)
            meta_p[t] = mp

            if not pos.is_open and rm.check(df.index[t]):
                want = 0
                if mode in ("adaptive", "no_meta", "regime_free"):
                    gate = (mp >= cfg.meta_threshold) if (use_meta and meta.ready) else True
                    if abs(s) >= cfg.score_entry and gate:
                        want = 1 if s > 0 else -1
                elif mode == "fixed":
                    if abs(s) >= cfg.score_entry:
                        want = 1 if s > 0 else -1
                elif mode == "random":
                    if rng.random() < 0.02:
                        want = 1 if rng.random() < 0.5 else -1
                elif mode == "buy_hold":
                    want = 0
                if want != 0:
                    lev = abs(target_leverage(s if mode != "random" else 40.0, vol_ann[t], cfg))
                    if lev >= 0.05:
                        pending = {"dir": want, "lev": lev, "reg": reg}
            elif mode in ("adaptive", "no_meta", "regime_free", "fixed"):
                if pos.is_open:
                    if (pos.side > 0 and s < cfg.score_exit) or (pos.side < 0 and s > -cfg.score_exit):
                        pending = {"dir": 0, "lev": 0.0, "reg": reg}
                    elif not rm.check(df.index[t]):
                        pending = {"dir": 0, "lev": 0.0, "reg": reg}

            # ---------- 5) التعلّم — بتأخير صارم بلا تطلّع ----------
            # النتيجة المقابلة للشمعة t لا تكتمل إلا بعد h_max شمعة.
            # لذلك نتعلّم الآن من الشمعة  j = t - h_max  فقط.
            if use_learn and t - h_max >= warmup:
                j = t - h_max
                realized = lab["fwd_ret"].iloc[j]
                if np.isfinite(realized):
                    xj = F[j]
                    dj = {k: float(Dv.iloc[j][k]) for k in Dv.columns}
                    rj = regimes[j]
                    y_b = lab["y_barrier"].iloc[j] if np.isfinite(lab["y_barrier"].iloc[j]) else 0.0
                    m_t = lab["meta"].iloc[j] if np.isfinite(lab["meta"].iloc[j]) else 0.0
                    eng.learn(xj, dj, float(realized), rj, float(y_b), float(m_t))
                    sj = scores[j] if np.isfinite(scores[j]) else 0.0
                    meta.update(sj, 0.5, rj, float(f["vol_z"].iloc[j] or 0), hr, 0.0, float(m_t > 0))
                    eng.update_drift(1.0 if np.sign(realized) == np.sign(sj) else 0.0)

    eq_s = pd.Series(eq, index=df.index, dtype=float)
    eq_s.iloc[:warmup] = 10000.0
    return {"equity": eq_s, "scores": scores, "regimes": regimes, "meta_prob": meta_p,
            "trades": pd.DataFrame(trades), "engine": eng, "features": f}


def metrics(eq: pd.Series, cfg, n_trades=0) -> dict:
    eq = eq.dropna()
    if len(eq) < 10:
        return {}
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    bpy = bars_per_year(cfg.interval)
    peak = eq.cummax()
    dd = (1 - eq / peak)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (bpy / max(len(eq), 1)) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = float(np.std(r) * np.sqrt(bpy))
    sharpe = float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(bpy))
    dn = r[r < 0]
    sortino = float(np.mean(r) / (np.std(dn) + 1e-12) * np.sqrt(bpy)) if len(dn) else float("nan")
    maxdd = float(dd.max())
    return {"total_return": float(total), "CAGR": float(cagr), "vol_ann": vol,
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": maxdd,
            "calmar": float(cagr / (maxdd + 1e-9)), "n_trades": int(n_trades),
            "final_equity": float(eq.iloc[-1])}


def walk_forward(df, cfg, n_splits=4, mode="adaptive", verbose=True):
    n = len(df); seg = n // (n_splits + 1); rows = []
    for k in range(n_splits):
        tr_end = seg * (k + 1); te_end = min(seg * (k + 2), n)
        r_tr = run(df.iloc[:tr_end], cfg, mode=mode, warmup=600, learn=True)
        engine = r_tr["engine"]
        r_te = run(df.iloc[max(0, tr_end - 400):te_end], cfg, mode=mode,
                   engine=engine, warmup=400, learn=False)   # متجمّد
        m = metrics(r_te["equity"], cfg, len(r_te["trades"]))
        m.update({"split": k, "test_start": str(df.index[tr_end]),
                  "test_end": str(df.index[te_end - 1])})
        rows.append(m)
        if verbose:
            print(f"  split {k}: {m.get('total_return',0):+7.2%} | Sharpe {m.get('sharpe',0):5.2f} "
                  f"| DD {m.get('max_drawdown',0):5.2%} | trades {m.get('n_trades',0)}")
    return pd.DataFrame(rows)
