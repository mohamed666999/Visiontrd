#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""التسميات الثلاث، مفصولة بوضوح.

1) signal  : عائد الإغلاق بعد h_signal شمعة  → تدريب الخبراء التعلميين (اتجاه)
2) barrier : نتيجة Triple-Barrier محاكاة حقيقية — من ضُرب أولًا؟ SL أم TP؟
   سياسة محافظة: لو ضربت الشمعة الحاجزين معًا → نعتبر SL أولًا.
3) meta    : لو دخلنا بهذا الاتجاه (SL/TP كما في الاستراتيجية)،
             هل كانت الـR-multiple موجبة بعد التكاليف؟ → بوابة الدخول.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from features import atr


def _first_barrier_hit(high, low, entry, sl_price, tp_price, o, h, l, c, start, max_bars):
    """يمسح الشمعات بعد الدخول بارًّا بارًّا ويعيد (النتيجة، سعر الخروج، عدد البارات).
    داخل الشمعة الواحدة: إن ضُرب الحاجزان معًا → SL أولًا (محافظ)."""
    for j in range(start, min(start + max_bars, len(c))):
        hj, lj = h[j], l[j]
        hit_sl = lj <= sl_price if sl_price is not None else False
        hit_tp = hj >= tp_price if tp_price is not None else False
        if hit_sl:                       # SL أولًا دائمًا عند الالتباس
            return -1.0, sl_price, j - start + 1
        if hit_tp:
            return +1.0, tp_price, j - start + 1
    # لم يُضرب أي حاجز → خروج زمني عند إغلاق آخر شمعة في النافذة
    end = min(start + max_bars, len(c)) - 1
    if end < start:
        return 0.0, c[min(start, len(c) - 1)], 0
    return 0.0, c[end], end - start + 1


def compute_labels(df, cfg) -> pd.DataFrame:
    c = df["close"]
    a = atr(df, 14).values
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; cc = c.values
    n = len(df)
    hs = cfg.h_signal

    # --- 1) هدف الإشارة -------------------------------------------------
    fwd_ret = c.shift(-hs) / c - 1.0

    # --- 2) + 3) محاكاة حاجز/ميتا بارًّا بارًّا (دخول عند فتح t+1) ---------
    sign = np.sign(fwd_ret.values)
    y_barrier = np.zeros(n); r_mult = np.full(n, np.nan); bars_out = np.zeros(n, dtype=int)
    cost_rt = (2 * (cfg.fee_bps + cfg.slip_bps) / 10_000.0)  # ذهاب+إياب على القيمة الاسمية

    for i in range(n - 1):
        if not np.isfinite(sign[i]) or sign[i] == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        entry = o[i + 1]                      # تنفيذ عند فتح الشمعة التالية
        stop = cfg.atr_stop_mult * a[i]
        slp = entry - sign[i] * stop
        tpp = entry + sign[i] * cfg.atr_tp_mult * a[i]
        out, exit_px, nb = _first_barrier_hit(h, l, entry, slp, tpp, o, h, l, cc,
                                              i + 1, cfg.h_barrier)
        y_barrier[i] = out * sign[i]          # +1 رابح، -1 خاسر، 0 زمني
        # R-multiple صحيح: العائد النسبي ÷ المخاطرة النسبية (لا خلط وحدات)
        risk_frac = stop / entry
        gross_frac = sign[i] * (exit_px / entry - 1.0)
        r_mult[i] = gross_frac / risk_frac
        bars_out[i] = nb

    size_atr = pd.Series(cfg.atr_stop_mult * a / cc, index=df.index).replace(0, np.nan)
    cost_r = ((2 * (cfg.fee_bps + cfg.slip_bps) / 10_000.0) / size_atr)
    net_r = pd.Series(r_mult, index=df.index) - cost_r
    meta = (net_r > 0).astype(float)
    meta[~np.isfinite(net_r)] = np.nan

    return pd.DataFrame({"fwd_ret": fwd_ret, "y_barrier": y_barrier,
                         "r_mult": net_r, "bars_to_exit": bars_out, "meta": meta},
                        index=df.index)


def label_report(lab: pd.DataFrame) -> dict:
    v = lab.dropna(subset=["r_mult"])
    if len(v) == 0:
        return {}
    return {"n": int(len(v)),
            "win_share": float((v["y_barrier"] > 0).mean()),
            "loss_share": float((v["y_barrier"] < 0).mean()),
            "timeout_share": float((v["y_barrier"] == 0).mean()),
            "avg_R": float(v["r_mult"].mean()),
            "meta_positive_share": float(v["meta"].mean())}
