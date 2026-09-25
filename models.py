#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUGU-MAX — Adaptive Online Ensemble.

كل خبير يُخرج إشارة ∈ [-1,+1]:
    momentum, trend, reversion, flow, vlr (vol-regime), logistic, rls, baseline

F(t) = Σ w_i(t) · s_i(t)   →   ثم يُسحب إلى [-100,+100]

الأوزان تتعلّم online بطريقة Exponentiated Gradient (Hedge) وتُعاقَب عند الخطأ،
مع نسخة أوزان مختلفة لكل نظام سوق (Regime-specific weights)، ومراقبة انجراف.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

EXPERT_NAMES = ["momentum", "trend", "reversion", "flow", "volatility",
                "logistic", "rls", "baseline"]
REGIME_NAMES = ["QUIET", "TREND", "HIGH_VOL"]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ----------------------------------------------------------------------
# خبراء الدرجة الأولى (Rule Experts) — تُغذّى من نوافذ الاتجاه مباشرة
# ----------------------------------------------------------------------
def rule_signals(dirs: Dict[str, float]) -> Dict[str, float]:
    mom = dirs.get("dir_momentum", 0.0)
    trend = dirs.get("dir_trend", 0.0)
    rev = dirs.get("dir_reversion", 0.0)
    flow = dirs.get("dir_flow", 0.0)
    volr = dirs.get("dir_volatility", 0.0)
    return {"momentum": float(np.clip(mom, -1, 1)),
            "trend": float(np.clip(trend, -1, 1)),
            "reversion": float(np.clip(rev, -1, 1)),
            "flow": float(np.clip(flow, -1, 1)),
            "volatility": float(np.clip(volr, -1, 1)),
            "baseline": float(np.clip(-rev, -1, 1))}   # مرجع مضاد للارتداد


# ----------------------------------------------------------------------
# خبراء تعلّميون (ML Experts)
# ----------------------------------------------------------------------
class LogisticExpert:
    def __init__(self, n, lr=0.02, l2=1e-4):
        self.n, self.lr, self.l2 = n, lr, l2
        self.w = np.zeros(n); self.b = 0.0

    def proba(self, x):
        return float(_sig(self.w @ x + self.b))

    def signal(self, x):
        return float(2.0 * self.proba(x) - 1.0)

    def update(self, x, y):          # y ∈ {0,1}
        p = self.proba(x); g = p - y
        self.w -= self.lr * (g * x + self.l2 * self.w)
        self.b -= self.lr * g


class RLSExpert:
    """مربعات صغرى تكرارية بنسيان أسّي — تتكيّف مع تغيّر السوق لحظيًا."""
    def __init__(self, n, lam=0.995, delta=50.0):
        self.n, self.lam = n, lam
        self.w = np.zeros(n); self.b = 0.0
        self.P = np.eye(n + 1) * delta

    def predict(self, x):
        return float(self.w @ x + self.b)

    def signal(self, x):
        return float(np.tanh(self.predict(x) * 5.0))

    def update(self, x, y):
        xm = np.append(x, 1.0); wm = np.append(self.w, self.b)
        px = self.P @ xm
        k = px / (self.lam + float(xm @ px))
        wm += k * (y - float(wm @ xm))
        self.P = (self.P - np.outer(k, px)) / self.lam
        self.w, self.b = wm[:-1], wm[-1]


# ----------------------------------------------------------------------
# كاشف الانجراف (Page-Hinkley)
# ----------------------------------------------------------------------
class DriftMonitor:
    def __init__(self, delta=0.0004, threshold=3.5, warm=80):
        self.delta, self.threshold, self.warm = delta, threshold, warm
        self.n = 0; self.mean = 0.0; self.cum = 0.0; self.mn = 0.0
        self.drift = False; self._hit = 0.5; self.hit_rate = 0.5

    def update(self, correct: float) -> bool:
        self.n += 1
        self._hit = 0.98 * self._hit + 0.02 * correct
        self.hit_rate = self._hit
        err = 1.0 - correct
        self.mean += (err - self.mean) / self.n
        self.cum += (err - self.mean - self.delta)
        self.mn = min(self.mn, self.cum)
        if self.n > self.warm and (self.cum - self.mn) > self.threshold:
            self.cum = self.mn = 0.0; self.n = 0; self.drift = True
            return True
        return False


# ----------------------------------------------------------------------
# كاشف النظام (Regime) — قائم على كمّيات التقلب، لا تعلّم مُعلَّم
# ----------------------------------------------------------------------
class RegimeDetector:
    def __init__(self, lookback=500):
        self.lookback = lookback
        self.vol_hist: List[float] = []

    def classify(self, vol: float, trend_strength: float) -> int:
        self.vol_hist.append(vol)
        if len(self.vol_hist) > self.lookback:
            self.vol_hist.pop(0)
        if len(self.vol_hist) < 50:
            return 0
        q1, q2 = np.percentile(self.vol_hist, [40, 75])
        if vol > q2:
            return 2                     # HIGH_VOL
        if abs(trend_strength) > 0.5:
            return 1                     # TREND
        return 0                         # QUIET


# ----------------------------------------------------------------------
# محرّك المجتمع
# ----------------------------------------------------------------------
@dataclass
class ExpertStat:
    weight: float = 1.0
    wins: int = 0
    n: int = 0
    cum_r: float = 0.0
    @property
    def hit_rate(self): return self.wins / self.n if self.n else 0.5


class Ensemble:
    def __init__(self, cfg, n_features):
        self.cfg = cfg
        self.n_features = n_features
        self.logistic = LogisticExpert(n_features, cfg.logistic_lr, cfg.logistic_l2)
        self.rls = RLSExpert(n_features, cfg.rls_lambda, cfg.rls_delta)
        self.stats: Dict[str, ExpertStat] = {k: ExpertStat() for k in EXPERT_NAMES}
        self.regime_w = np.ones((cfg.n_regimes, len(EXPERT_NAMES)))
        self.cur_regime = 0
        self.steps = 0
        self._last = {}
        self._scale = None        # معايرة online لسعة التجميع
        self._n_seen = 0

    # ---------- الإشارات ----------
    def signals(self, x, dirs) -> Dict[str, float]:
        s = rule_signals(dirs)
        s["logistic"] = self.logistic.signal(x)
        s["rls"] = self.rls.signal(x)
        return s

    def weights(self) -> np.ndarray:
        """وزن نهائي = تعلّم × سياق النظام، مع تسوية نحو التوزيع المتساوي
        وتحديد سقف للوزن الواحد، حتى لا يبتلع خبير واحد القرار كله."""
        base = np.array([self.stats[k].weight for k in EXPERT_NAMES])
        rw = self.regime_w[self.cur_regime]
        w = base * (rw / (rw.mean() + 1e-12))
        w = np.clip(w, self.cfg.weight_floor, self.cfg.weight_cap)
        w = w / w.sum()
        # تسوية: لا يزيد وزن أي خبير على max_weight ولا يقل عن min_weight
        w = (1 - self.cfg.weight_smooth) * w + self.cfg.weight_smooth / len(w)
        w = np.clip(w, self.cfg.min_weight, self.cfg.max_weight)
        return w / w.sum()

    # ---------- الدرجة ----------
    def score(self, x, dirs, regime) -> float:
        self.cur_regime = regime
        s = self.signals(x, dirs)
        w = self.weights()
        sv = np.array([s[k] for k in EXPERT_NAMES])
        agg = float(np.dot(w, sv))
        agreement = float(1.0 - np.clip(sv.std() / (np.abs(sv).mean() + 1e-6), 0, 1))
        conf = 0.55 + 0.45 * agreement
        # ---- معايرة online: نحوّل التجميع الخام إلى مقياس -100..100 ثابت الدلالة ----
        self._n_seen += 1
        a_abs = abs(agg)
        if self._scale is None:
            self._scale = max(a_abs, 0.03)
        else:
            # نتبع وسيطًا تقريبيًا للقيمة المطلقة (مقاوم للشواذ)
            self._scale += 0.01 * (a_abs - self._scale)
        z = agg / (self.cfg.score_gain * self._scale + 1e-9)
        dex = float(np.clip(100 * np.tanh(z) * conf, -100, 100))
        self._last = {"signals": s, "weights": {k: float(w[i]) for i, k in enumerate(EXPERT_NAMES)},
                      "agg": agg, "agreement": agreement, "score": dex, "regime": regime}
        return dex

    # ---------- التعلّم ----------
    def learn(self, x, dirs, realized_ret, regime, y_barrier, meta_target):
        """يُحدّث الأوزان والمعاملات من نتيجة حقيقية معروفة (بعد مرور الزمن)."""
        self.steps += 1
        s = self.signals(x, dirs)
        sign = np.sign(realized_ret)
        if sign == 0:
            return False
        # مكافأة محدودة بالخسارة/الربح الفعلي (R-like)
        reward = float(np.clip(realized_ret * 40.0, -1.0, 1.0))
        for k in EXPERT_NAMES:
            st = self.stats[k]
            r = np.sign(s[k]) * sign * abs(reward)
            st.n += 1
            if np.sign(s[k]) == sign and s[k] != 0:
                st.wins += 1
            st.cum_r += r
            # Exponentiated Gradient
            st.weight *= float(np.exp(self.cfg.hedge_eta * r))
            st.weight = float(np.clip(st.weight, self.cfg.weight_floor * 0.5, self.cfg.weight_cap))
        # إعادة تطبيع الأوزان الأساسية
        ws = np.array([self.stats[k].weight for k in EXPERT_NAMES])
        ws = ws / ws.sum() * len(EXPERT_NAMES)
        for k, v in zip(EXPERT_NAMES, ws):
            self.stats[k].weight = float(v)
        # تحديث الخبراء التعلّميين
        self.logistic.update(x, 1.0 if y_barrier > 0 else 0.0)
        self.rls.update(x, float(np.sign(realized_ret)) * min(abs(realized_ret) * 30, 1.0))
        # أوزان خاصة بالنظام
        rw = self.regime_w[regime]
        for i, k in enumerate(EXPERT_NAMES):
            rw[i] = 0.97 * rw[i] + 0.03 * (1.0 + self.stats[k].cum_r * 0.05)
        self.regime_w[regime] = np.clip(rw, 0.2, 5.0)
        return None

    def update_drift(self, correct: float) -> bool:
        return self.drift.update(correct)

    def attach_drift(self, drift):
        self.drift = drift

    def snapshot(self) -> dict:
        return {"expert_weights": {k: self.stats[k].weight for k in EXPERT_NAMES},
                "expert_hit_rate": {k: self.stats[k].hit_rate for k in EXPERT_NAMES},
                "regime_weights": self.regime_w.tolist(),
                "steps": self.steps}
