#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""الاستراتيجية: بوابة الميتا + حجم المركز + إدارة المخاطر + بنية SL/TP."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------
# بوابة الميتا (Meta Filter) — هل نصدّق هذه الإشارة؟
# ----------------------------------------------------------------------
class MetaFilter:
    """بوابة الميتا. لا تُطبَّق قبل معايرة كافية (إلا حجبت كل الصفقات)."""

    def __init__(self, lr=0.05, calib_steps=400):
        self.w = np.zeros(6); self.b = 0.0; self.lr = lr
        self.n = 0
        self.calib_steps = calib_steps

    @property
    def ready(self) -> bool:
        return self.n >= self.calib_steps

    def _phi(self, score, agree, regime, vol_z, hit, bank):
        return np.array([score / 100.0, agree, regime / 2.0,
                         float(np.tanh(vol_z)), (hit - 0.5) * 2.0, float(np.tanh(bank))])

    def proba(self, score, agree, regime, vol_z, hit, bank=0.0) -> float:
        x = self._phi(score, agree, regime, vol_z, hit, bank)
        return float(1.0 / (1.0 + np.exp(-np.clip(self.w @ x + self.b, -30, 30))))

    def update(self, score, agree, regime, vol_z, hit, bank, y):
        x = self._phi(score, agree, regime, vol_z, hit, bank)
        p = 1.0 / (1.0 + np.exp(-np.clip(self.w @ x + self.b, -30, 30)))
        g = p - y
        self.w -= self.lr * g * x
        self.b -= self.lr * g
        self.n += 1


# ----------------------------------------------------------------------
# حجم المركز
# ----------------------------------------------------------------------
def target_leverage(score, vol_ann, cfg) -> float:
    """رافعة موجّهة من الدرجة + هدف تقلبي + كيلي جزئي. النتيجة ∈ [-max, +max]."""
    direction = float(np.tanh(score / 40.0))
    vol_scalar = float(np.clip(cfg.target_vol / max(vol_ann, 1e-4), 0.1, cfg.max_leverage))
    kelly = cfg.kelly_fraction * abs(direction)
    return float(np.clip(direction * vol_scalar * kelly * 3.0, -cfg.max_leverage, cfg.max_leverage))


@dataclass
class Position:
    side: int = 0                 # +1 LONG, -1 SHORT, 0 flat
    qty: float = 0.0              # بحسب الوحدة (base asset)
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    bars: int = 0
    peak_r: float = 0.0
    reg: int = 0

    @property
    def is_open(self): return self.side != 0


# ----------------------------------------------------------------------
# مدير المخاطر
# ----------------------------------------------------------------------
class RiskManager:
    def __init__(self, cfg, equity0=10000.0):
        self.cfg = cfg
        self.equity = equity0
        self.peak = equity0
        self.day_start = equity0
        self.day = None
        self.consec_losses = 0
        self.halted = False
        self.reason = ""

    def check(self, ts) -> bool:
        d = ts.date() if hasattr(ts, "date") else ts
        if self.day != d:
            self.day = d
            self.day_start = self.equity
        dd = 1.0 - self.equity / self.peak
        if dd >= self.cfg.max_drawdown:
            self.halted, self.reason = True, f"سحب كلي {dd:.1%}"
            return False
        if (1.0 - self.equity / self.day_start) >= self.cfg.max_daily_loss:
            self.reason = "إيقاف يومي"
            return False
        if self.consec_losses >= self.cfg.max_consec_losses:
            self.reason = "خسائر متتالية"
            return False
        return True

    def on_trade_close(self, pnl: float):
        self.equity += pnl
        self.peak = max(self.peak, self.equity)
        if pnl < 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0


# ----------------------------------------------------------------------
# سياسة SL/TP
# ----------------------------------------------------------------------
def make_sltp(side: int, price: float, atr_val: float, cfg) -> tuple[float, float]:
    if side > 0:
        return price - cfg.atr_stop_mult * atr_val, price + cfg.atr_tp_mult * atr_val
    return price + cfg.atr_stop_mult * atr_val, price - cfg.atr_tp_mult * atr_val


def trailing_update(pos: Position, price: float, atr_val: float, cfg, be_at_r=1.0):
    """نقل الوقف إلى نقطة التعادل بعد ربح = 1R، ثم تتبّعه."""
    if not pos.is_open:
        return
    risk = abs(pos.entry - pos.sl) if pos.sl else cfg.atr_stop_mult * atr_val
    if risk <= 0:
        return
    r = (price - pos.entry) * pos.side / risk
    pos.peak_r = max(pos.peak_r, r)
    if pos.peak_r >= be_at_r:
        if pos.side > 0:
            pos.sl = max(pos.sl, pos.entry)
        else:
            pos.sl = min(pos.sl, pos.entry)
