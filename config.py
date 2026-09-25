#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUGU-MAX — إعدادات المشروع.

الأهداف التصميمية:
  • النسخة الأولى = بحث + Paper trading فقط. لا تنفيذ حقيقي في هذه الطبقة.
  • التنفيذ المتاح = Binance USDⓈ-M **Demo** (demo-fapi.binance.com) فقط.
  • لا تُخزَّن أي مفاتيح داخل الشيفرة: تُقرأ من متغيرات البيئة أو ملف .env
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
import json, os


# ----------------------------------------------------------------------
# محمّل .env بسيط (بدون تبعيات خارجية)
# ----------------------------------------------------------------------
def load_env(path: str = ".env") -> dict:
    env = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = load_env(os.environ.get("FUGU_ENV", ".env"))


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, _ENV.get(key, default))


# ----------------------------------------------------------------------
# الإعدادات الرئيسية
# ----------------------------------------------------------------------
@dataclass
class Config:
    # ---------- السوق ----------
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT",
                                                        "BNBUSDT", "XRPUSDT"])
    interval: str = "15m"
    history_bars: int = 8000
    market_base: str = "https://fapi.binance.com"        # بيانات عامة (لا تحتاج مفاتيح)

    # ---------- الميزات ----------
    zscore_window: int = 500
    vol_halflife: int = 24
    horizons: List[int] = field(default_factory=lambda: [1, 4, 12, 48])

    # ---------- التسميات ----------
    h_signal: int = 4        # أفق target الإشارة (شمعات)
    h_barrier: int = 12      # أفق Triple-Barrier
    barrier_k: float = 1.6   # مضاعف ATR للحواجز
    meta_cost_buffer: float = 1.0   # مضاعف التكلفة المطلوبة لاعتبار الصفقة "مجدية"

    # ---------- النماذج ذاتية التعلّم ----------
    rls_lambda: float = 0.995
    rls_delta: float = 50.0
    logistic_lr: float = 0.02
    logistic_l2: float = 1e-4
    hedge_eta: float = 0.25          # شدة Exponentiated Gradient
    weight_floor: float = 0.03
    weight_cap: float = 8.0
    weight_smooth: float = 0.30      # تسوية نحو التوزيع المتساوي
    max_weight: float = 0.30         # سقف وزن أي خبير
    min_weight: float = 0.02         # أرضية وزن أي خبير
    drift_delta: float = 0.0004
    drift_threshold: float = 3.5
    regime_lookback: int = 500
    n_regimes: int = 3               # 0=QUIET 1=TREND 2=HIGH_VOL

    # ---------- الاستراتيجية ----------
    score_gain: float = 1.35      # مضاعف المعايرة (يُعاير تلقائيًا)
    score_entry: float = 35.0
    score_exit: float = 8.0
    meta_threshold: float = 0.55
    target_vol: float = 0.35
    kelly_fraction: float = 0.30
    max_leverage: float = 3.0
    atr_stop_mult: float = 3.0
    atr_tp_mult: float = 6.0
    time_stop_bars: int = 96
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.25
    max_consec_losses: int = 8

    # ---------- التكاليف ----------
    fee_bps: float = 4.0
    slip_bps: float = 1.0
    funding_bars: int = 32           # 15m × 32 = 8h

    # ---------- الملفات ----------
    cache_dir: str = "data_cache"
    report_dir: str = "reports"
    db_path: str = "reports/fugu_observer.db"
    state_path: str = "reports/model_state.npz"

    # ---------- التنفيذ ----------
    mode: str = "demo"               # demo | testnet | live  (افتراضيًا demo)
    live_confirm: bool = False       # لا يُسمح بـlive إلا إذا = True
    leverage: float = 3.0
    margin_type: str = "ISOLATED"
    # ---------- Telegram (كاشف/مُبلِّغ فقط) ----------
    telegram_enabled: bool = True
    telegram_heartbeat_min: int = 60

    @property
    def root(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def path(self, *p) -> str:
        return os.path.join(self.root, *p)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


# ----------------------------------------------------------------------
# الأسرار (لا تُكتب في الشيفرة أبدًا)
# ----------------------------------------------------------------------
ENDPOINTS = {
    "demo":    "https://demo-fapi.binance.com",   # حساب الديمو (تداول وهمي بأموال وهمية)
    "testnet": "https://testnet.binancefuture.com",
    "live":    "https://fapi.binance.com",
}


def get_secrets(mode: str = None) -> dict:
    """يقرأ مفاتيح وضع معيّن من البيئة/‎.env‎."""
    mode = mode or _get("FUGU_MODE", "demo")
    pre = {"demo": "DEMO", "testnet": "TESTNET", "live": "LIVE"}[mode]
    return {
        "mode": mode,
        "base_url": ENDPOINTS[mode],
        "key": _get(f"BINANCE_{pre}_KEY"),
        "secret": _get(f"BINANCE_{pre}_SECRET"),
    }


def get_telegram() -> dict:
    return {
        "token": _get("TELEGRAM_TOKEN"),
        "chat_id": _get("TELEGRAM_CHAT_ID"),
    }


DEFAULT = Config()
