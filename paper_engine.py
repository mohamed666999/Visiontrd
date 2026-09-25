#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرّك التشغيل الحيّ — Paper/Demo فقط.

  • يقرأ السوق من بينانس، يحسب إشارة FUGU-MAX، ويدير مركزًا **ورقيًا** داخليًا.
  • يمرّر كل حدث إلى Observer (تسجيل + Telegram) — والـObserver لا يتداول.
  • إن فُعّل التنفيذ الديمو، تُرسل أوامر MARKET حقيقية على demo-fapi.binance.com
    (أموال وهمية) — وتبقى الاستراتيجيةُ نفسها هي المتحكّم الوحيد.

    python paper_engine.py --symbols BTCUSDT,ETHUSDT --interval 15m --demo-account
    python paper_engine.py --symbols ETHUSDT --interval 5m --once
"""
from __future__ import annotations
import argparse
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import Config, get_telegram, get_secrets
from data import load_market, make_synthetic_data
from features import build_features, build_expert_inputs, atr
from labels import compute_labels
from models import Ensemble, DriftMonitor, RegimeDetector, REGIME_NAMES
from strategy import MetaFilter, RiskManager, Position, make_sltp, trailing_update, target_leverage
from observer import Observer, TelegramNotifier


class SymbolTracker:
    """حالة مركز ورقية + محرّك تعلّم مستقل لكل رمز."""

    def __init__(self, symbol, cfg):
        self.symbol = symbol
        self.cfg = cfg
        self.engine = None
        self.meta = MetaFilter()
        self.rm = RiskManager(cfg)
        self.pos = Position()
        self.reg_det = RegimeDetector(cfg.regime_lookback)
        self.last_bar = None
        self.last_signal = None
        self.pending = None

    def ensure_engine(self, n_features):
        if self.engine is None or self.engine.n_features != n_features:
            self.engine = Ensemble(self.cfg, n_features)
            self.engine.attach_drift(DriftMonitor(self.cfg.drift_delta, self.cfg.drift_threshold))


def process_symbol(tr: SymbolTracker, cfg, observer: Observer, df, demo_account=False, client=None):
    """يعالج رمزًا واحدًا على أحدث البيانات؛ يعيد حدث الإشارة."""
    f = build_features(df, cfg)
    dirs = build_expert_inputs(f)
    lab = compute_labels(df, cfg)
    F = f.fillna(0.0).values
    tr.ensure_engine(F.shape[1])
    eng = tr.engine
    n = len(df)
    t = n - 1
    a = atr(df, 14).values
    bpy = 365 * 96

    # --- تعلّم تدريجي على الشمعات الجديدة منذ آخر معالجة ---
    h_max = max(cfg.h_signal, cfg.h_barrier) + 1
    start = max(600, (tr.last_bar_index if hasattr(tr, "last_bar_index") else n - 200))
    for j in range(start, n - h_max):
        rr = lab["fwd_ret"].iloc[j]
        if not np.isfinite(rr):
            continue
        d = {k: float(dirs.iloc[j][k]) for k in dirs.columns}
        reg = tr.reg_det.classify(float(f["vol_ew"].iloc[j] or 0.2),
                                  float(f["dist_ema50"].iloc[j] or 0))
        s = eng.score(F[j], d, reg)
        yb = lab["y_barrier"].iloc[j]; mt = lab["meta"].iloc[j]
        eng.learn(F[j], d, float(rr), reg,
                  float(yb) if np.isfinite(yb) else 0.0,
                  float(mt) if np.isfinite(mt) else 0.0)
        tr.meta.update(s, eng._last["agreement"], reg, float(f["vol_z"].iloc[j] or 0),
                       eng.drift.hit_rate, tr.rm.equity / 10000 - 1,
                       float(mt > 0) if np.isfinite(mt) else 0.0)
        eng.update_drift(1.0 if np.sign(rr) == np.sign(s) else 0.0)
    tr.last_bar_index = n

    # --- فحص SL/TP على آخر شمعة ---
    price = float(df["close"].iloc[t])
    lo = float(df["low"].iloc[t]); hi = float(df["high"].iloc[t])
    events = []
    if tr.pos.is_open:
        tr.pos.bars += 1
        trailing_update(tr.pos, price, a[t], cfg)
        hit_sl = (lo <= tr.pos.sl) if tr.pos.side > 0 else (hi >= tr.pos.sl)
        hit_tp = (hi >= tr.pos.tp) if tr.pos.side > 0 else (lo <= tr.pos.tp)
        exit_px, reason = None, None
        if hit_sl:
            exit_px, reason = tr.pos.sl, "SL"
        elif hit_tp:
            exit_px, reason = tr.pos.tp, "TP"
        elif tr.pos.bars >= cfg.time_stop_bars:
            exit_px, reason = price, "TIME"
        if exit_px is not None:
            pnl_pct = tr.pos.side * (exit_px / tr.pos.entry - 1.0) * tr.pos.qty
            cost = (cfg.fee_bps + cfg.slip_bps) / 10_000.0 * tr.pos.qty * 2
            pnl_pct -= cost
            tr.rm.on_trade_close(pnl_pct * tr.rm.equity)
            ev = {"symbol": tr.symbol, "timeframe": cfg.interval, "side": "LONG" if tr.pos.side > 0 else "SHORT",
                  "entry_price": round(tr.pos.entry, 6), "exit_price": round(exit_px, 6),
                  "exit_reason": reason, "pnl_pct": round(pnl_pct, 5),
                  "duration_bars": tr.pos.bars, "equity": round(tr.rm.equity, 2)}
            if demo_account and client is not None:
                try:
                    client.close_position(tr.symbol, tr.pos.side)
                    ev["demo_order"] = "closed"
                except Exception as e:
                    ev["demo_order_error"] = str(e)[:120]
            tr.pos = Position()
            observer.record("CLOSE", ev, notify=True)
            events.append(ev)

    # --- إشارة الشمعة الحالية ---
    reg = tr.reg_det.classify(float(f["vol_ew"].iloc[t] or 0.2), float(f["dist_ema50"].iloc[t] or 0))
    s = eng.score(F[t], {k: float(dirs.iloc[t][k]) for k in dirs.columns}, reg)
    agree = eng._last["agreement"]
    mp = tr.meta.proba(s, agree, reg, float(f["vol_z"].iloc[t] or 0), eng.drift.hit_rate,
                       tr.rm.equity / 10000 - 1)
    action = ("🟢 شراء (LONG)" if s >= cfg.score_entry else
              "🔴 بيع (SHORT)" if s <= -cfg.score_entry else
              "⚪ انتظار / إلغاء" if abs(s) <= cfg.score_exit else "🟡 مراقبة")
    sig_ev = {
        "symbol": tr.symbol, "timeframe": cfg.interval, "timestamp": str(df.index[t]),
        "last_price": price, "score": round(float(s), 2), "score_entry": cfg.score_entry,
        "action": action, "regime": int(reg), "regime_name": REGIME_NAMES[int(reg)],
        "agreement": round(float(agree), 3), "drift": bool(eng.drift.drift),
        "master_hit_rate": round(float(eng.drift.hit_rate), 3),
        "meta_probability": round(float(mp), 3),
        "expert_signals": {k: round(float(v), 3) for k, v in eng._last["signals"].items()},
        "expert_weights": {k: round(float(v), 3) for k, v in eng._last["weights"].items()},
    }
    tr.last_signal = sig_ev
    observer.record("SIGNAL", sig_ev, notify=False)   # الإشارات تُسجّل دائمًا، وتُرسل عند الدخول/الخروج

    # --- قرار الدخول (مركز ورقى) ---
    if (not tr.pos.is_open) and tr.rm.check(df.index[t]) and abs(s) >= cfg.score_entry \
            and (mp >= cfg.meta_threshold if tr.meta.ready else True):
        lev = abs(target_leverage(s, 0.35, cfg))
        side = 1 if s > 0 else -1
        sl, tp = make_sltp(side, price, a[t] if a[t] > 0 else price * 0.01, cfg)
        tr.pos = Position(side=side, qty=lev, entry=price, sl=sl, tp=tp, bars=0, reg=int(reg))
        open_ev = dict(sig_ev)
        open_ev.update({"side": "LONG" if side > 0 else "SHORT", "quantity": round(lev, 4),
                        "entry_price": round(price, 6), "sl": round(sl, 6), "tp": round(tp, 6),
                        "entry_reason": f"|score|={abs(s):.1f}≥{cfg.score_entry} & meta={mp:.2f}≥{cfg.meta_threshold}"})
        if demo_account and client is not None:
            try:
                client.set_leverage(tr.symbol, max(1, int(round(cfg.leverage))))
                client.set_margin_type(tr.symbol, cfg.margin_type)
                qty = round(lev * 10000 / price, client.quantity_precision(tr.symbol))
                if qty > 0:
                    o = client.market_order(tr.symbol, "BUY" if side > 0 else "SELL", qty)
                    open_ev["demo_order_id"] = o.get("orderId")
            except Exception as e:
                open_ev["demo_order_error"] = str(e)[:150]
        observer.record("OPEN", open_ev, notify=True)
        events.append(open_ev)
    return sig_ev, events


def main(argv=None):
    ap = argparse.ArgumentParser(description="FUGU-MAX — محرّك Paper/Demo")
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--demo-data", action="store_true", help="بيانات صناعية (بلا إنترنت)")
    ap.add_argument("--demo-account", action="store_true", help="إرسال أوامر فعلية على حساب بينانس الديمو")
    ap.add_argument("--once", action="store_true", help="تشغيل مرة واحدة ثم الخروج")
    a = ap.parse_args(argv)

    cfg = Config(); cfg.interval = a.interval; cfg.history_bars = a.bars
    tg = get_telegram()
    observer = Observer(cfg.path(cfg.db_path),
                        TelegramNotifier(tg["token"], tg["chat_id"], enabled=cfg.telegram_enabled))

    client = None
    if a.demo_account:
        try:
            sec = get_secrets("demo")
            from exchange import BinanceFuturesClient
            client = BinanceFuturesClient(mode="demo")
            print(f"✅ متصل بحساب الديمو: {sec['base_url']}  (مفتاح منتهٍ بـ…{sec['key'][-4:]})")
        except Exception as e:
            print(f"⚠️ تعذّر الاتصال بالديمو: {e}")
            client = None

    trackers = {s: SymbolTracker(s, cfg) for s in a.symbols.split(",")}
    observer.record("HEARTBEAT", {"symbol": ",".join(trackers), "timeframe": cfg.interval,
                                  "note": "بدء التشغيل"}, notify=False)
    print(f"🚀 FUGU-MAX يعمل | رموز: {list(trackers)} | فريم: {cfg.interval}"
          f"{' | DEMO ACCOUNT' if client else ' | DATA ONLY'}")

    while True:
        for sym, tr in trackers.items():
            try:
                if a.demo_data:
                    df = make_synthetic_data(cfg.history_bars)
                else:
                    df = load_market(sym, cfg.interval, cfg.history_bars, cfg.market_base,
                                     cfg.path(cfg.cache_dir))
                sig, evs = process_symbol(tr, cfg, observer, df, a.demo_account, client)
                tag = "⚠️انجراف" if sig["drift"] else ""
                print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {sym} {sig['last_price']:,.2f} | "
                      f"{sig['score']:+.1f} | {sig['action']} | {sig['regime_name']} | "
                      f"meta {sig['meta_probability']:.2f} {tag}")
            except Exception as e:
                observer.record("ERROR", {"symbol": sym, "timeframe": cfg.interval, "error": str(e)[:200]},
                                notify=True)
                print(f"⚠️ {sym}: {e}")
        if a.once:
            break
        time.sleep(a.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
