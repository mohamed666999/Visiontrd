#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تقرير كامل: مقارنة الأنظمة + Walk-Forward + رسم + حفظ حالة النموذج.

    python train_and_report.py --symbol BTCUSDT --interval 15m --bars 8000
    python train_and_report.py --demo --bars 6000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import load_market, make_synthetic_data
from backtest import run, metrics, walk_forward
from baselines import SCHEMES, capital_curves, pretty


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--bars", type=int, default=8000)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--outdir", default="reports")
    a = ap.parse_args(argv)

    cfg = Config(); cfg.interval = a.interval; cfg.history_bars = a.bars
    out = cfg.path(a.outdir); os.makedirs(out, exist_ok=True)

    syms = [a.symbol] if not a.symbols else a.symbols.split(",")
    report = {"interval": a.interval, "symbols": {}}
    curves_all = {}

    for sym in syms:
        try:
            df = make_synthetic_data(a.bars) if a.demo else load_market(
                sym, a.interval, a.bars, cfg.market_base, cfg.path(cfg.cache_dir))
        except Exception as e:
            print(f"⚠️ {sym}: تعذّر الجلب ({e}) — بيانات صناعية.")
            df = make_synthetic_data(a.bars)

        print(f"\n=== {sym} {a.interval} ({len(df)} شمعة) ===")
        warm = max(800, len(df) // 5)

        curves = capital_curves(df, cfg, warmup=warm, seed=7)
        curves_all[sym] = curves
        table = pd.DataFrame({n: metrics(curves[n], cfg,
                                         len(run(df, cfg, mode=m, warmup=warm, seed=7)["trades"]))
                              for n, m in SCHEMES}).T
        table = table[["total_return", "sharpe", "sortino", "max_drawdown", "calmar", "n_trades"]]
        print("\n[مقارنة الأنظمة]")
        print(pretty(table))

        print("\n[Walk-Forward — النموذج متجمّد في الاختبار]")
        wf = walk_forward(df, cfg, n_splits=4, mode="adaptive", verbose=True)

        report["symbols"][sym] = {
            "bars": int(len(df)),
            "comparison": json.loads(table.to_json(orient="index")),
            "walk_forward": json.loads(wf.to_json(orient="records")),
        }

        # --- رسم ---
        fig, ax = plt.subplots(2, 1, figsize=(13, 9))
        for name in curves.columns:
            ax[0].plot(curves.index, curves[name], lw=1.3, label=name)
        ax[0].set_yscale("log"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
        ax[0].set_title(f"FUGU-MAX — منحنيات رأس المال ({sym} {a.interval})")
        r = run(df, cfg, mode="adaptive", warmup=warm)
        ax[1].plot(df.index, r["scores"], color="#c0392b", lw=1)
        ax[1].axhline(cfg.score_entry, ls="--", c="g", alpha=.7)
        ax[1].axhline(-cfg.score_entry, ls="--", c="r", alpha=.7)
        ax[1].set_title("درجة المؤشر (-100..100) وحدود الدخول")
        ax[1].grid(alpha=.3)
        plt.tight_layout()
        png = os.path.join(out, f"fugu_max_{sym}.png")
        plt.savefig(png, dpi=130); plt.close()
        print("saved", png)

        np.savez(os.path.join(out, f"model_state_{sym}.npz"),
                 **{f"logistic_w": r["engine"].logistic.w,
                    f"rls_w": r["engine"].rls.w,
                    f"regime_w": r["engine"].regime_w,
                    **{f"w_{k}": np.array([r["engine"].stats[k].weight]) for k in r["engine"].stats}})

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n✅ التقرير محفوظ في {out}/report.json")
    return report


if __name__ == "__main__":
    main()
