#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبار قناة Telegram فقط: يرسل رسالة تعريفية واحدة ويتأكد أن الاتصال يعمل."""
from __future__ import annotations
from config import Config, get_telegram
from observer import TelegramNotifier, format_event

if __name__ == "__main__":
    cfg = Config()
    tg = get_telegram()
    print(f"التوكن موجود: {bool(tg['token'])}  |  معرّف المحادثة: {bool(tg['chat_id'])}")
    n = TelegramNotifier(tg["token"], tg["chat_id"], enabled=True)
    ok = n.send("✅ FUGU-MAX: قناة المراقبة تعمل. سيصلك هنا سبب كل صفقة، أوزان الخبراء، SL/TP، والنتيجة.")
    print("تم الإرسال:", ok)
    demo = {"symbol": "BTCUSDT", "timeframe": "15m", "score": 41.2, "action": "🟢 شراء (LONG)",
            "score_entry": 35, "regime_name": "TREND", "agreement": 0.72, "drift": False,
            "meta_probability": 0.68,
            "expert_signals": {"momentum": 0.6, "trend": 0.7, "reversion": -0.2, "rls": 0.5},
            "expert_weights": {"momentum": 0.18, "trend": 0.22, "reversion": 0.12, "rls": 0.14},
            "side": "LONG", "quantity": 0.0031, "entry_price": 68000.0, "sl": 66530.0, "tp": 70370.0,
            "entry_reason": "|score|=41.2≥35 & meta=0.68≥0.55"}
    n.send(format_event("OPEN", demo))
