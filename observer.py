#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observer مستقل: يسجّل الحالة فقط ويرسل إشعارات Telegram.
لا يعيد حساب أي شيء ولا يتداول ولا يبدّل الاستراتيجية."""
from __future__ import annotations
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class TelegramNotifier:
    """مُرسل Telegram عبر stdlib. لا يطبع التوكن ولا يخزّنه في قاعدة البيانات."""

    def __init__(self, token: str = "", chat_id: str = "", enabled: bool = True):
        self.token = token
        self.chat_id = str(chat_id or "")
        self.enabled = bool(enabled and token and chat_id)
        self._lock = threading.Lock()

    def send(self, text: str, silent: bool = False) -> bool:
        if not self.enabled:
            return False
        payload = urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "disable_notification": "true" if silent else "false",
        }).encode()
        req = Request(f"https://api.telegram.org/bot{self.token}/sendMessage",
                      data=payload, method="POST")
        try:
            with self._lock:
                with urlopen(req, timeout=15) as r:
                    return 200 <= r.status < 300
        except Exception:
            return False


class Observer:
    """يستقبل event جاهزًا ويخزّنه كما هو. لا يحسب إشارة جديدة."""

    def __init__(self, db_path: str, notifier: TelegramNotifier | None = None):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.notifier = notifier or TelegramNotifier(enabled=False)
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self):
        with self._connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                event_type TEXT NOT NULL, payload_json TEXT NOT NULL)""")
            con.execute("CREATE INDEX IF NOT EXISTS ix_events_ts ON events(timestamp)")
            con.execute("CREATE INDEX IF NOT EXISTS ix_events_symbol ON events(symbol)")

    def record(self, event_type: str, payload: dict, notify: bool = False, silent: bool = False):
        event = dict(payload)
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        event.setdefault("symbol", "UNKNOWN")
        event.setdefault("timeframe", "UNKNOWN")
        with self._connect() as con:
            con.execute(
                "INSERT INTO events(timestamp,symbol,timeframe,event_type,payload_json) VALUES(?,?,?,?,?)",
                (event["timestamp"], event["symbol"], event["timeframe"],
                 event_type, json.dumps(event, ensure_ascii=False, default=str)))
        if notify:
            self.notifier.send(format_event(event_type, event), silent=silent)

    def recent(self, limit=100):
        with self._connect() as con:
            rows = con.execute(
                "SELECT timestamp,symbol,timeframe,event_type,payload_json "
                "FROM events ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        out = []
        for r in rows:
            d = json.loads(r[4])
            d.update({"timestamp": r[0], "symbol": r[1], "timeframe": r[2], "event_type": r[3]})
            out.append(d)
        return out


def _pct(x):
    try:
        return f"{float(x):+.2%}"
    except Exception:
        return "-"


def format_event(event_type: str, e: dict) -> str:
    """رسالة كاشفة: لماذا فُتحت الصفقة، أوزان الخبراء، SL/TP، والنتيجة."""
    head = {"SIGNAL": "📡 إشارة FUGU-MAX",
            "OPEN": "🟢 فتح صفقة",
            "CLOSE": "🏁 إغلاق صفقة",
            "HEARTBEAT": "💓 نبضة FUGU-MAX",
            "ERROR": "⚠️ خطأ FUGU-MAX"}.get(event_type, f"ℹ️ {event_type}")
    lines = [head,
             f"الرمز: {e.get('symbol')} | الفريم: {e.get('timeframe')}",
             f"الوقت: {e.get('timestamp', '')}"]
    if "score" in e:
        lines += [f"القرار: {e.get('action', '-')}",
                  f"Score: {float(e['score']):+.1f} | Threshold: ±{e.get('score_entry', '-')}",
                  f"Regime: {e.get('regime_name', e.get('regime', '-'))} | "
                  f"Agreement: {float(e.get('agreement', 0)):.0%}",
                  f"Drift: {'true' if e.get('drift') else 'false'} | "
                  f"Meta: {float(e.get('meta_probability', 0)):.2f}"]
    signals = e.get("expert_signals", {})
    weights = e.get("expert_weights", {})
    if signals:
        lines.append("الخبراء (signal / weight):")
        for k in (weights or signals):
            if k in signals or k in weights:
                lines.append(f"  {k}: {float(signals.get(k, 0)):+.2f} / {float(weights.get(k, 0)):.2%}")
    if e.get("side"):
        lines += [f"المركز: {e.get('side')} | الكمية: {e.get('quantity', '-')}",
                  f"الدخول: {e.get('entry_price', '-')} | SL: {e.get('sl', '-')} | "
                  f"TP: {e.get('tp', '-')}",
                  f"سبب الدخول: {e.get('entry_reason', '-')}"]
    if event_type == "CLOSE":
        lines += [f"الخروج: {e.get('exit_price', '-')} | السبب: {e.get('exit_reason', '-')}",
                  f"PnL: {_pct(e.get('pnl_pct'))} | المدة: {e.get('duration_bars', '-')} شمعة"]
    if event_type == "ERROR":
        lines.append(f"الخطأ: {e.get('error', '-')}")
    return "\n".join(lines)
