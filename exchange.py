#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
عميل بينانس الآجلة (USDⓈ-M) موقّع — عبر stdlib فقط (hmac + urllib).

يدعم:
    demo    : https://demo-fapi.binance.com   ← الافتراضي (أموال وهمية)
    testnet : https://testnet.binancefuture.com
    live    : https://fapi.binance.com  (يتطلب FUGU_MODE=live و LIVE_CONFIRM=true)

الأمان:
    • لا يُرسل أي أمر حقيقي إلا إذا mode="demo" (أو "live" مع تأكيد صريح).
    • المفاتيح تُقرأ من البيئة/‎.env‎ فقط — لا تُكتب في الشيفرة أبدًا.
"""
from __future__ import annotations
import hmac, hashlib, time, json
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from config import get_secrets, ENDPOINTS


class BinanceFuturesClient:
    def __init__(self, mode: str = "demo", key: str = "", secret: str = "",
                 live_confirm: bool = False):
        if mode == "live" and not live_confirm:
            raise RuntimeError("الوضع الحقيقي معطّل: يتطلب LIVE_CONFIRM=true صريحًا.")
        self.mode = mode
        self.base = ENDPOINTS[mode]
        sec = get_secrets(mode) if not (key and secret) else {"key": key, "secret": secret}
        self.key = key or sec["key"]
        self.secret = secret or sec["secret"]
        if not self.key or not self.secret:
            raise RuntimeError(f"مفاتيح '{mode}' غير موجودة — ضعها في .env")

    # ------------------------------------------------------------------
    def _sign(self, params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        qs = urlencode(params)
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        return params, qs + f"&signature={sig}"

    def _req(self, method: str, path: str, params: Optional[dict] = None,
             signed: bool = False):
        params = params or {}
        if signed:
            params, qs = self._sign(params)
        else:
            qs = urlencode(params)
        url = f"{self.base}{path}"
        if qs:
            url += "?" + qs
        req = Request(url, method=method)
        if signed:
            req.add_header("X-MBX-APIKEY", self.key)
        try:
            with urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode() or "{}")
        except HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {e.code} من {path}: {body[:300]}") from e

    # ------------------------------------------------------------------
    def ping(self):  return self._req("GET", "/fapi/v1/ping")
    def time(self):  return self._req("GET", "/fapi/v1/time")
    def exchange_info(self): return self._req("GET", "/fapi/v1/exchangeInfo")

    def balance(self):        return self._req("GET", "/fapi/v2/balance", signed=True)
    def account(self):        return self._req("GET", "/fapi/v2/account", signed=True)
    def positions(self):      return self._req("GET", "/fapi/v2/positionRisk", signed=True)

    def mark_price(self, symbol):
        return self._req("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})

    def set_leverage(self, symbol, leverage):
        return self._req("POST", "/fapi/v1/leverage",
                         {"symbol": symbol, "leverage": int(leverage)}, signed=True)

    def set_margin_type(self, symbol, margin="ISOLATED"):
        try:
            return self._req("POST", "/fapi/v1/marginType",
                             {"symbol": symbol, "marginType": margin}, signed=True)
        except RuntimeError:
            return {}   # غالبًا "already" — نتجاهل

    def market_order(self, symbol, side, qty, reduce_only=False):
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._req("POST", "/fapi/v1/order", params, signed=True)

    def close_position(self, symbol, side_of_pos):
        """إغلاق كامل المركز الحالي بالاتجاه المعاكس (reduceOnly)."""
        side = "SELL" if side_of_pos > 0 else "BUY"
        qty = self.position_qty(symbol)
        if abs(qty) < 1e-9:
            return {}
        return self.market_order(symbol, side, abs(qty), reduce_only=True)

    def position_qty(self, symbol) -> float:
        rows = self.positions()
        if isinstance(rows, dict):
            rows = rows.get("positionRisk", rows)
        for p in rows:
            if p.get("symbol") == symbol:
                return float(p.get("positionAmt", 0.0))
        return 0.0

    def quantity_precision(self, symbol) -> int:
        """عدد الخانات العشرية المسموحة للكمية (من exchangeInfo)."""
        info = self.exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        step = f.get("stepSize", "0.001")
                        return max(0, len(step.split(".")[1].rstrip("0"))) if "." in step else 0
        return 3
