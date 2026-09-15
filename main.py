"""
CAPE — Cross-Asset Predictive Engine (Quant Workstation Edition)
================================================================
Production-ready single-file deployment.

Architecture (isolated asyncio tasks):
  [1] data_collector_loop   -> WebSocket ingest -> memory buffer (deque)
  [2] flush_data_to_db      -> batched executemany write + 24h self-cleanup
  [3] analyzer_loop         -> CAPE quantitative engine (pause-aware)
  [4] telegram_bot_loop     -> aiogram v3 admin Quant Workstation
  [5] signal_evaluator_loop -> evaluates pending AI LAB signals via CAPE gate

Design guarantees:
  * SQLite in WAL mode -> no "database is locked".
  * Batched writes (executemany) + retention purge.
  * errors.log is UTF-8 and retrievable via Telegram.
  * Admin-only Telegram C&C.
  * Atomic DB migration (integrity_check + os.replace).
  * Runtime-tunable parameters (no redeploy needed).
  * Pause/Resume engine without stopping data collection.
"""

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
import asyncio
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import aiohttp
    import aiosqlite
    import numpy as np
    import pandas as pd
    from aiogram import Bot, Dispatcher, F, Router
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode
    from aiogram.filters import Command
    from aiogram.types import (
        CallbackQuery,
        FSInputFile,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Message,
    )
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}. Run: pip install aiohttp aiosqlite numpy pandas aiogram")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1.1 EMBEDDED SECRETS (Railway-ready)
# ---------------------------------------------------------------------------
EMBEDDED_BOT_TOKEN = "8658061104:AAGWM9ghPYD_XiYq3uzm0f0JVQ55Edywo6k"
EMBEDDED_ADMIN_ID  = 6033203084

API_KEYS = {
    "demo": {
        "key":    "uQozmWB6O6ZvdEPU7GCoTjFTdJWnhIGDsMuEqgI99wIWnS11EZCU7ArCvDUOTtwj",
        "secret": "WLi3YMbZWhXEicrAuUeODNiGnjlhvYgO9GlN6HaDlb9FAXiUxO1CprlVjKvCqRwK",
    },
    "testnet": {"key": "", "secret": ""},
    "live": {
        "key":    "IX7kLH0ssWHP5TpYMUGcp0pzq4LX4Lqi7m4XtlqMkkq6DCZAsLhoeYZ3533jJFF4",
        "secret": "LmICnpSpMxL1riv4RfIf0HBGRfhDTP5JhDUYdlPSukpqV7kDTonrZ0j3DWp1a7hU",
    },
}
MODE         = "demo"
LIVE_CONFIRM = False

# --- user config -----------------------------------------------------------
BOT_TOKEN        = os.getenv("CAPE_BOT_TOKEN", EMBEDDED_BOT_TOKEN)
ADMIN_ID         = int(os.getenv("CAPE_ADMIN_ID", str(EMBEDDED_ADMIN_ID)))

DB_PATH          = os.getenv("CAPE_DB", "cape.db")
TEMP_DB_PATH     = "cape_temp.db"
BACKUP_DB_PATH   = "cape_backup.db"
ERROR_LOG_PATH   = "errors.log"

SYMBOLS          = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "VETUSDT"]

WS_BASE          = "wss://stream.binance.com:9443/stream?streams="

BUFFER_FLUSH_INTERVAL    = 5
ANALYZER_INTERVAL        = 30
SIGNAL_EVAL_INTERVAL     = 10
WS_RECONNECT_BACKOFF_MAX = 60

TICKS_RETENTION_HOURS    = 24
TICKS_RETENTION_MS       = TICKS_RETENTION_HOURS * 60 * 60 * 1000

EWMA_VOL_SPAN        = 60
MAX_LAG              = 10
SIGNIFICANCE_K       = 1.8
SHOCK_TRIGGER_SIGMA  = 2.0
FOLLOWER_GAP_MIN_Z   = 0.5
ANALYZER_LOOKBACK_M  = 90
PREDICTIVE_ALLOW_MIN = 60

# ---------------------------------------------------------------------------
# 1.2 RUNTIME TUNABLE PARAMETERS (Quant Workstation)
#     These mirror the constants above but can be changed live from Telegram
#     without touching the code or redeploying.
# ---------------------------------------------------------------------------
RUNTIME: dict[str, Any] = {
    "predictive_allow_min": PREDICTIVE_ALLOW_MIN,   # 0-100
    "follower_gap_min_z":   FOLLOWER_GAP_MIN_Z,     # sigma
    "max_lag":              MAX_LAG,                # bars
}
ENGINE_PAUSED: bool = False

# ---------------------------------------------------------------------------
# 2. UTF-8 SILENT FILE LOGGER
# ---------------------------------------------------------------------------
logger = logging.getLogger("cape")
logger.setLevel(logging.INFO)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_console)

_file = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
_file.setLevel(logging.WARNING)
_file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_file)


def log_error(exc: BaseException, ctx: str = ""):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(f"{ctx} :: {exc}\n{tb}")


# ---------------------------------------------------------------------------
# 3. GLOBAL RUNTIME STATE
# ---------------------------------------------------------------------------
_start_time: float = time.time()

tick_buffer: deque[tuple] = deque(maxlen=200_000)
buffer_lock = asyncio.Lock()

pair_relations_cache: dict[str, dict] = {}
market_regime_cache: dict[str, float] = {"stability": 0.0, "lambda1": 0.0, "mp_bound": 0.0}
_cache_lock = asyncio.Lock()

pending_signals_queue: asyncio.Queue = asyncio.Queue()


# ---------------------------------------------------------------------------
# 4. DATABASE LAYER
# ---------------------------------------------------------------------------
DDL = [
    """
    CREATE TABLE IF NOT EXISTS ticks (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol  TEXT    NOT NULL,
        ts      INTEGER NOT NULL,
        price   REAL    NOT NULL,
        volume  REAL    DEFAULT 0
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ticks_sym_ts ON ticks(symbol, ts);",
    """
    CREATE TABLE IF NOT EXISTS signals (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        ts               INTEGER NOT NULL,
        algorithm        TEXT,
        symbol           TEXT    NOT NULL,
        side             TEXT    NOT NULL,
        algo_score       INTEGER DEFAULT 0,
        predictive_score REAL    DEFAULT 0,
        decision         TEXT    DEFAULT 'PENDING',
        details          TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);",
    """
    CREATE TABLE IF NOT EXISTS pair_relations (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            INTEGER NOT NULL,
        follower      TEXT NOT NULL,
        leader        TEXT NOT NULL,
        best_lag      INTEGER,
        corr_forward  REAL,
        corr_backward REAL,
        asymmetry     REAL,
        beta          REAL
    );
    """,
]


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    return db


async def init_db():
    db = await get_db()
    try:
        for stmt in DDL:
            await db.execute(stmt)
        await db.commit()
    finally:
        await db.close()


async def ensure_tables_in_file(path: str):
    db = await aiosqlite.connect(path)
    try:
        await db.execute("PRAGMA journal_mode=WAL;")
        for stmt in DDL:
            await db.execute(stmt)
        await db.commit()
    finally:
        await db.close()


async def prune_old_ticks() -> int:
    """Delete ticks older than TICKS_RETENTION_HOURS. Returns rows deleted."""
    db = await get_db()
    try:
        cutoff = int(time.time() * 1000) - TICKS_RETENTION_MS
        # count first so we can report to Telegram
        cur = await db.execute("SELECT COUNT(*) FROM ticks WHERE ts < ?", (cutoff,))
        row = await cur.fetchone()
        n = row[0] if row else 0
        await db.execute("DELETE FROM ticks WHERE ts < ?", (cutoff,))
        await db.commit()
        return int(n)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# 5. DATA COLLECTOR (WebSocket)
# ---------------------------------------------------------------------------
def _build_ws_url() -> str:
    streams = "/".join(f"{s.lower()}@aggTrade" for s in SYMBOLS)
    return WS_BASE + streams


async def _ws_session():
    url = _build_ws_url()
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, heartbeat=20, max_msg_size=2**20) as ws:
            logger.info(f"[collector] connected: {len(SYMBOLS)} streams")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    data = payload.get("data") or {}
                    sym = data.get("s")
                    price = data.get("p")
                    qty = data.get("q")
                    ts = data.get("T") or data.get("E")
                    if not (sym and price and ts):
                        continue
                    async with buffer_lock:
                        tick_buffer.append((sym, int(ts), float(price), float(qty or 0.0)))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(f"WS error: {ws.exception()}")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    raise ConnectionError("WS closed by server")


async def data_collector_loop():
    backoff = 1
    while True:
        try:
            await _ws_session()
            backoff = 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_error(e, "collector")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, WS_RECONNECT_BACKOFF_MAX)


# ---------------------------------------------------------------------------
# 6. BUFFER FLUSH + SELF-CLEANUP
# ---------------------------------------------------------------------------
async def flush_data_to_db():
    db = await get_db()
    try:
        while True:
            await asyncio.sleep(BUFFER_FLUSH_INTERVAL)

            if tick_buffer:
                async with buffer_lock:
                    rows = list(tick_buffer)
                    tick_buffer.clear()
                try:
                    await db.executemany(
                        "INSERT INTO ticks (symbol, ts, price, volume) VALUES (?, ?, ?, ?)",
                        rows,
                    )
                    await db.commit()

                    cutoff_time = int(time.time() * 1000) - TICKS_RETENTION_MS
                    await db.execute("DELETE FROM ticks WHERE ts < ?", (cutoff_time,))
                    await db.commit()
                except Exception as e:
                    log_error(e, "flush_data_to_db")
            else:
                try:
                    cutoff_time = int(time.time() * 1000) - TICKS_RETENTION_MS
                    await db.execute("DELETE FROM ticks WHERE ts < ?", (cutoff_time,))
                    await db.commit()
                except Exception as e:
                    log_error(e, "cleanup_only")
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# 7. CAPE QUANTITATIVE ENGINE
# ---------------------------------------------------------------------------
def ewma_vol(returns: pd.Series, span: int = EWMA_VOL_SPAN) -> pd.Series:
    return returns.ewm(span=span, adjust=False).std()


def standardized_shock(ret: pd.Series, vol: pd.Series) -> pd.Series:
    v = vol.replace(0, np.nan)
    return (ret / v).fillna(0.0)


def cross_correlation_lead_lag(
    leader: pd.Series, follower: pd.Series, max_lag: int = MAX_LAG
) -> dict[str, Any]:
    df = pd.concat([leader.rename("L"), follower.rename("F")], axis=1).dropna()
    if len(df) < max_lag * 5 + 10:
        return {"best_lag": 0, "corr_forward": 0.0, "corr_backward": 0.0,
                "asymmetry": 0.0, "beta": 0.0, "valid": False}

    L = df["L"]
    F = df["F"]
    n = len(df)
    sig = SIGNIFICANCE_K / math.sqrt(n)

    fwd, bwd = [], []
    for tau in range(1, max_lag + 1):
        c_f = L.shift(tau).corr(F)
        c_b = F.shift(tau).corr(L)
        if pd.notna(c_f):
            fwd.append((tau, c_f))
        if pd.notna(c_b):
            bwd.append((tau, c_b))

    if not fwd or not bwd:
        return {"best_lag": 0, "corr_forward": 0.0, "corr_backward": 0.0,
                "asymmetry": 0.0, "beta": 0.0, "valid": False}

    best_lag, best_corr = max(fwd, key=lambda x: abs(x[1]))
    lambda_lf = float(np.mean([c for _, c in fwd if abs(c) > sig]) if any(abs(c) > sig for _, c in fwd) else 0.0)
    lambda_fl = float(np.mean([c for _, c in bwd if abs(c) > sig]) if any(abs(c) > sig for _, c in bwd) else 0.0)

    denom = abs(lambda_lf) + abs(lambda_fl) + 1e-9
    asymmetry = (lambda_lf - lambda_fl) / denom

    varL = float(L.var())
    beta = float(L.cov(F) / varL) if varL > 0 else 0.0

    return {
        "best_lag": int(best_lag),
        "corr_forward": float(best_corr),
        "corr_backward": float(np.max([abs(c) for _, c in bwd])),
        "asymmetry": float(asymmetry),
        "beta": beta,
        "valid": abs(best_corr) > sig and asymmetry > 0,
    }


def follower_catchup_gap(follower_ret: pd.Series, leader_ret: pd.Series,
                         beta: float, window: int = 60) -> float:
    df = pd.concat([follower_ret.rename("F"), leader_ret.rename("L")], axis=1).dropna().tail(window)
    if len(df) < 10:
        return 0.0
    resid = df["F"] - beta * df["L"]
    sd = float(resid.std()) or 1e-9
    return float(resid.iloc[-1] / sd)


def spectral_stability(returns: pd.DataFrame) -> dict[str, float]:
    df = returns.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[1] < 2 or df.shape[0] < 10:
        return {"lambda1": 0.0, "mp_bound": 0.0, "stability": 0.0}
    corr = df.corr().fillna(0.0).values
    ev = np.linalg.eigvalsh(corr)
    lam1 = float(ev[-1])
    N, T = df.shape[1], df.shape[0]
    q = N / max(T, 1)
    mp = float((1.0 + math.sqrt(q)) ** 2)
    return {"lambda1": lam1, "mp_bound": mp, "stability": lam1 / mp if mp > 0 else 0.0}


def predictive_score(
    corr_forward: float, asymmetry: float, gap_z: float, stability: float
) -> float:
    a = np.clip(abs(corr_forward), 0.0, 1.0)
    b = np.clip((asymmetry + 1.0) / 2.0, 0.0, 1.0)
    c = float(np.tanh(gap_z / 2.0))
    d = np.clip(stability, 0.0, 2.0) / 2.0
    raw = 0.35 * a + 0.25 * b + 0.25 * (c + 1.0) / 2.0 + 0.15 * d
    return round(raw * 100.0, 2)


# ---------------------------------------------------------------------------
# 8. ANALYZER (brain) — respects ENGINE_PAUSED
# ---------------------------------------------------------------------------
async def _load_recent_ticks(lookback_minutes: int) -> pd.DataFrame:
    cutoff = int((time.time() - lookback_minutes * 60) * 1000)
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT symbol, ts, price FROM ticks WHERE ts >= ? ORDER BY ts ASC",
            (cutoff,),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "price"])
    return pd.DataFrame(rows, columns=["symbol", "ts", "price"])


def _compute_returns_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(index="ts", columns="symbol", values="price", aggfunc="last")
    pivot.index = pd.to_datetime(pivot.index, unit="ms", utc=True)
    bars = pivot.resample("1min").last()
    bars = bars.ffill(limit=1)
    rets = np.log(bars).diff()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


async def run_analysis_cycle():
    df = await _load_recent_ticks(ANALYZER_LOOKBACK_M)
    if df.empty:
        return

    rets = _compute_returns_matrix(df)
    if rets.shape[1] < 2 or rets.shape[0] < 30:
        return

    regime = spectral_stability(rets)
    async with _cache_lock:
        market_regime_cache.update(regime)

    symbols = list(rets.columns)
    new_relations: dict[str, dict] = {}
    live_max_lag = int(RUNTIME.get("max_lag", MAX_LAG))
    for follower in symbols:
        best: Optional[dict] = None
        for leader in symbols:
            if leader == follower:
                continue
            info = cross_correlation_lead_lag(rets[leader], rets[follower], max_lag=live_max_lag)
            if not info["valid"]:
                continue
            if best is None or abs(info["corr_forward"]) > abs(best["corr_forward"]):
                info["leader"] = leader
                info["follower"] = follower
                best = info
        if best:
            new_relations[follower] = best

    async with _cache_lock:
        pair_relations_cache.clear()
        pair_relations_cache.update(new_relations)

    ts_now = int(time.time() * 1000)
    rows = [
        (ts_now, r["follower"], r["leader"], r["best_lag"],
         r["corr_forward"], r["corr_backward"], r["asymmetry"], r["beta"])
        for r in new_relations.values()
    ]
    if rows:
        db = await get_db()
        try:
            await db.executemany(
                "INSERT INTO pair_relations (ts, follower, leader, best_lag, corr_forward, "
                "corr_backward, asymmetry, beta) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()
        except Exception as e:
            log_error(e, "persist_pair_relations")
        finally:
            await db.close()

    logger.info(
        f"[analyzer] {len(new_relations)} relations | "
        f"lambda1={regime['lambda1']:.2f} mp={regime['mp_bound']:.2f} "
        f"stability={regime['stability']:.2f}"
    )


async def analyzer_loop():
    while True:
        try:
            if not ENGINE_PAUSED:
                await run_analysis_cycle()
            else:
                logger.info("[analyzer] paused — skipping cycle")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_error(e, "analyzer_loop")
        await asyncio.sleep(ANALYZER_INTERVAL)


# ---------------------------------------------------------------------------
# 9. SIGNAL EVALUATOR (predictive gatekeeper) — uses RUNTIME params
# ---------------------------------------------------------------------------
async def _fetch_pending_signals(limit: int = 20) -> list[tuple]:
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, symbol, side, algo_score FROM signals "
            "WHERE decision='PENDING' ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        return await cur.fetchall()
    finally:
        await db.close()


async def _update_signal(sig_id: int, predictive: float, decision: str, details: dict):
    db = await get_db()
    try:
        await db.execute(
            "UPDATE signals SET predictive_score=?, decision=?, details=? WHERE id=?",
            (predictive, decision, json.dumps(details, default=str), sig_id),
        )
        await db.commit()
    finally:
        await db.close()


async def evaluate_signal(symbol: str, side: str) -> dict[str, Any]:
    async with _cache_lock:
        relation = pair_relations_cache.get(symbol)
        regime = dict(market_regime_cache)

    details: dict[str, Any] = {
        "relation": relation,
        "regime": regime,
        "reason": None,
    }

    if relation is None:
        details["reason"] = "no_asymmetric_leader"
        return {"score": 0.0, "decision": "WAIT", "details": details}

    leader = relation["leader"]
    corr_fwd = relation["corr_forward"]
    asym = relation["asymmetry"]
    beta = relation["beta"]
    stability = regime.get("stability", 0.0)

    df = await _load_recent_ticks(ANALYZER_LOOKBACK_M)
    rets = _compute_returns_matrix(df)
    if rets.empty or symbol not in rets.columns or leader not in rets.columns:
        details["reason"] = "insufficient_data"
        return {"score": 0.0, "decision": "WAIT", "details": details}

    gap_z = follower_catchup_gap(rets[symbol], rets[leader], beta)
    details["gap_z"] = gap_z

    vol_leader = ewma_vol(rets[leader]).iloc[-1] or 1e-9
    shock_z = float(rets[leader].iloc[-1] / vol_leader)
    details["shock_z_leader"] = shock_z

    score = predictive_score(corr_fwd, asym, gap_z, stability)
    details["predictive_score"] = score

    live_gap_min  = float(RUNTIME.get("follower_gap_min_z", FOLLOWER_GAP_MIN_Z))
    live_allow    = float(RUNTIME.get("predictive_allow_min", PREDICTIVE_ALLOW_MIN))

    if side.upper() == "LONG":
        if gap_z >= live_gap_min and score >= live_allow and shock_z > 0:
            decision = "ALLOW"
        else:
            decision = "WAIT"
            details["reason"] = (
                "follower_not_lagging" if gap_z < live_gap_min else "low_score"
            )
    else:
        if gap_z <= -live_gap_min and score >= live_allow and shock_z < 0:
            decision = "ALLOW"
        else:
            decision = "WAIT"
            details["reason"] = "leader_not_weak_enough"

    return {"score": score, "decision": decision, "details": details}


async def signal_evaluator_loop():
    while True:
        try:
            if not ENGINE_PAUSED:
                pending = await _fetch_pending_signals(limit=30)
                for sig_id, symbol, side, _algo_score in pending:
                    result = await evaluate_signal(symbol, side)
                    await _update_signal(sig_id, result["score"], result["decision"], result["details"])
                    logger.info(
                        f"[gate] sig#{sig_id} {symbol} {side} -> {result['decision']} "
                        f"({result['score']:.1f})"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_error(e, "signal_evaluator_loop")
        await asyncio.sleep(SIGNAL_EVAL_INTERVAL)


# ---------------------------------------------------------------------------
# 10. TELEGRAM QUANT WORKSTATION
# ---------------------------------------------------------------------------
router = Router()


def _admin_only(message_or_cb) -> bool:
    return message_or_cb.from_user and message_or_cb.from_user.id == ADMIN_ID


def _dashboard_kb() -> InlineKeyboardMarkup:
    pause_label = "▶️ Resume Engine" if ENGINE_PAUSED else "⏸️ Pause Engine"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 System Status",     callback_data="status"),
         InlineKeyboardButton(text="🌡 Market Regime",     callback_data="regime")],
        [InlineKeyboardButton(text="🎯 Latest Decisions",  callback_data="decisions"),
         InlineKeyboardButton(text="👑 Top Lead/Lag Pairs", callback_data="top_pairs")],
        [InlineKeyboardButton(text="💾 Download DB",       callback_data="dl_db"),
         InlineKeyboardButton(text="📜 View Errors",       callback_data="view_errors")],
        [InlineKeyboardButton(text="🧹 Prune Old Ticks",   callback_data="prune_ticks"),
         InlineKeyboardButton(text="🗑 Clear Errors",      callback_data="clear_errors")],
        [InlineKeyboardButton(text="⚙️ Settings",          callback_data="settings"),
         InlineKeyboardButton(text=pause_label,            callback_data="toggle_pause")],
        [InlineKeyboardButton(text="🔄 Refresh",           callback_data="refresh")],
    ])


def _settings_kb() -> InlineKeyboardMarkup:
    p = RUNTIME["predictive_allow_min"]
    g = RUNTIME["follower_gap_min_z"]
    m = RUNTIME["max_lag"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"➕ Allow Min +5  (now {p:.0f})", callback_data="set_allow_p5"),
         InlineKeyboardButton(text=f"➖ Allow Min -5  (now {p:.0f})", callback_data="set_allow_m5")],
        [InlineKeyboardButton(text=f"➕ Gap σ +0.25  (now {g:.2f})",  callback_data="set_gap_p"),
         InlineKeyboardButton(text=f"➖ Gap σ -0.25  (now {g:.2f})",  callback_data="set_gap_m")],
        [InlineKeyboardButton(text=f"➕ Max Lag +1  (now {m})",       callback_data="set_lag_p"),
         InlineKeyboardButton(text=f"➖ Max Lag -1  (now {m})",       callback_data="set_lag_m")],
        [InlineKeyboardButton(text="♻️ Reset Defaults",              callback_data="set_reset")],
        [InlineKeyboardButton(text="⬅️ Back",                        callback_data="refresh")],
    ])


# --- texts -----------------------------------------------------------------
def _regime_text(stability: float, lam1: float, mp: float) -> str:
    if stability <= 0.0:
        head = "⚪️ لا توجد بيانات كافية بعد"
        note = "انتظر دقيقة أخرى لتجمع العينات."
    elif stability < 0.70:
        head = "🔴 تفكك عشوائي (Decoupled)"
        note = "العلاقات ضعيفة → تجنّب استراتيجيات اللحاق حاليًا."
    elif stability < 1.00:
        head = "🟡 شبه متماسك (Weak Coherence)"
        note = "العلاقات موجودة لكن ضعيفة → قلّل حجم المخاطرة."
    elif stability < 1.50:
        head = "🟢 متناغم (Coherent Market Mode)"
        note = "بيئة مثالية لتداول الـ Lead/Lag catch-up."
    else:
        head = "🔥 نمط قوي جدًا (Strong Regime)"
        note = "ارتباط جماعي عالٍ — إشارات اللحاق عادةً أعلى دقة."

    return (
        "🌡 <b>Market Regime</b>\n"
        f"الحالة: {head}\n"
        f"λ₁ = <code>{lam1:.3f}</code>  |  "
        f"MP bound = <code>{mp:.3f}</code>  |  "
        f"stability = <code>{stability:.3f}</code>\n"
        f"<i>{note}</i>"
    )


_REASON_AR = {
    "no_asymmetric_leader": "لا يوجد قائد غير متماثل",
    "insufficient_data":    "بيانات غير كافية",
    "follower_not_lagging": "التابع لم يتأخر بعد",
    "low_score":            "درجة الثقة منخفضة",
    "leader_not_weak_enough": "القائد ليس ضعيفًا بما يكفي",
    "":                     "—",
}


async def _system_status_text() -> str:
    uptime_s = int(time.time() - _start_time)
    h, rem = divmod(uptime_s, 3600)
    m, s = divmod(rem, 60)

    async with buffer_lock:
        buf_len = len(tick_buffer)

    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    ticks_count = signals_count = 0
    db = await get_db()
    try:
        c1 = await db.execute("SELECT COUNT(*) FROM ticks")
        ticks_count = (await c1.fetchone())[0]
        c2 = await db.execute("SELECT COUNT(*) FROM signals")
        signals_count = (await c2.fetchone())[0]
    except Exception as e:
        log_error(e, "status_query")
    finally:
        await db.close()

    async with _cache_lock:
        rel_n = len(pair_relations_cache)
        regime = dict(market_regime_cache)

    return (
        "🛡 <b>CAPE — System Status</b>\n"
        f"⏱ Uptime: <code>{h}h {m}m {s}s</code>\n"
        f"🗄 DB size: <code>{db_size/1024:.1f} KB</code>\n"
        f"📈 Ticks recorded: <code>{ticks_count:,}</code>\n"
        f"🧠 Signals stored: <code>{signals_count:,}</code>\n"
        f"🧺 Buffer in RAM: <code>{buf_len}</code>\n"
        f"🔗 Pair relations cached: <code>{rel_n}</code>\n"
        f"📐 λ1: <code>{regime.get('lambda1', 0):.3f}</code> | "
        f"MP: <code>{regime.get('mp_bound', 0):.3f}</code> | "
        f"stability: <code>{regime.get('stability', 0):.3f}</code>\n"
        f"🎚 Allow Min: <code>{RUNTIME['predictive_allow_min']:.0f}</code> | "
        f"Gap σ: <code>{RUNTIME['follower_gap_min_z']:.2f}</code> | "
        f"Max Lag: <code>{RUNTIME['max_lag']}</code>\n"
        f"🧹 Retention: <code>{TICKS_RETENTION_HOURS}h</code> | "
        f"🎛 Mode: <code>{MODE}</code> | "
        f"⚙️ Engine: <code>{'PAUSED' if ENGINE_PAUSED else 'RUNNING'}</code>"
    )


async def _latest_decisions_text(limit: int = 5) -> str:
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT ts, algorithm, symbol, side, algo_score, predictive_score, decision, details "
            "FROM signals ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
    finally:
        await db.close()

    if not rows:
        return "🎯 <b>Latest Decisions</b>\nلا توجد إشارات بعد. جرّب <code>/inject SOLUSDT LONG</code>"

    lines = ["🎯 <b>Latest Decisions</b>"]
    for i, (ts, algo, sym, side, a_score, p_score, decision, details_json) in enumerate(rows, 1):
        try:
            det = json.loads(details_json) if details_json else {}
        except Exception:
            det = {}
        reason_key = (det or {}).get("reason") or ""
        reason_ar = _REASON_AR.get(reason_key, reason_key or "—")
        emoji = {"ALLOW": "✅", "WAIT": "⏸️", "PENDING": "⏳"}.get(decision, "•")
        when = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")
        lines.append(
            f"{i}. {emoji} <b>{sym}</b> [{side}] → <b>{decision}</b> "
            f"(Algo {a_score} | CAPE {p_score:.0f})\n"
            f"   ↳ {reason_ar}  ·  <code>{when}</code>"
        )
    return "\n".join(lines)


async def _top_pairs_text(limit: int = 3) -> str:
    async with _cache_lock:
        relations = list(pair_relations_cache.values())
    if not relations:
        return "👑 <b>Top Lead/Lag Pairs</b>\nلا توجد علاقات مرصودة بعد."

    relations.sort(key=lambda r: abs(r.get("corr_forward", 0.0)), reverse=True)
    medals = ["🥇", "🥈", "🥉", "🏅", "🏅"]
    lines = ["👑 <b>Top Lead/Lag Pairs</b>"]
    for i, r in enumerate(relations[:limit]):
        lines.append(
            f"{medals[i]} <b>{r['leader']}</b> يقود <b>{r['follower']}</b>\n"
            f"   ⏱ Lag: <code>{r['best_lag']} bars</code> | "
            f"ρ(fwd): <code>{r['corr_forward']:.3f}</code> | "
            f"ρ(bwd): <code>{r['corr_backward']:.3f}</code>\n"
            f"   ↔ Asymmetry: <code>{r['asymmetry']:.2f}</code> | "
            f"β: <code>{r['beta']:.2f}</code>"
        )
    return "\n".join(lines)


# --- commands --------------------------------------------------------------
@router.message(Command("start"))
async def cmd_start(message: Message):
    if not _admin_only(message):
        return
    await message.answer("🛡 <b>CAPE — Quant Workstation</b>", reply_markup=_dashboard_kb())


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _admin_only(message):
        return
    await message.answer(await _system_status_text(), reply_markup=_dashboard_kb())


@router.message(Command("inject"))
async def cmd_inject(message: Message):
    if not _admin_only(message):
        return
    parts = (message.text or "").split()
    if len(parts) < 3:
        await message.reply("Usage: /inject SYMBOL LONG|SHORT")
        return
    symbol, side = parts[1].upper(), parts[2].upper()
    if side not in ("LONG", "SHORT"):
        await message.reply("Side must be LONG or SHORT")
        return
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO signals (ts, algorithm, symbol, side, algo_score, decision) "
            "VALUES (?, ?, ?, ?, 100, 'PENDING')",
            (int(time.time() * 1000), "AI_LAB_DEMO", symbol, side),
        )
        await db.commit()
    finally:
        await db.close()
    await message.reply(f"✅ Injected {side} {symbol} (Score=100) -> pending CAPE evaluation")


# --- callbacks -------------------------------------------------------------
@router.callback_query(F.data == "refresh")
async def cb_refresh(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    await cb.message.edit_text(await _system_status_text(), reply_markup=_dashboard_kb())
    await cb.answer("Refreshed")


@router.callback_query(F.data == "status")
async def cb_status(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    await cb.message.edit_text(await _system_status_text(), reply_markup=_dashboard_kb())
    await cb.answer()


@router.callback_query(F.data == "regime")
async def cb_regime(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    async with _cache_lock:
        regime = dict(market_regime_cache)
    await cb.message.answer(
        _regime_text(regime.get("stability", 0.0),
                     regime.get("lambda1", 0.0),
                     regime.get("mp_bound", 0.0)),
        reply_markup=_dashboard_kb(),
    )
    await cb.answer()


@router.callback_query(F.data == "decisions")
async def cb_decisions(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    await cb.message.answer(await _latest_decisions_text(5), reply_markup=_dashboard_kb())
    await cb.answer()


@router.callback_query(F.data == "top_pairs")
async def cb_top_pairs(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    await cb.message.answer(await _top_pairs_text(3), reply_markup=_dashboard_kb())
    await cb.answer()


@router.callback_query(F.data == "prune_ticks")
async def cb_prune_ticks(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    try:
        before = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        deleted = await prune_old_ticks()
        # shrink the file physically
        db = await get_db()
        try:
            await db.execute("VACUUM;")
            await db.commit()
        finally:
            await db.close()
        after = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        freed_kb = max(0.0, (before - after) / 1024.0)
        await cb.message.answer(
            "🧹 <b>Prune Complete</b>\n"
            f"• Ticks deleted: <code>{deleted:,}</code>\n"
            f"• Size before: <code>{before/1024:.1f} KB</code>\n"
            f"• Size after:  <code>{after/1024:.1f} KB</code>\n"
            f"• Freed:       <code>{freed_kb:.1f} KB</code>",
            reply_markup=_dashboard_kb(),
        )
    except Exception as e:
        log_error(e, "cb_prune_ticks")
        await cb.message.answer(f"❌ Prune failed: {e}", reply_markup=_dashboard_kb())
    await cb.answer()


@router.callback_query(F.data == "toggle_pause")
async def cb_toggle_pause(cb: CallbackQuery):
    global ENGINE_PAUSED
    if not _admin_only(cb):
        return
    ENGINE_PAUSED = not ENGINE_PAUSED
    state = "⏸️ PAUSED" if ENGINE_PAUSED else "▶️ RUNNING"
    await cb.message.edit_text(
        f"⚙️ <b>Engine {state}</b>\n"
        f"{'المحلل متوقف مؤقتًا. جمع البيانات ما زال يعمل.' if ENGINE_PAUSED else 'المحلل يعمل الآن بشكل طبيعي.'}",
        reply_markup=_dashboard_kb(),
    )
    await cb.answer(f"Engine {state}")


# --- settings --------------------------------------------------------------
@router.callback_query(F.data == "settings")
async def cb_settings(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    await cb.message.edit_text(
        "⚙️ <b>Live Settings</b>\n"
        "هذه القيم تُطبَّق فورًا على دورة التحليل التالية بدون Redeploy.\n\n"
        f"• Allow Min: <code>{RUNTIME['predictive_allow_min']:.0f}</code> "
        "<i>(الحد الأدنى لدرجة CAPE للسماح)</i>\n"
        f"• Gap σ:     <code>{RUNTIME['follower_gap_min_z']:.2f}</code> "
        "<i>(الحد الأدنى لتأخر التابع)</i>\n"
        f"• Max Lag:   <code>{RUNTIME['max_lag']}</code> "
        "<i>(أقصى إزاحة زمنية بالشموع)</i>",
        reply_markup=_settings_kb(),
    )
    await cb.answer()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@router.callback_query(F.data.startswith("set_"))
async def cb_set(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    action = cb.data[4:]
    if action == "reset":
        RUNTIME["predictive_allow_min"] = PREDICTIVE_ALLOW_MIN
        RUNTIME["follower_gap_min_z"]   = FOLLOWER_GAP_MIN_Z
        RUNTIME["max_lag"]              = MAX_LAG
    elif action == "allow_p5":
        RUNTIME["predictive_allow_min"] = _clamp(RUNTIME["predictive_allow_min"] + 5, 0, 100)
    elif action == "allow_m5":
        RUNTIME["predictive_allow_min"] = _clamp(RUNTIME["predictive_allow_min"] - 5, 0, 100)
    elif action == "gap_p":
        RUNTIME["follower_gap_min_z"] = round(_clamp(RUNTIME["follower_gap_min_z"] + 0.25, 0.0, 5.0), 2)
    elif action == "gap_m":
        RUNTIME["follower_gap_min_z"] = round(_clamp(RUNTIME["follower_gap_min_z"] - 0.25, 0.0, 5.0), 2)
    elif action == "lag_p":
        RUNTIME["max_lag"] = int(_clamp(RUNTIME["max_lag"] + 1, 1, 30))
    elif action == "lag_m":
        RUNTIME["max_lag"] = int(_clamp(RUNTIME["max_lag"] - 1, 1, 30))

    await cb.message.edit_text(
        "⚙️ <b>Live Settings</b>\n"
        f"• Allow Min: <code>{RUNTIME['predictive_allow_min']:.0f}</code>\n"
        f"• Gap σ:     <code>{RUNTIME['follower_gap_min_z']:.2f}</code>\n"
        f"• Max Lag:   <code>{RUNTIME['max_lag']}</code>",
        reply_markup=_settings_kb(),
    )
    await cb.answer("Updated")


# --- maintenance -----------------------------------------------------------
@router.callback_query(F.data == "dl_db")
async def cb_download_db(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    if not os.path.exists(DB_PATH):
        await cb.answer("DB not found", show_alert=True)
        return
    db = await get_db()
    try:
        await db.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        await db.commit()
    except Exception as e:
        log_error(e, "wal_checkpoint")
    finally:
        await db.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    await cb.message.answer_document(
        FSInputFile(DB_PATH, filename=f"cape_{stamp}.db"),
        caption=f"💾 DB snapshot @ {stamp} UTC",
    )
    await cb.answer()


@router.callback_query(F.data == "view_errors")
async def cb_view_errors(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    if not os.path.exists(ERROR_LOG_PATH) or os.path.getsize(ERROR_LOG_PATH) == 0:
        await cb.answer("No errors logged ✅", show_alert=True)
        return
    await cb.message.answer_document(
        FSInputFile(ERROR_LOG_PATH, filename=f"errors_{int(time.time())}.log"),
        caption="📜 errors.log",
    )
    await cb.answer()


@router.callback_query(F.data == "clear_errors")
async def cb_clear_errors(cb: CallbackQuery):
    if not _admin_only(cb):
        return
    try:
        open(ERROR_LOG_PATH, "w", encoding="utf-8").close()
    except Exception as e:
        log_error(e, "clear_errors")
    await cb.answer("Cleared ✅", show_alert=True)


# ---------- ATOMIC DB MIGRATION ----------
def _integrity_check(path: str) -> tuple[bool, str]:
    try:
        con = sqlite3.connect(path)
        try:
            cur = con.execute("PRAGMA integrity_check;")
            row = cur.fetchone()
            result = row[0] if row else "unknown"
            return (result == "ok", result)
        finally:
            con.close()
    except Exception as e:
        return (False, f"exception: {e}")


@router.message(F.document)
async def handle_db_upload(message: Message, bot: Bot):
    if not _admin_only(message):
        return
    doc = message.document
    if not doc.file_name or not doc.file_name.lower().endswith(".db"):
        return
    await message.reply("📥 Downloading incoming DB...")
    try:
        tg_file = await bot.get_file(doc.file_id)
        await bot.download_file(tg_file.file_path, TEMP_DB_PATH)
    except Exception as e:
        log_error(e, "db_upload_download")
        await message.reply(f"❌ Download failed: {e}")
        return

    ok, msg = await asyncio.to_thread(_integrity_check, TEMP_DB_PATH)
    if not ok:
        os.remove(TEMP_DB_PATH)
        await message.reply(f"❌ Integrity check FAILED: {msg}")
        return

    try:
        await ensure_tables_in_file(TEMP_DB_PATH)
    except Exception as e:
        log_error(e, "ensure_tables_in_file")
        await message.reply(f"❌ Schema migration failed: {e}")
        os.remove(TEMP_DB_PATH)
        return

    if os.path.exists(DB_PATH):
        try:
            shutil.copy2(DB_PATH, BACKUP_DB_PATH)
        except Exception as e:
            log_error(e, "db_backup")
            await message.reply(f"⚠️ Backup failed: {e}")

    try:
        os.replace(TEMP_DB_PATH, DB_PATH)
    except Exception as e:
        log_error(e, "db_atomic_replace")
        await message.reply(f"❌ Atomic replace failed: {e}")
        return

    await message.reply(
        "✅ DB migrated atomically.\n"
        f"Backup kept at: <code>{BACKUP_DB_PATH}</code>",
    )


async def telegram_bot_loop():
    if not BOT_TOKEN or BOT_TOKEN.startswith("PUT_"):
        logger.warning("[telegram] BOT_TOKEN not set — Telegram C&C disabled.")
        return
    if ADMIN_ID == 0:
        logger.warning("[telegram] ADMIN_ID not set — Telegram C&C disabled.")
        return

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except Exception as e:
        log_error(e, "telegram_bot_loop")
    finally:
        await bot.session.close()


# ---------------------------------------------------------------------------
# 11. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
async def main():
    await init_db()
    logger.info("=" * 60)
    logger.info("CAPE — Quant Workstation Edition starting")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"DB: {DB_PATH} | Errors: {ERROR_LOG_PATH}")
    logger.info(f"Mode: {MODE} | Admin: {ADMIN_ID} | Retention: {TICKS_RETENTION_HOURS}h")
    logger.info(f"Runtime: {RUNTIME}")
    logger.info("=" * 60)

    tasks = [
        asyncio.create_task(data_collector_loop(),   name="collector"),
        asyncio.create_task(flush_data_to_db(),      name="flusher"),
        asyncio.create_task(analyzer_loop(),         name="analyzer"),
        asyncio.create_task(signal_evaluator_loop(), name="evaluator"),
        asyncio.create_task(telegram_bot_loop(),     name="telegram"),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] shutdown requested.")
