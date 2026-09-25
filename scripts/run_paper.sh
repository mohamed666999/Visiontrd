#!/usr/bin/env bash
# تشغيل ورقى بالكامل: يقرأ السوق من بينانس ولا يرسل أي أمر إطلاقًا.
set -e
cd "$(dirname "$0")/.."
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT}"
INTERVAL="${INTERVAL:-15m}"
exec python paper_engine.py --symbols "$SYMBOLS" --interval "$INTERVAL"
