#!/usr/bin/env bash
# تشغيل على حساب بينانس الديمو (أموال وهمية) — يتطلب .env مملوءًا بمفاتيح الديمو.
set -e
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  echo "❌ لا يوجد ملف .env — انسخه من .env.example واملأ مفاتيح الديمو."
  exit 1
fi
python -m pytest tests -q
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT}"
INTERVAL="${INTERVAL:-15m}"
exec python paper_engine.py --symbols "$SYMBOLS" --interval "$INTERVAL" --demo-account
