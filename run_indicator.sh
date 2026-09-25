#!/usr/bin/env bash
# إشارة سريعة لرمز واحد بلا تشغيل دائم.
set -e
cd "$(dirname "$0")/.."
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-15m}"
exec python fugu_indicator.py --symbol "$SYMBOL" --interval "$INTERVAL"
