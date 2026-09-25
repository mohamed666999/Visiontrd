#!/usr/bin/env bash
# تقرير بحثي كامل: مقارنة أنظمة + Walk-Forward + رسوم.
set -e
cd "$(dirname "$0")/.."
SYMBOL="${SYMBOL:-BTCUSDT}"
BARS="${BARS:-8000}"
exec python train_and_report.py --symbol "$SYMBOL" --bars "$BARS"
