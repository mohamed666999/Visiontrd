#!/usr/bin/env bash
# فحص سريع: تجميع + اختبارات وحدة
set -e
cd "$(dirname "$0")/.."
python -m py_compile *.py tests/*.py
python -m pytest tests -q
echo "✅ كل الفحوص نجحت"
