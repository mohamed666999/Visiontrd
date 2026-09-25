.PHONY: help install check test compile indicator report paper demo clean

help:
	@echo "FUGU-MAX — أوامر متاحة:"
	@echo "  make install    تثبيت المتطلبات"
	@echo "  make check      تجميع + اختبارات"
	@echo "  make test       اختبارات وحدة فقط"
	@echo "  make indicator  إشارة سريعة (BTCUSDT 15m)"
	@echo "  make report     تقرير بحثي على بيانات حقيقية"
	@echo "  make paper      تشغيل ورقى (لا أوامر)"
	@echo "  make demo       تشغيل على حساب الديمو"
	@echo "  make clean      حذف الملفات المؤقتة"

install:
	pip install -r requirements.txt

compile:
	python -m py_compile *.py tests/*.py

test:
	python -m pytest tests -q

check: compile test
	@echo "✅ كل الفحوص نجحت"

indicator:
	python fugu_indicator.py --symbol $${SYMBOL:-BTCUSDT} --interval $${INTERVAL:-15m}

report:
	python train_and_report.py --symbol $${SYMBOL:-BTCUSDT} --bars $${BARS:-8000}

paper:
	bash scripts/run_paper.sh

demo:
	bash scripts/run_demo.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
