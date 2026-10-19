# Reproduce the benchmark and the write-up. Data come from the public Vesuvius Challenge bucket (CC BY-NC 4.0).
# Ink inference needs the model environment: run `make env` once, then export the variables it prints.
.PHONY: env labels benchmark report test

env:
	scripts/setup_inkenv.sh

labels:          ## label inventory, level-2 labels and the measured line period (no model needed)
	uv run python scripts/benchmark/bench_inventory.py
	uv run python scripts/benchmark/line_period.py

benchmark: labels ## PREREG B1-B4: downloads, two-direction inference, all outcomes -> data/bench/results.json
	uv run python scripts/benchmark/bench_b1.py
	uv run python scripts/benchmark/bench_b2.py
	uv run python scripts/benchmark/bench_eval.py

report:          ## write-up page -> report/index.html and docs/index.html (GitHub Pages)
	uv run python scripts/writeup/build_report.py

test:
	uv run python -m pytest -q tests
