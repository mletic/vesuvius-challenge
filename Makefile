# Reproduce the benchmark and the write-up. Data come from the public Vesuvius Challenge bucket (CC BY-NC 4.0).
# Ink inference needs the model environment: run `make env` once, then export the variables it prints.
.PHONY: env labels benchmark report test

env:
	scripts/setup_inkenv.sh

labels:          ## label inventory, level-2 labels and the measured line period (no model needed)
	uv run python scripts/line_period.py

benchmark:       ## PREREG B1-B4: downloads, two-direction inference, all outcomes -> data/bench/results.json
	uv run python scripts/bench_b1.py
	uv run python scripts/bench_b2.py
	uv run python scripts/bench_eval.py

report:          ## write-up page -> report/index.html
	uv run python scripts/build_report.py

test:
	uv run python -m pytest -q tests
