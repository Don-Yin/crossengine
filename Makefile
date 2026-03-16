export PYTHONPATH := $(CURDIR)/benchmarks

run-01:
	python benchmarks/01-equal-weight/run.py

run-02:
	python benchmarks/02-stay-drift/run.py

run-03:
	python benchmarks/03-rotation/run.py

run-04:
	python benchmarks/04-rotation-with-cost/run.py

run-05:
	python benchmarks/05-sma-momentum/run.py

run-06:
	python benchmarks/06-inverse-vol/run.py

run-07:
	python benchmarks/07-cross-momentum/run.py

run-08:
	python benchmarks/08-ml-signal/run.py

run-09:
	python benchmarks/09-daily-binary-switch/run.py

run-10:
	python benchmarks/10-cash-starved/run.py

run-11:
	python benchmarks/11-concentrated-cascade/run.py

run-12:
	python benchmarks/12-daily-equal-weight/run.py

run-summary:
	python benchmarks/summary/run.py

run-all: run-01 run-02 run-03 run-04 run-05 run-06 run-07 run-08 run-09 run-10 run-11 run-12 run-summary
