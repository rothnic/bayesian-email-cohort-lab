PYTHON ?= python
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1

.PHONY: test evaluate figures reproduce
test:
	$(PYTHON) -m unittest discover -s dynamic_lab -t . -p 'test_*.py' -v
	cd prototype && $(PYTHON) -m unittest -v

evaluate:
	$(PYTHON) -m dynamic_lab.evaluate --phase all

figures:
	$(PYTHON) -m dynamic_lab.figures --artifacts dynamic_lab/artifacts --output dynamic_lab/artifacts/figures

reproduce: test evaluate figures
