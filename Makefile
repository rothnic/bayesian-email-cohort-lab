.PHONY: test demo cold-start calibration baseline-smoke sensitivity controlled verify reproduce
PYTHON ?= python

test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	MPLCONFIGDIR=/tmp/cohort-lab-matplotlib $(PYTHON) run_bayesian_demo.py demo

cold-start:
	$(PYTHON) run_bayesian_demo.py cold-start

calibration:
	MPLCONFIGDIR=/tmp/cohort-lab-matplotlib $(PYTHON) run_bayesian_demo.py calibration

baseline-smoke:
	MPLCONFIGDIR=/tmp/cohort-lab-matplotlib $(PYTHON) run_pilot.py smoke

sensitivity:
	$(PYTHON) run_sensitivity.py

controlled:
	$(PYTHON) scripts/validate_conjugate_scenario.py

verify:
	$(PYTHON) scripts/validate_public_artifacts.py

reproduce: test demo cold-start calibration sensitivity controlled verify
