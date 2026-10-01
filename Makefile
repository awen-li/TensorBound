PYTHON ?= python3

.PHONY: help verify reproduce-main test test-small

help:
	@echo "make reproduce-main  Verify data hashes and recompute main results"
	@echo "make test-small      Run the artifact integrity test"
	@echo "make test            Run all offline tests"

verify:
	$(PYTHON) scripts/verify_artifact.py

reproduce-main: verify

test-small: verify
	$(PYTHON) -m unittest discover -s tests -p 'test_artifact_package.py' -v

test:
	$(PYTHON) -m unittest discover -s tests -v
