PYTHON = python3.11
VENV = venv
REQ = requirements.txt
CONFIG = config.yaml
LOG = process_log.txt

.PHONY: all
all: setup run

.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r $(REQ)

.PHONY: run
run:
	$(VENV)/bin/python main.py --config $(CONFIG)

.PHONY: clean
clean:
	rm -f $(LOG)
	rm -rf __pycache__

.PHONY: reset
reset: clean
	rm -rf $(VENV)
