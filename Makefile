# Makefile for PartIIProject WISE environment + Jupyter
# Usage:
#   make env         # create venv
#   make deps        # install requirements into venv
#   make kernel      # register Jupyter kernel
#   make imports     # run basic import sanity checks
#   make jupyter     # launch Jupyter Lab on port 8888
#   make all         # env + deps + kernel + imports
#   make clean       # remove venv

PYTHON      ?= python3.11
VENV_DIR    ?= test-env
KERNEL_NAME ?= wise-env
PROJECT_ROOT := $(shell pwd)

.PHONY: all env deps kernel imports jupyter clean

all: env deps kernel imports

# 1) Create virtual environment
env:
	$(PYTHON) -m venv $(VENV_DIR)

# 2) Install dependencies into the venv
deps: env
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip wheel && \
	pip install -r requirements.txt

# 3) Register Jupyter kernel for this env
kernel: env
	. $(VENV_DIR)/bin/activate && \
	export PYTHONPATH=$(PROJECT_ROOT)/src:$$PYTHONPATH && \
	python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "WISE (PartIIProject)"

# 4) Sanity checks: can we import the key bits?
imports: env
	. $(VENV_DIR)/bin/activate && \
	export PYTHONPATH=$(PROJECT_ROOT)/src:$$PYTHONPATH && \
	python -c "import numpy, pandas; print('ok numpy', numpy.__version__)" && \
	python -c "from pysat.solvers import Minisat22; print('pysat ok')" && \
	python -c "import src.simulator.qccd_circuit as qc; print('qccd import ok')"

# 5) Run Jupyter Lab on port 8888 using this env
jupyter: env
	. $(VENV_DIR)/bin/activate && \
	export PYTHONPATH=$(PROJECT_ROOT)/src:$$PYTHONPATH && \
	python -m jupyterlab --no-browser --port 8888

# 6) Nuke the environment
clean:
	rm -rf $(VENV_DIR)