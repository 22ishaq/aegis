# Aegis: one-command workflows.  `make help` lists targets.
PY      ?= .venv/bin/python
BUILD   ?= build
export PYTHONPATH := $(CURDIR)/python

# Demo-only credentials for local runs. NEVER reuse these anywhere real:
# production keys come from a secret manager via the environment.
DEMO_ENV = AEGIS_ANALYST_KEY=demo-analyst-key-change-me AEGIS_VIEWER_KEY=demo-viewer-key-change-me

.PHONY: help setup build test test-cpp test-py sanitize bench train serve demo-injection demo-tamper lint clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install Python dependencies
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

build:  ## Build the C++ core, tests, benchmark and Python module
	cmake -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release -DAEGIS_WERROR=ON -DPython_EXECUTABLE="$(abspath $(PY))"
	cmake --build $(BUILD)

test: test-cpp test-py  ## Run every test

test-cpp: build  ## C++ unit tests
	ctest --test-dir $(BUILD) --output-on-failure

test-py: build  ## Python tests (crypto parity, model, agent, API)
	$(PY) -m pytest

sanitize:  ## C++ tests under AddressSanitizer+UBSan and ThreadSanitizer
	cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DAEGIS_PYTHON=OFF -DAEGIS_SANITIZE=address,undefined
	cmake --build build-asan && ./build-asan/aegis_tests
	cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DAEGIS_PYTHON=OFF -DAEGIS_SANITIZE=thread
	cmake --build build-tsan && ./build-tsan/aegis_tests

bench: build  ## Benchmark the engine (writes reports/bench.json)
	./$(BUILD)/aegis_bench --json reports/bench.json

train: build  ## Simulate data, train, evaluate, sign the model
	$(PY) -m aegis.train

serve: build  ## Run the API + live dashboard on http://127.0.0.1:8000
	$(DEMO_ENV) $(PY) -m aegis.service

demo-injection: build  ## Watch the investigator contain a prompt-injection attack
	$(PY) scripts/demo_injection.py

demo-tamper: build  ## Watch the audit chain catch an insider edit
	$(PY) scripts/demo_tamper.py

lint:  ## Static checks
	$(PY) -m ruff check python tests scripts

clean:  ## Remove build outputs
	rm -rf $(BUILD) build-asan build-tsan python/aegis/_core*.so
