# Variables
VENV = venv
TEST_DIR = test
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip3
COVERAGE = $(VENV)/bin/coverage
FLASK = $(VENV)/bin/flask
PORT = 8080

# By default, run help
default: help

.PHONY: init_hooks test

# npm install dependencies if package.json or package-lock.json changes
client/node_modules: client/package.json client/package-lock.json
	cd client && npm install && cd ..

# npm install dependencies if package.json or package-lock.json changes
node_modules: package.json package-lock.json
	npm install

# Init the virtual environment and install requirements
$(VENV)/bin/activate: requirements.txt requirements-dev.txt
	python3.11 -m venv $(VENV) || python3.10 -m venv $(VENV) || python3.9 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt

# Activate the virtual environment
venv: $(VENV)/bin/activate node_modules client/node_modules

# init git pre-commit hooks
$(VENV)/bin/pre-commit: venv .pre-commit-config.yaml
	$(PIP) install -U pre_commit
	$(VENV)/bin/pre-commit install

init_hooks: venv $(VENV)/bin/pre-commit

# build svelte front end
build:
	cd client && npm install && npm run build && cd ..

# Run app
run: build venv $(VENV)/bin/pre-commit
	PYTHONPATH=. $(PYTHON) -m uvicorn app:app --reload --port $(PORT)

# Run tests
test: venv $(VENV)/bin/pre-commit
	FIN_GENIE_USERNAME=ci FIN_GENIE_PASSWORD=cd PYTHONPATH=. $(PYTHON) -m pytest

# Measure test coverage and generate HTML report
coverage: venv
	FIN_GENIE_USERNAME=ci FIN_GENIE_PASSWORD=cd PYTHONPATH=. $(COVERAGE) run --branch -m pytest $(TEST_DIR)
	$(COVERAGE) report

# Measure test coverage and generate XML report
coverage_xml: coverage
	$(COVERAGE) xml

# Run HTTP server to expose HTML report
run_coverage: coverage
	$(COVERAGE) html
	$(PYTHON) -m http.server -d htmlcov

# Lint the code
lint: venv $(VENV)/bin/pre-commit
	PYTHONPATH=. $(PYTHON) -m pylint --ignore-paths llama2,venv,.venv **/*.py
	PYTHONPATH=. $(PYTHON) -m mypy . --exclude llama2
	PYTHONPATH=. $(PYTHON) -m flake8 --extend-exclude llama2,venv,.venv .
	PYTHONPATH=. $(PYTHON) -m isort --check-only --profile black .
	PYTHONPATH=. $(PYTHON) -m black --check .
	npx eslint . --ignore-pattern venv --ignore-pattern htmlcov --ignore-pattern client/dist
	npx standard

# Reformats the code
format: venv $(VENV)/bin/pre-commit
	$(PYTHON) -m isort . --profile black
	$(PYTHON) -m black .
	npx standard --fix

# Clean up Python artifacts
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf client/dist/*

# Clean up Python artifacts and virtual environment
real_clean: clean
	rm -rf $(VENV)
	rm -rf node_modules
	rm -rf client/node_modules

# Show help
help:
	@echo "make venv - Init the virtual environment"
	@echo "make init_hooks - Init git pre-commit hooks"
	@echo "make lint - Lint the code"
	@echo "make format - Reformats the code"
	@echo "make build - Build svelte front end"
	@echo "make run - Run flask"
	@echo "make test - Run tests"
	@echo "make coverage - Measure test coverage and generate HTML report"
	@echo "make coverage_xml - Measure test coverage and generate XML report"
	@echo "make run_coverage - Run HTTP server to expose HTML report"
	@echo "make clean - Clean up Python artifacts"
	@echo "make real_clean - Clean up Python artifacts and virtual environment"
