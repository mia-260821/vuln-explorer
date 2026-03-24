PYTHON ?= python3
PIP ?= pip3
ENV_FILE ?= .env.dev
COMPOSE ?= docker compose
UVICORN_APP ?= src.api.main:app
STREAMLIT_APP ?= src/ui/app.py
QUESTION ?= OpenSSL vulnerability analysis

.PHONY: help install test ingest api ui chat compose-build compose-up compose-down compose-logs compose-ingest pycompile

help:
	@echo "Available targets:"
	@echo "  make install        Install Python dependencies"
	@echo "  make test           Run the test suite"
	@echo "  make ingest         Run standalone NVD ingestion"
	@echo "  make api            Start the FastAPI server"
	@echo "  make ui             Start the Streamlit UI"
	@echo "  make chat           Run a sample CLI question"
	@echo "  make pycompile      Run a Python syntax check"
	@echo "  make compose-build  Build Docker services"
	@echo "  make compose-up     Start Docker Compose services"
	@echo "  make compose-ingest Run the dedicated ingestion container"
	@echo "  make compose-down   Stop Docker Compose services"
	@echo "  make compose-logs   Tail Docker Compose logs"

install:
	$(PIP) install -r requirements.txt

test:
	PYTHONPYCACHEPREFIX=$(CURDIR)/.pycache $(PYTHON) -m pytest -v

ingest:
	set -a; . $(ENV_FILE); set +a; $(PYTHON) main.py --ingest

api:
	set -a; . $(ENV_FILE); set +a; uvicorn $(UVICORN_APP) --host 0.0.0.0 --port 8000

ui:
	set -a; . $(ENV_FILE); set +a; streamlit run $(STREAMLIT_APP) --server.address=0.0.0.0 --server.port=8501

chat:
	set -a; . $(ENV_FILE); set +a; $(PYTHON) main.py --question "$(QUESTION)"

pycompile:
	PYTHONPYCACHEPREFIX=$(CURDIR)/.pycache $(PYTHON) -m py_compile main.py $$(find src -name '*.py' -type f | sort)

compose-build:
	$(COMPOSE) build

compose-up:
	$(COMPOSE) up --build

compose-ingest:
	$(COMPOSE) up --build ingestion

compose-down:
	$(COMPOSE) down

compose-logs:
	$(COMPOSE) logs -f
