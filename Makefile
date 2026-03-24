.PHONY: install lint format typecheck test test-unit test-integration test-smoke build-llama server help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python deps with uv
	uv sync --dev

lint: ## Run ruff linter
	uv run ruff check . --fix

format: ## Run ruff formatter
	uv run ruff format .

typecheck: ## Run mypy type checker
	uv run mypy src/

test: ## Run all tests (unit + integration)
	uv run pytest tests/unit/ tests/integration/ -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests (needs llama-server)
	uv run pytest tests/integration/ -v --tb=short

test-smoke: ## Run smoke tests (needs model + server)
	uv run pytest tests/smoke/ -v --tb=short

test-cov: ## Run tests with coverage
	uv run pytest tests/unit/ -v --cov=llama_video --cov-report=term-missing --cov-report=html

check: lint typecheck test-unit ## Run all checks (lint + typecheck + unit tests)

build-llama: ## Build patched llama.cpp
	./scripts/build.sh

setup: ## Full setup: install deps + clone/patch/build llama.cpp
	./scripts/setup.sh

server: ## Start the Python API service
	uv run llama-video-server

debug-extract: ## Extract frames from a test video (usage: make debug-extract VIDEO=path/to/video.mp4)
	uv run llama-video-debug extract $(VIDEO) --fps 2 --output-dir /tmp/llama-video-debug/

debug-preprocess: ## Preprocess a test video (usage: make debug-preprocess VIDEO=path/to/video.mp4)
	uv run llama-video-debug preprocess $(VIDEO) --fps 2
