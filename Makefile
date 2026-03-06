.PHONY: help install run dev clean clean-cache

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

run: ## Run the app
	uv run spoticast

dev: ## Run the app with auto-reload
	uv run uvicorn spoticast.server:app --host 127.0.0.1 --port $${PORT:-8765} --reload

clean-cache: ## Clear generated audio, scripts, and research cache (forces regeneration)
	rm -rf generated/ .cache/ .research_cache/
	@echo "Cleared generated audio, script cache, and research cache."

clean: ## Remove caches and build artifacts
	rm -rf .venv __pycache__ spoticast/__pycache__ spoticast/**/__pycache__ dist build *.egg-info
