.PHONY: setup run-with run-without clean help

# Default target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  setup         Install dependencies and copy .env.example"
	@echo "  run-with      Run the agent with LangChain"
	@echo "  run-without   Run the agent without LangChain (manual)"
	@echo "  run-mcp       Run the agent with MCP tools (Airbnb/Flights)"
	@echo "  lint          Check for errors using ruff"
	@echo "  format        Format code using ruff"
	@echo "  clean         Remove cache files and virtual environment"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from .env.example"; \
	fi
	pip install -r requirements.txt
	@echo "✅ Setup complete. Please update .env with your OPENAI_API_KEY."

run-with:
	./venv/bin/python -m with_langchain.agent

run-without:
	./venv/bin/python without_langchain/agent.py

run-mcp:
	./venv/bin/python -m with_mcp.agent

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	@echo "✅ Cleaned up temporary files."
