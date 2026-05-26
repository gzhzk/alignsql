.PHONY: install test lint clean

# Install package in development mode
install:
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v

# Lint
lint:
	ruff check alignsql/
	black --check alignsql/

# Format
format:
	black alignsql/ tests/

# Run checks
check: test lint

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
