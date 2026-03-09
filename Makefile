.PHONY: setup lint test smoke build-index fetch

setup:
	pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest -q tests/

smoke:
	python scripts/smoke.py

fetch:
	python scripts/fetch_movies.py

build-index:
	python scripts/fetch_movies.py
	python scripts/build_index.py
