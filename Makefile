.PHONY: default lint pylint mypy black pytest clean

default: lint

lint:
	ruff qmlant

pylint:
	pylint -rn qmlant

mypy:
	mypy qmlant

black:
	black qmlant setup.py

pytest:
	pytest tests
