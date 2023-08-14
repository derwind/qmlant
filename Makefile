.PHONY: default lint mypy black pytest clean

default: lint

lint:
	pylint -rn qmlant

mypy:
	mypy qmlant

black:
	black qmlant setup.py

pytest:
	pytest tests
