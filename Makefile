.PHONY : install test check build docs clean push_release

test:
	python setup.py build_ext --inplace
	rm -f .coverage
	nosetests --verbose --with-coverage --cover-package fbgbp tests/*

check:
	flake8 fbgbp && echo OK
	yapf -r --diff fbgbp && echo OK

build:
	python setup.py sdist

docs:
	sphinx-build -a docs docs/_build

clean:
	rm -rf fbgbp/*.cpp
	rm -rf fbgbp/*.so
	rm -rf build
	rm -rf dist
	rm -rf fbgbp.egg-info
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf .coverage

bump_patch:
	bumpversion patch

bump_minor:
	bumpversion minor

bump_major:
	bumpversion major

push_release:
	git push && git push --tags
