default: doc

VERSION=`python3 -c'import MotionClouds; print(MotionClouds.__version__)'`
PYTHON = python3

test:
	$(PYTHON)  test/test_color.py
	$(PYTHON)  test/test_grating.py
	$(PYTHON) test/test_radial.py
	$(PYTHON) test/test_speed.py

figures:
	$(PYTHON) figures/fig_artwork_eschercube.py
	$(PYTHON) figures/fig_contrast.py
	$(PYTHON) figures/fig_wave.py

wiki:
	$(PYTHON) wiki/fig_ApertureProblem.py
	$(PYTHON) wiki/fig_MotionPlaid.py
	$(PYTHON) wiki/fig_orientation.py

doc:
	@(cd doc && $(MAKE))
edit:
	open Makefile &
	spe &

# https://docs.python.org/3/distutils/packageindex.html
pypi_all: pypi_tags pypi_upload
pypi_tags:
	git commit -am' tagging for PyPI'
	# in case you wish to delete tags, visit http://wptheming.com/2011/04/add-remove-github-tags/
	git tag $(VERSION) -m "New release."
	git push --tags origin master

pypi_upload:
	$(PYTHON) setup.py sdist #upload
	twine upload dist/*

pypi_docs:
	rm web.zip
	zip web.zip index.html
	open https://pypi.python.org/pypi?action=pkg_edit&name=$NAME

clean:
	rm -fr build dist results/* *.pyc **/*.pyc ./MotionClouds.egg-info ./MotionClouds/MotionClouds.egg-info
