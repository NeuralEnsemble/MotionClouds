default: doc

VERSION=`python3 -c'import MotionClouds; print(MotionClouds.__version__)'`

test:
	python3 test/test_color.py
	python3 test/test_grating.py
	python3 test/test_radial.py
	python3 test/test_speed.py

figures:
	python3 figures/fig_artwork_eschercube.py
	python3 figures/fig_contrast.py
	python3 figures/fig_wave.py

wiki:
	python3 wiki/fig_ApertureProblem.py
	python3 wiki/fig_MotionPlaid.py
	python3 wiki/fig_orientation.py

doc:
	@(cd doc && $(MAKE))
edit:
	open Makefile &
	spe &

# https://docs.python.org/3/distutils/packageindex.html
pypi_all: pypi_tags pypi_push pypi_upload
pypi_tags:
	git commit -am' tagging for PyPI '
	# in case you wish to delete tags, visit http://wptheming.com/2011/04/add-remove-github-tags/
	git tag $(VERSION) -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_push:
	python3 setup.py register

pypi_upload:
	python3 setup.py sdist #upload
	twine upload dist/*

pypi_docs:
	rm web.zip
	zip web.zip index.html
	open https://pypi.python.org/pypi?action=pkg_edit&name=$NAME

clean:
	rm -fr build dist results/* *.pyc **/*.pyc ./MotionClouds.egg-info ./MotionClouds/MotionClouds.egg-info
