default: test experiments doc
	
report_%.pdf: report_%.py MotionParticles.py
	    pyreport --double $< && open $@

test:
	python test/test_color.py
	python test/test_grating.py
	python test/test_radial.py
	python test/test_speed.py

experiments:
	# experiments/experiment
	python experiments/experiment_B_sf.py
	# python experiments/experiment_VSDI.py
	python experiments/experiment_competing.py
	# python experiments/experiment_concentric.py
	python experiments/experiment_smooth.py

figures:
	python figures/fig_artwork_eschercube.py
	python figures/fig_contrast.py
	python figures/fig_wave.py

wiki:
	python wiki/fig_ApertureProblem.py
	python wiki/fig_MotionPlaid.py
	python wiki/fig_orientation.py

doc:
	@(cd doc && $(MAKE))
edit:
	open Makefile &
	spe &

# https://docs.python.org/2/distutils/packageindex.html
pypi_tags:
	git tag 0.1.1 -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_push:
	python setup.py register

pypi_upload:
	python setup.py sdist bdist_wininst upload

pypi_docs: index.html
	zip web.zip index.html
	open http://pypi.python.org/pypi?action=pkg_edit&name=LogGabor

clean:
	touch *py
	rm -f results/* *.pyc


