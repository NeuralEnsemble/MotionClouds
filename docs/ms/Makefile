default: pdf
#default: run pdf
#default: pdf diff figures_png
################################################
figures:
	python3 fig_Sanz12JNP.py
	cd ../files/; echo "Creating figure 1"; montage figure1A.png figure1B.png figure1C.png morphing-B_angle-0_1.png morphing-B_angle-1_0.png morphing-B_angle-10_0.png -geometry 564x512 figure1.pdf
	cd ../files/; echo "Creating figure 2"; montage figure2A.png figure2B.png figure2C.png -geometry 564x512 figure2.pdf
	cd ../files/; echo "Creating figure 3"; montage figure3A.png figure3B.png figure3C.png grating.png aperture.png grating-rdk.png -geometry 564x512 figure3.pdf
	cd ../files/; echo "Creating figure 4"; montage figure4A.png figure4B.png figure4C.png grating-B_sf-0_05.png grating-B_sf-0_15.png grating-B_sf-0_4.png -geometry 564x512 figure4.pdf
	cd ../files/; echo "Creating figure 5"; montage figure5A.png figure5B.png figure5C.png grating-noise.png plaid.png counterphase_grating.png -geometry 564x512 figure5.pdf

################################################
# macros
3D: # TODO embbed to doc
	asy  --keep -tex pdflatex figureMotionClouds.asy

%.jpg: %.pdf
	convert $< $@
%.png: %.pdf
	convert $< $@

################################################
# LaTeX stuff
SRC = MotionClouds

edit_linux: linux_edit
linux_edit:
	texmaker $(SRC).tex &
	gedit Makefile &

edit_mac: mac_edit
mac_edit:
	open $(SRC).tex
	open -e Makefile

#################################################
pdf: $(SRC).pdf $(SRC)_VideoFigures.pdf $(SRC)_Supplementary.pdf
LATEXMK = latexmk -pdf

figures_png: figure1.png figure2.png figure3.png figure4.png figure5.png

$(SRC).pdf: $(SRC).tex figures $(figures_png)
	#wget "http://www.citeulike.org/bibtex/user/LaurentPerrinet/tag/sanz12jnp?fieldmap=posted-at:date-added&do_username_prefix=0&key_type=4" -O MotionClouds.bib
	$(LATEXMK) $(SRC)

$(SRC)_Supplementary.pdf: $(SRC)_Supplementary.tex
	$(LATEXMK) $(SRC)_Supplementary

$(SRC)_VideoFigures.pdf: $(SRC)_VideoFigures.tex
	@(cd .. && $(MAKE) test)
	$(LATEXMK) $(SRC)_VideoFigures
################################################
# post-production
diff: $(SRC).tex
	wget https://raw.github.com/meduz/MotionClouds/928ae138400ea3d556b1d10938ac35a69209af0f/doc/MotionClouds.tex -O oldMotionClouds.tex
	latexdiff oldMotionClouds.tex  $(SRC).tex > diff.tex
	# latexdiff /Users/lup/Desktop/RTC/12-01-31_MotionClouds\ resubmission/main_MotionClouds.tex /Users/lup/Desktop/12-03-07_MotionClouds\ resubmission/MotionClouds.tex   > diff.tex
	latexmk -pdf diff.tex
	rm oldMotionClouds.tex

################################################
# cleaning macros

touch:
	touch *.tex *py
clean: # removing all temporary files from LaTeX
	rm -f *.bcf *idx  *.md5 *.out  *.auxlock  *.run.xml *.dvi *.ps *.out *.log *.aux *.bbl *.blg *.snm *.fls *.nav *.toc *.fff *.synctex.gz* *.fdb_latexmk
	rm -fr *.fls *.fff *.lof *.dvi *.ps  *.out *.log *.aux *.bbl *.blg *.snm *.nav *.toc *.synctex.gz *.fdb_latexmk
clean_hard: clean_soft # careful! this is quite radical!
	rm -fr figure* diff* MotionClouds.bib  *.pdf  results

.PHONY: all touch clean clean_hard diff pdf 3D figures
