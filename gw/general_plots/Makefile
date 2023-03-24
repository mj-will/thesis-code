
EXE_JUPYTER = jupyter nbconvert --to notebook --execute

waveform_plot := figures/waveform.pdf
ring_plot := figures/ring_plot.pdf

.PHONY : all
all : waveform_plot ring_plot

.PHONY : waveform_plot
waveform_plot :
	$(EXE_JUPYTER) waveform_plot.ipynb
	rm $@.nbconvert.ipynb

.PHONY : ring_plot
ring_plot :
	$(EXE_JUPYTER) ring_plot.ipynb
	rm $@.nbconvert.ipynb

clean :
	rm *.nbconvert.ipynb
