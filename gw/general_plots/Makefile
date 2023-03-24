BASE_CONFIG = ../../config.mk
include $(BASE_CONFIG)

waveform_plot := figures/waveform.pdf
ring_plot := figures/ring_plot.pdf
mass_space_plot := figures/mass_space.pdf
mass_likelihood_plot := figures/chirp_mass_vs_mass_ratio.pdf

.PHONY : all
all : waveform_plot ring_plot mass_space mass_likelihood_plot

.PHONY : waveform_plot
waveform_plot : $(waveform_plot)

$(waveform_plot) :
	$(EXE_JUPYTER) waveform_plot.ipynb
	rm waveform_plot.nbconvert.ipynb

.PHONY : ring_plot
ring_plot : $(ring_plot)

$(ring_plot) :
	$(EXE_JUPYTER) ring_plot.ipynb
	rm ring_plot.nbconvert.ipynb

.PHONY : mass_space_plot
mass_space_plot : $(mass_space_plot)

$(mass_space_plot) :
	$(EXE_JUPYTER) parameter_space_plots.ipynb
	rm parameter_space_plots.nbconvert.ipynb

.PHONY : mass_likelihood_plot
mass_likelihood_plot : $(mass_likelihood_plot)

$(mass_likelihood_plot) :
	$(EXE_JUPYTER) parameter_space_plots.ipynb
	rm parameter_space_plots.nbconvert.ipynb

clean :
	rm *.nbconvert.ipynb
