CONFIG=config.mk
include $(CONFIG)

# ET_only_fig := figures/ET_only.pdf
# ET_CE_fig := figures/ET_CE.pdf

.PHONY : ET_ONLY_POST
ET_ONLY_POST : skymap_fits/ET_only/posterior_samples.hdf5

.PHONY : ET_CE_POST
ET_CE_POST : skymap_fits/ET_CE/posterior_samples.hdf5

# These were unused in the final thesis
# .PHONY : ET_only
# ET_only : $(ET_only_fig)

# .PHONY : ET_CE
# ET_CE : $(ET_CE_fig)

# $(ET_only_fig) :
# 	thesis-plot-bilby-corner -r $(ET_ONLY_RESULT) --filename ET_only

# $(ET_CE_fig) :
# 	thesis-plot-bilby-corner -r $(ET_CE_RESULT) --filename ET_CE


skymap_fits/ET_only/posterior_samples.hdf5 :
	python convert_result.py $(ET_ONLY_RESULT) skymap_fits/ET_only/ posterior_samples.hdf5

skymap_fits/ET_CE/posterior_samples.hdf5 :
	python convert_result.py $(ET_CE_RESULT) skymap_fits/ET_CE/ posterior_samples.hdf5

ET_ONLY_FIT : ET_ONLY_POST
	ligo-skymap-from-samples \
		--samples skymap_fits/ET_only/posterior_samples.hdf5 \
		--outdir skymap_fits/ET_only/ \
		--path /posterior_samples \
		--disable-distance-map \
		--jobs 4


ET_CE_FIT : ET_CE_POST
	ligo-skymap-from-samples \
		--samples skymap_fits/ET_CE/posterior_samples.hdf5 \
		--outdir skymap_fits/ET_CE/ \
		--path /posterior_samples \
		--disable-distance-map \
		--jobs 4
