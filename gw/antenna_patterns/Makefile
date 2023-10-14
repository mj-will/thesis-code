plus_static := figures/antenna_plus.pdf
cross_static := figures/antenna_cross.pdf

.PHONY : static_plots
static_plots : $(plus_static) $(cross_static)

$(plus_static) :
	python plot_static.py

$(cross_static) :
	python plot_static.py

.PHONY : html
html:
	python plot_static.py --html

.PHONY : upload
upload :
	python plot_static.py --n-points=50 --html --upload
