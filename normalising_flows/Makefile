BASE_CONFIG = ../config.mk
include $(BASE_CONFIG)

multimodal := figures/multimodal_comparison.pdf
lu := figures/lu_factorization.pdf
train_w_weights := figures/flow_weights_plot.pdf

.PHONY : all
all : multimodal lu train_w_weights

.PHONY : multimodal
multimodal : $(multimodal)

$(multimodal) :
	python multimodal_target.py 0

.PHONY : lu
lu : $(lu)

$(lu):
	$(EXE_JUPYTER) lu_factorization.ipynb
	rm lu_factorization.nbconvert.ipynb

.PHONY : train_w_weights
train_w_weights : $(train_w_weights)

$(train_w_weights) :
	$(EXE_JUPYTER) training_w_weights.ipynb
	rm training_w_weights.nbconvert.ipynb

clean :
	rm *.nbconvert.ipynb
