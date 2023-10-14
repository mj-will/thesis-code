"""
Code to calculate the JS divergence between samples.

Based on the code used in https://doi.org/10.5281/zenodo.8124198
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde


def calc_median_error(jsvalues, quantiles=(0.16, 0.84)):
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(jsvalues, quants_to_compute * 100)
    median = quants[1]
    plus = quants[2] - median
    minus = median - quants[0]
    return median, plus, minus


def calculate_js(samplesA, samplesB, ntests=100, xsteps=100, nsamples=1000, base=2):
    js_array = np.zeros(ntests)
    if nsamples is None:
        nsamples = min([len(samplesA), len(samplesB)])
    for j in range(ntests):
        A = np.random.choice(samplesA, size=nsamples, replace=False)
        B = np.random.choice(samplesB, size=nsamples, replace=False)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)

        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf, base=base), 2))

    return calc_median_error(js_array)
