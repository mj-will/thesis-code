"""Plot of Amdahfs law"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from thesis_utils.plotting import save_figure, set_plotting

from scaling_plot import amdahls_law


def main():

    set_plotting()

    fraction = dict(
        bbh=0.5,
        bns=0.95,
        bns_roq=0.6,
    )
    labels = dict(
        bbh=r"BBH ($4\,\textrm{s}$)",
        bns=r"BNS ($80\,\textrm{s}$)",
        bns_roq=r"BNS w ROQ ($80\,\textrm{s}$)",
    )

    fig = plt.figure()

    optimal = np.arange(1, 33, 1)
    plt.plot(optimal, optimal, color="k", label="Optimal")
    markers = ["o", "+", "^", "x", "s", "*"]
    ls = ["--", "-.", ":"]
    colours = sns.color_palette(n_colors=len(fraction))

    for i, (k, f) in enumerate(fraction.items()):
        plt.plot(
            optimal,
            amdahls_law(f, optimal),
            ls=ls[i],
            color=colours[i],
            label=labels.get(k, k),
        )

    plt.legend()
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    save_figure(fig, f"scaling_gw", "figures")


if __name__ == "__main__":
    main()
