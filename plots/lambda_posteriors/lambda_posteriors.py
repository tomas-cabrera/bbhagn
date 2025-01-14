import os
import os.path as pa
import sys

import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.graham23_tables as g23
from utils import inference
from utils.paths import PROJDIR

# Style file
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")
#
DF_FITPARAMS = pd.read_csv(f"{PROJDIR}/fit_lightcurves/fitparams.csv")

################################################################################


def plot_lambda_posterior(path, plot_kwargs={}, ax=None):
    # Load samples
    samples = np.loadtxt(pa.join(path, "O4_samples_graham23.dat"))
    # Quantiles
    quants = np.quantile(samples, [0.16, 0.5, 0.84])
    med = quants[1]
    lo = quants[1] - quants[0]
    hi = quants[2] - quants[1]
    quantstr = f"${med:.2f}_{{- {lo:.2f}}}^{{+ {hi:.2f}}}$"
    # ax.text(
    #     0.5,
    #     0.5 - float(pa.basename(path)) / 16,
    #     quantstr,
    #     ha="center",
    #     va="center",
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     bbox=dict(
    #         facecolor=patches[0].get_facecolor(),
    #         edgecolor=patches[0].get_edgecolor(),
    #         pad=0.2,
    #     ),
    #     rasterized=True,
    # )
    # Plot
    plot_kwargs["label"] += f": $\lambda$={quantstr}"
    n, bins, patches = ax.hist(
        samples,
        bins=np.linspace(0, 0.3, 30),
        histtype="step",
        density=True,
        lw=2,
        rasterized=True,
        **plot_kwargs,
    )

    # Shade hist between lo and hi
    def hist_at_x(x):
        inds = np.digitize(x, bins) - 1
        inds = np.clip(inds, 0, len(n) - 1)
        return n[inds]

    x = np.linspace(0, 1, 1000)
    ax.fill_between(
        x,
        0,
        hist_at_x(x),
        where=(x >= quants[0]) & (x <= quants[2]),
        color=patches[0].get_facecolor(),
        alpha=0.5,
        rasterized=True,
    )
    # Plot line for median
    ax.vlines(
        med,
        0,
        n[np.digitize(med, bins) - 1],
        color=patches[0].get_edgecolor(),
        rasterized=True,
    )


def plot_lambda_posteriors(paths):
    # Initialize figure
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4, 3),
    )
    # Plot
    for path, label in zip(paths, ["Volumetric", "QLF"]):
        plot_lambda_posterior(
            path,
            ax=ax,
            plot_kwargs={"label": label},
        )
    # Format
    ax.set_xlim(0, 0.3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("PDF")
    ax.legend(
        title="AGN Distribution",
        loc="upper right",
        edgecolor="k",
    )
    # Save
    plt.tight_layout()
    plt.savefig(
        __file__.replace(".py", ".pdf"),
        dpi=300,
    )
    plt.savefig(
        __file__.replace(".py", ".png"),
        dpi=300,
    )


################################################################################

# Get the directory path from the command line
if len(sys.argv) == 1:
    print("Usage: python gw_association_probabilities.py <path_to_directory>")
    print("Defaulting to array jobs 1 and 3.")
    paths = [pa.join(PROJDIR, f"Posterior_sims_lambda_O4/array/{i}") for i in [1, 3]]
else:
    paths = sys.argv[1:]

# Plot the association probabilities
plot_lambda_posteriors(paths)
