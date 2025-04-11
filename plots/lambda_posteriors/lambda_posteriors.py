import os
import os.path as pa
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.graham23_tables as g23
from utils import inference
from utils.paths import PROJDIR
from utils.stats import calc_zero_cl, cl_around_mode

# Style file
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")
#
DF_FITPARAMS = pd.read_csv(f"{PROJDIR}/fit_lightcurves/fitparams.csv")

################################################################################


def plot_lambda_posterior(path, plot_kwargs={}, ax=None):
    # Load samples
    samples = np.loadtxt(pa.join(path, "O4_samples_graham23.dat"))
    samples_kde = np.concatenate([samples, -samples])
    # Gaussian kde
    kernel = gaussian_kde(samples_kde)
    x = np.linspace(0, 0.2, 1001)
    pdf = 2 * kernel(x) # "2 *" because the KDE is normalized over [-samples_max, samples_max]
    # Quantiles
    quants = cl_around_mode(x, pdf)
    peak = quants[0]
    lo = peak - quants[1]
    hi = quants[2] - peak
    quantstr = f"${peak:.3f}_{{- {lo:.3f}}}^{{+ {hi:.3f}}}$"
    # Plot
    plot_kwargs["label"] += f": $\lambda$={quantstr}"
    lines = ax.plot(x, pdf, rasterized=True, **plot_kwargs)
    ax.fill_between(
        x,
        0,
        pdf,
        where=(x >= quants[1]) & (x <= quants[2]),
        color=lines[0].get_color(),
        alpha=0.5,
        rasterized=True,
    )
    # Plot line for median
    ax.vlines(
        peak,
        0,
        pdf[np.digitize(peak, x) - 1],
        color=lines[0].get_color(),
        rasterized=True,
    )
    # Print quantiles
    for q in [0.1, 0.16, 0.5, 0.84, 0.9]:
        v = np.quantile(samples, q)
        print(f"Quantiles {q}: {v:6.3f}")
    print(f"Bayes factor [peak={peak:.3f}]/0: {(kernel(peak) / kernel(0))[0]}")
    # calc_zero_cl(x, pdf)


def plot_lambda_posterior_hist(path, plot_kwargs={}, ax=None):
    # Load samples
    samples = np.loadtxt(pa.join(path, "O4_samples_graham23.dat"))
    # Plot
    ax.hist(
        samples,
        bins=50,
        density=True,
        histtype="step",
        **plot_kwargs,
    )


def plot_lambda_posteriors(paths):
    # Initialize figure
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4, 3),
    )
    # Plot
    for path, label in zip(paths, [r"1.06 $\times$ 10$^{-8}$", r"4.79 $\times$ 10$^{-8}$"]):
        plot_lambda_posterior(
        # plot_lambda_posterior_hist(
            path,
            ax=ax,
            plot_kwargs={"label": label},
        )
    # Format
    ax.set_xlim(0, 0.2)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("PDF")
    ax.legend(
        title="Flares/AGN/day",
        loc="upper right",
        edgecolor="k",
    )
    # Save
    plt.tight_layout()
    plt.savefig(
        __file__.replace(".py", ".pdf"),
        # __file__.replace(".py", "_hist.pdf"),
        dpi=300,
    )
    plt.savefig(
        __file__.replace(".py", ".png"),
        # __file__.replace(".py", "_hist.png"),
        dpi=300,
    )
    plt.close()


################################################################################

# Get the directory path from the command line
if len(sys.argv) == 1:
    print("Usage: python gw_association_probabilities.py <path_to_directory>")
    print("Defaulting to array jobs 3, 7.")
    paths = [pa.join(PROJDIR, f"Posterior_sims_lambda_O4/array/{i}") for i in [3, 7]]
else:
    paths = sys.argv[1:]

# Plot the association probabilities
plot_lambda_posteriors(paths)
