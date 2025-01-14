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
from utils.stats import cl_around_mode

# Style file
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")
#
DF_FITPARAMS = pd.read_csv(f"{PROJDIR}/fit_lightcurves/fitparams.csv")

################################################################################


def plot_lambda_posterior(path, plot_kwargs={}, ax=None):
    # Load samples
    samples = np.loadtxt(pa.join(path, "O4_samples_graham23.dat"))
    # Gaussian kde
    kernel = gaussian_kde(samples)
    x = np.linspace(0, 0.3, 1001)
    pdf = kernel(x)
    # Quantiles
    quants = cl_around_mode(x, pdf)
    peak = quants[0]
    lo = peak - quants[1]
    hi = quants[2] - peak
    quantstr = f"${peak:.2f}_{{- {lo:.2f}}}^{{+ {hi:.2f}}}$"
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
