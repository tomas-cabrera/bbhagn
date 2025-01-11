import os
import os.path as pa
import sys

import astropy.units as u
import matplotlib.pyplot as plt
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


def calc_arrs_for_directory(directory, force=False):
    # Check if cached
    cache_dir = pa.join(pa.dirname(__file__), ".cache", pa.basename(directory))
    s_arr_path = pa.join(cache_dir, "s_arrs.npy")
    b_arr_path = pa.join(cache_dir, "b_arrs.npy")
    n_flares_bgs_path = pa.join(cache_dir, "n_flares_bgs.npy")
    # Load if cached
    if pa.exists(s_arr_path) and pa.exists(b_arr_path) and not force:
        s_arrs = pd.read_csv(s_arr_path, index_col=0)
        b_arrs = pd.read_csv(b_arr_path, index_col=0)
        n_flares_bgs = pd.read_csv(n_flares_bgs_path, index_col=0)
    else:
        # Config
        config = yaml.safe_load(open(pa.join(path, "config.yaml")))
        if config["agn_distribution"]["model"] == "ConstantPhysicalDensity":
            config["agn_distribution"]["args"] = (
                config["agn_distribution"]["args"] * u.Mpc**-3
            )
        if "brightness_limits" in config["agn_distribution"]["density_kwargs"]:
            config["agn_distribution"]["density_kwargs"]["brightness_limits"] = [
                float(bl)
                for bl in config["agn_distribution"]["density_kwargs"][
                    "brightness_limits"
                ]
            ] * u.ABmag
        # Calculations
        lnprob_args = inference.setup(config, DF_FITPARAMS, nproc=config["nproc"])
        s_arrs, b_arrs, n_flares_bgs = inference.calc_arrs(
            config["H00"],
            config["Om0"],
            config["followup_prob"],
            *lnprob_args,
            config["agn_distribution"],
            config["z_min_b"],
            config["z_max_b"],
        )
        # Cast as pd.DataFrames
        gweventnames = g23.DF_GW["gweventname"].values
        gweventnames = np.array([gn.replace("*", "") for gn in gweventnames])
        flarenames = g23.DF_FLARE["flarename"].values
        s_arrs = pd.DataFrame(
            s_arrs,
            index=gweventnames,
            columns=flarenames,
        )
        b_arrs = pd.DataFrame(
            b_arrs,
            index=gweventnames,
            columns=flarenames,
        )
        n_flares_bgs = pd.DataFrame(
            n_flares_bgs,
            index=gweventnames,
            columns=flarenames,
        )
        # Cache
        os.makedirs(cache_dir, exist_ok=True)
        s_arrs.to_csv(s_arr_path)
        b_arrs.to_csv(b_arr_path)
        n_flares_bgs.to_csv(n_flares_bgs_path)
    return s_arrs, b_arrs, n_flares_bgs


def plot_association_pdf(directory, signal, background, ax=None):
    """
    Plot the association probabilities for the given directory.
    """
    # Load samples
    samples = np.loadtxt(pa.join(directory, "O4_samples_graham23.dat"))

    # Convert the samples to the association samples
    assoc_samples = samples * signal / (samples * signal + background)

    # Calculate kernel
    # kernel = gaussian_kde(assoc_samples)
    # x_kernel = np.linspace(0, 1, 1000)
    # y_kernel = kernel(x_kernel)

    # Calculate quantiles
    # TODO: calculate as horizontal slice containing sigma intervals

    # Plot
    savefig = False
    if ax is None:
        savefig = True
        fig, ax = plt.subplots()
    ax.hist(assoc_samples, bins=np.linspace(0, 1, 30), histtype="step", density=True)
    # ax.plot(x_kernel, y_kernel)
    if savefig:
        ax.set_xlabel("Association Probability")
        ax.set_ylabel("PDF")
        plt.savefig(pa.join(directory, "association_pdf.png"))


def plot_association_pdfs(directory, s_arr, b_arr):
    # Trim s_arr and b_arr to rows with associations
    assoc_rows = np.unique(np.where(s_arr > 0)[0])
    s_arr_assoc = s_arr.iloc[assoc_rows, :]
    b_arr_assoc = b_arr.iloc[assoc_rows, :]
    # Get GW and flare names
    gweventnames = g23.DF_GWBRIGHT.sort_values("dataset")["gweventname"].values
    flarenames = g23.DF_FLARE["flarename"].values
    # Initialize figure
    mosaic_arr = []
    for fn in flarenames:
        mosaic_row = []
        for gwn in gweventnames:
            mosaic_row.append(f"{fn}|{gwn}")
        mosaic_arr.append(mosaic_row)
    fig, axs = plt.subplot_mosaic(
        mosaic_arr,
        figsize=(7, 5),
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )
    # Plot; iterate over flares + gws
    for fi, fn in enumerate(flarenames):
        for gi, gn in enumerate(gweventnames):
            # Select axis
            ax = axs[f"{fn}|{gn}"]
            # Get the association probabilities
            plot_association_pdf(
                directory,
                s_arr_assoc.loc[gn, fn],
                b_arr_assoc.loc[gn, fn],
                ax=ax,
            )
            # If non-assoc, set background to gray
            if s_arr_assoc.loc[gn, fn] == 0:
                ax.set_facecolor("lightgray")
            # Formatting
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            # Label axes
            if gi == 0:
                ax.set_ylabel(fn, rotation=45, ha="right")
            if fi == len(flarenames) - 1:
                ax.set_xlabel(gn, rotation=45, ha="right")
    # Save
    plt.tight_layout()
    figpath = pa.join(
        pa.dirname(__file__),
        f"association_pdf_grid_{pa.basename(directory)}.png",
    )
    os.makedirs(pa.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)


################################################################################

# Get the directory path from the command line
try:
    path = sys.argv[1]
except IndexError:
    print("Usage: python gw_association_probabilities.py <path_to_directory>")
    raise

# Calc the association probabilities
s_arr, b_arr, n_flares_bgs = calc_arrs_for_directory(path, force=False)

# Plot the association probabilities
plot_association_pdfs(path, s_arr, b_arr)
