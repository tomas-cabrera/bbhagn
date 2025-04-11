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
        config = yaml.safe_load(open(pa.join(directory, "config.yaml")))
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

    # Gaussian kde
    savefig = False
    if np.nanmin(assoc_samples) != np.nanmax(assoc_samples):
        assoc_samples_kde = np.concatenate([assoc_samples, -assoc_samples])
        kernel = gaussian_kde(assoc_samples_kde)
        try:
            x = np.linspace(0, 1, 1001)
            pdf = 2 * kernel(x)
            quants = cl_around_mode(x, pdf)
        except ValueError:
            x = np.linspace(0, 1, 10001)
            pdf = 2 * kernel(x)
            quants = cl_around_mode(x, pdf)
        lines = ax.plot(x, pdf, rasterized=True)
        # Quantiles
        peak = quants[0]
        lo = peak - quants[1]
        hi = quants[2] - peak
        quantstr = f"${peak:.2f}_{{- {lo:.2f}}}^{{+ {hi:.2f}}}$"
        ax.fill_between(
            x,
            0,
            pdf,
            where=(x >= quants[1]) & (x <= quants[2]),
            color=lines[0].get_color(),
            alpha=0.5,
            rasterized=True,
        )
        ax.text(
            0.95,
            1.05 - float(pa.basename(directory)) / 16,
            f"${peak:.2f}_{{- {lo:.2f}}}^{{+ {hi:.2f}}}$",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="none",
                edgecolor=lines[0].get_color(),
                pad=0.2,
            ),
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
    # # Plot
    # if ax is None:
    #     savefig = True
    #     fig, ax = plt.subplots()
    # n, bins, patches = ax.hist(
    #     assoc_samples,
    #     bins=np.linspace(0, 1, 30),
    #     histtype="step",
    #     density=True,
    #     rasterized=True,
    # )
    # # Quantiles
    # if signal != 0:
    #     quants = np.quantile(assoc_samples, [0.16, 0.5, 0.84])
    #     med = quants[1]
    #     lo = quants[1] - quants[0]
    #     hi = quants[2] - quants[1]

    #     # Shade hist between lo and hi
    #     def hist_at_x(x):
    #         inds = np.digitize(x, bins) - 1
    #         inds = np.clip(inds, 0, len(n) - 1)
    #         return n[inds]

    #     x = np.linspace(0, 1, 1000)
    #     ax.fill_between(
    #         x,
    #         0,
    #         hist_at_x(x),
    #         where=(x >= quants[0]) & (x <= quants[2]),
    #         color=patches[0].get_facecolor(),
    #         alpha=0.5,
    #         rasterized=True,
    #     )
    #     # Plot line for median
    #     ax.vlines(
    #         med,
    #         0,
    #         n[np.digitize(med, bins) - 1],
    #         color=patches[0].get_edgecolor(),
    #         rasterized=True,
    #     )

    if savefig:
        ax.set_xlabel("Association Probability")
        ax.set_ylabel("PDF")
        plt.savefig(pa.join(directory, "association_pdf.png"))


def initialize_mosaic_axes(
    gweventnames,
    flarenames,
    subplot_mosaic_kwargs={
        "figsize": (11, 7.5),
        "gridspec_kw": {"wspace": 0.0, "hspace": 0.0},
    },
):
    # Initialize figure
    mosaic_arr = []
    for fn in flarenames:
        mosaic_row = []
        for gwn in gweventnames:
            mosaic_row.append(f"{fn}|{gwn}")
        mosaic_arr.append(mosaic_row)
    # Initialize axes
    fig, axs = plt.subplot_mosaic(
        mosaic_arr,
        **subplot_mosaic_kwargs,
    )
    return fig, axs


def plot_association_pdf_grid(
    directory,
    gweventnames,
    flarenames,
    s_arr_assoc,
    b_arr_assoc,
    axs,
):
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
            # Formatting
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 7)
            # Define flags
            bottom = fi == len(flarenames) - 1
            top = fi == 0
            left = gi == 0
            right = gi == len(gweventnames) - 1
            assoc = s_arr_assoc.loc[gn, fn] != 0
            # General labels
            ax.set_xticks(np.arange(0, 1, 0.25))
            ax.set_yticks(np.arange(0, 8, 2))
            ax.tick_params(
                "both",
                direction="in",
                length=2,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            # Ticks
            if not assoc:
                ax.set_facecolor("lightgray")
                tps = ax.tick_params()
                ax.tick_params(
                    "both",
                    length=1,
                )
            # Label axes
            if top:
                ax.annotate(
                    gn,
                    xy=(0.5, 1.1),
                    xycoords="axes fraction",
                    ha="left",
                    # va="center",
                    rotation=45,
                    rasterized=True,
                )
            if bottom:
                ax.set_xlabel(r"$p^{\rm GW-AGN}_{ij}$")
                xtl = ax.get_xticklabels()
                xtl[0] = ""
                xtl[2] = ""
                ax.set_xticklabels(xtl)
            else:
                ax.set_xticklabels([])
            if right:
                ax.annotate(
                    fn,
                    xy=(1.1, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    # va="center",
                    rotation=45,
                    rasterized=True,
                )
            if left:
                ax.set_ylabel("PDF")
                ytl = ax.get_yticklabels()
                ytl[0] = ""
                ax.set_yticklabels(ytl)
            else:
                ax.set_yticklabels([])
    # # Save
    # plt.tight_layout()
    # figpath = pa.join(
    #     pa.dirname(__file__),
    #     f"association_pdf_grid_{pa.basename(directory)}.pdf",
    # )
    # os.makedirs(pa.dirname(figpath), exist_ok=True)
    # plt.savefig(
    #     figpath,
    #     dpi=300,
    # )
    # plt.savefig(
    #     figpath.replace(".pdf", ".png"),
    #     dpi=300,
    # )


def plot_association_pdfs(
    directories,
    gweventnames,
    flarenames,
    s_arrs,
    b_arrs,
    axs=None,
):
    # Initialize figure if needed
    if axs is None:
        fig, axs = initialize_mosaic_axes(gweventnames, flarenames)
    # Plot
    for d, s, b in zip(directories, s_arrs, b_arrs):
        # Trim s_arr and b_arr to rows with associations
        assoc_rows = np.unique(np.where(s > 0)[0])
        s_arr_assoc = s.iloc[assoc_rows, :]
        b_arr_assoc = b.iloc[assoc_rows, :]
        plot_association_pdf_grid(
            d,
            gweventnames,
            flarenames,
            s_arr_assoc,
            b_arr_assoc,
            axs,
        )
    # Add legend
    ax = axs[f"{flarenames[-1]}|{gweventnames[0]}"]
    legend_elements = [
        Patch(
            facecolor="none",
            edgecolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            label=r"$1.06 \times 10^{-8}$",
        ),
        Patch(
            facecolor="none",
            edgecolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
            label=r"$4.79 \times 10^{-8}$",
        ),
    ]
    ax.set_zorder(100)
    ax.legend(
        handles=legend_elements,
        title="Flares/AGN/day",
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        loc="lower left",
    )
    # Save
    plt.tight_layout()
    figpath = pa.join(
        pa.dirname(__file__),
        "association_pdf_grid.pdf",
    )
    os.makedirs(pa.dirname(figpath), exist_ok=True)
    plt.savefig(
        figpath,
        dpi=300,
    )
    plt.savefig(
        figpath.replace(".pdf", ".png"),
        dpi=300,
    )


################################################################################

# Get the directory path from the command line
if len(sys.argv) == 1:
    print("Usage: python gw_association_probabilities.py <path_to_directory>")
    print("Defaulting to array jobs 3 and 7.")
    paths = [pa.join(PROJDIR, f"Posterior_sims_lambda_O4/array/{i}") for i in [3, 7]]
else:
    paths = sys.argv[1:]

# Calc the association probabilities
s_arrs = []
b_arrs = []
n_flares_bgs = []
for p in paths:
    s_arr, b_arr, n_flares_bg = calc_arrs_for_directory(p, force=False)
    s_arrs.append(s_arr)
    b_arrs.append(b_arr)
    n_flares_bgs.append(n_flares_bg)

# Get GW and flare names
gweventnames = g23.DF_GWBRIGHT.sort_values("dataset")["gweventname"].values
flarenames = g23.DF_FLARE["flarename"].values
selected_flarenames = np.unique(g23.DF_ASSOC["flarename"].values)
selected_gws = np.unique(g23.DF_ASSOC["gweventname"].values)
gweventnames = np.array([gn for gn in gweventnames if gn in selected_gws])
flarenames = np.array([fn for fn in flarenames if fn in selected_flarenames])


# Plot the association probabilities
plot_association_pdfs(paths, gweventnames, flarenames, s_arrs, b_arrs)
