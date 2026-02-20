## NBe:
# The GW190521-J124942.30+344928.9 and GW190803_022701-J120437.98+500024.0 association posteriors are similar because s_arr/b_arr is similar for the two distributions, and the lambda samples are identical

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
LAMBDA_UPPERLIMIT = 0.2

################################################################################


def calc_arrs_for_directory(directory, force=False):
    # Check if cached
    # cache_dir = pa.join(pa.dirname(__file__), ".cache", pa.basename(directory))
    # This is for displaying jobs 11 and 18; should have identical s/b_arrs, but the lambda samples will change
    cache_dir = pa.join(pa.dirname(__file__), ".cache", pa.basename(directory))
    s_arr_path = pa.join(cache_dir, "s_arrs.npy")
    b_arr_path = pa.join(cache_dir, "b_arrs.npy")
    n_flares_bgs_path = pa.join(cache_dir, "n_flares_bgs.npy")
    # Load if cached
    if pa.exists(s_arr_path) and pa.exists(b_arr_path) and not force:
        s_arrs = pd.read_csv(s_arr_path, index_col=0)
        b_arrs = pd.read_csv(b_arr_path, index_col=0)
        n_agns = pd.read_csv(n_flares_bgs_path, index_col=0)
    else:
        # Config
        config_path = pa.join(directory, "config.yaml")
        config = yaml.safe_load(open(config_path))
        config["config_file"] = config_path
        # Parse AGN distribution config
        for k, v in config["agn_distribution"].items():
            if v["model"] == "ConstantPhysicalDensity":
                v["args"] = v["args"] * u.Mpc**-3
            if "brightness_limits" in v["density_kwargs"]:
                if "brightness_units" not in v["density_kwargs"]:
                    raise ValueError(
                        "Must specify brightness_units if brightness_limits is given."
                    )
                if v["density_kwargs"]["brightness_units"] == "ABmag":
                    bu = u.ABmag
                elif v["density_kwargs"]["brightness_units"] == "erg/s":
                    bu = u.erg / u.s
                else:
                    raise ValueError("brightness_units must be 'ABmag' or 'erg/s'.")
                v["density_kwargs"]["brightness_limits"] = [
                    float(bl) for bl in v["density_kwargs"]["brightness_limits"]
                ] * bu
                v["density_kwargs"].pop("brightness_units")
        # Calculations
        lnprob_args = inference.setup(config, nproc=config["nproc"])
        s_arrs, b_arrs, n_agns = inference.calc_arrs(
            config["H00"],
            config["Om0"],
            *lnprob_args[:-1],
            config["agn_distribution"]["astrophysical"],
            config["z_min_b"],
            config["z_max_b"],
        )
        # Cast as pd.DataFrames
        gweventnames = pd.read_csv(config["gw_csv"])["gweventname"].values
        gweventnames = np.array([gn.strip("*") for gn in gweventnames])
        flarenames = pd.read_csv(config["flare_csv"])["flarename"].values
        s_arrs = pd.DataFrame(
            s_arrs,
            index=gweventnames,
            columns=flarenames,
        )
        print(f"lnprob_args[-1] = {lnprob_args[-1]}")
        # NOTE: b_arr here includes the conversion to flare number density, unlike inference; the average flare rate is assumed
        b_arrs = pd.DataFrame(
            b_arrs * lnprob_args[-1],
            index=gweventnames,
            columns=flarenames,
        )
        n_agns = pd.DataFrame(
            n_agns,
            index=gweventnames,
            columns=flarenames,
        )
        # Cache
        os.makedirs(cache_dir, exist_ok=True)
        s_arrs.to_csv(s_arr_path)
        b_arrs.to_csv(b_arr_path)
        n_agns.to_csv(n_flares_bgs_path)
    return s_arrs, b_arrs, n_agns


def plot_association_pdf(directory, signal, signals, background, ax=None):
    """
    Plot the association probabilities for the given directory.
    """
    # Load samples
    samples = np.loadtxt(pa.join(directory, "O4_samples_graham23.dat"))

    # Convert the samples to the association samples
    assoc_samples = samples * signal / (samples * np.sum(signals) + background)
    assoc_colocation = LAMBDA_UPPERLIMIT * signal / (LAMBDA_UPPERLIMIT * np.sum(signals) + background)

    # Gaussian kde
    savefig = False
    if np.nanmin(assoc_samples) != np.nanmax(assoc_samples):
        assoc_samples_kde = np.concatenate(
            [assoc_samples, -assoc_samples, 2 - assoc_samples]
        )
        kernel = gaussian_kde(assoc_samples_kde, bw_method=0.01)
        try:
            x = np.linspace(0, 1, 1001)
            pdf = 3 * kernel(x)
            quants = cl_around_mode(x, pdf)
        except ValueError:
            x = np.linspace(0, 1, 10001)
            pdf = 3 * kernel(x)
            quants = cl_around_mode(x, pdf)
        lines = ax.plot(x, pdf, rasterized=True)
        # Quantiles
        peak = quants[0]
        lo = peak - quants[1]
        hi = quants[2] - peak
        if quants[1] < 0.01:
            quantstr = f"$p < {hi:.2f}$"
        else:
            quantstr = f"${peak:.2f}_{{- {lo:.2f}}}^{{+ {hi:.2f}}}$"
        ax.fill_between(
            x,
            0,
            pdf,
            where=(x >= quants[1]) & (x <= quants[2]),
            color=lines[0].get_color(),
            alpha=0.5,
            lw=0,
            rasterized=True,
        )
        ax.text(
            0.95,
            # 3.15 - float(pa.basename(directory)) / 4, # for jobs 9, 10
            0.95,
            quantstr,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="none",
                edgecolor=lines[0].get_color(),
                lw=1,
                pad=1,
            ),
            rasterized=True,
        )
        ax.text(
            0.95,
            # 3.15 - float(pa.basename(directory)) / 4, # for jobs 9, 10
            0.74,
            f"$p_{{\lambda = {LAMBDA_UPPERLIMIT}}} = {assoc_colocation:.2f}$",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="none",
                edgecolor=lines[0].get_color(),
                ls="--",
                lw=1,
                pad=1,
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
        # Plot line for colocation (prob GW occured at location of flare, sans lambda posterior)
        ax.vlines(
            assoc_colocation,
            0,
            3,
            color=lines[0].get_color(),
            ls="--",
            rasterized=True,
        )

    if savefig:
        ax.set_xlabel("Association Probability")
        ax.set_ylabel("PDF")
        plt.savefig(pa.join(directory, "association_pdf.png"))


def plot_background_pdf(directory, signals, background, ax=None):
    """
    Plot the association probabilities for the given directory.
    """
    # Load samples
    samples = np.loadtxt(pa.join(directory, "O4_samples_graham23.dat"))

    # Convert the samples to the association samples
    assoc_samples = background / (samples * np.sum(signals) + background)
    assoc_colocation = background / (LAMBDA_UPPERLIMIT * np.sum(signals) + background)

    # Gaussian kde
    savefig = False
    if np.nanmin(assoc_samples) != np.nanmax(assoc_samples):
        assoc_samples_kde = np.concatenate(
            [assoc_samples, -assoc_samples, 2 - assoc_samples]
        )
        kernel = gaussian_kde(assoc_samples_kde, bw_method=0.01)
        try:
            x = np.linspace(0, 1, 1001)
            pdf = 3 * kernel(x)
            quants = cl_around_mode(x, pdf)
        except ValueError:
            x = np.linspace(0, 1, 10001)
            pdf = 3 * kernel(x)
            quants = cl_around_mode(x, pdf)
        lines = ax.plot(x, pdf, rasterized=True)
        # Quantiles
        peak = quants[0]
        lo = peak - quants[1]
        hi = quants[2] - peak
        if quants[1] < 0.01:
            quantstr = f"$p < {quants[2]:.2f}$"
        elif quants[2] > 0.99:
            quantstr = f"$p > {quants[1]:.2f}$"
        else:
            quantstr = f"${peak:.2f}_{{- {lo:.2f}}}^{{+ {hi:.2f}}}$"
        ax.fill_between(
            x,
            0,
            pdf,
            where=(x >= quants[1]) & (x <= quants[2]),
            color=lines[0].get_color(),
            alpha=0.5,
            lw=0,
            rasterized=True,
        )
        ax.text(
            0.05,
            # 3.15 - float(pa.basename(directory)) / 4, # for jobs 9, 10
            0.95,
            quantstr,
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="none",
                edgecolor=lines[0].get_color(),
                lw=1,
                pad=1,
            ),
            rasterized=True,
        )
        ax.text(
            0.05,
            # 3.15 - float(pa.basename(directory)) / 4, # for jobs 9, 10
            0.74,
            f"$p_{{\lambda = {LAMBDA_UPPERLIMIT}}} = {assoc_colocation:.2f}$",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="none",
                edgecolor=lines[0].get_color(),
                ls="--",
                lw=1,
                pad=1,
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
        # Plot line for colocation (prob GW occured at location of flare, sans lambda posterior)
        ax.vlines(
            assoc_colocation,
            0,
            3,
            color=lines[0].get_color(),
            ls="--",
            rasterized=True,
        )

    if savefig:
        ax.set_xlabel("Association Probability")
        ax.set_ylabel("PDF")
        plt.savefig(pa.join(directory, "association_pdf.png"))


def initialize_mosaic_axes(
    gweventnames,
    flarenames,
    subplot_mosaic_kwargs={
        "figsize": (12, 12),
        "gridspec_kw": {
            "wspace": 0.0,
            "hspace": 0.1,
        },
    },
):
    # Initialize figure
    mosaic_arr = []
    for fn in sorted(flarenames):
        mosaic_row = []
        for gwn in sorted(gweventnames):
            mosaic_row.append(f"{fn}|{gwn}")
        mosaic_row.append(f"{fn}|Background")
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
        for gi, gn in enumerate([*gweventnames, "Background"]):
            # Select axis
            ax = axs[f"{fn}|{gn}"]
            # Get the association probabilities
            if gn != "Background":
                plot_association_pdf(
                    directory,
                    s_arr_assoc.loc[gn, fn],
                    s_arr_assoc.loc[:, fn][~np.isnan(s_arr_assoc.loc[:, fn])],
                    b_arr_assoc.loc[gn, fn],
                    ax=ax,
                )
                assoc = s_arr_assoc.loc[gn, fn] != 0
            else:
                # Find background terms
                mask = (
                    b_arr_assoc.loc[
                        [n for n in b_arr_assoc.index], fn
                    ]
                    != 2.12e-6
                ).values
                if not np.any(mask):
                    gn_temp = gweventnames[0]
                else:
                    gn_temp = gweventnames[mask][0]
                plot_background_pdf(
                    directory,
                    s_arr_assoc.loc[:, fn][~np.isnan(s_arr_assoc.loc[:, fn])],
                    b_arr_assoc.loc[gn_temp, fn],
                    ax=ax,
                )
                assoc = True
            # Formatting
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 7)
            # Define flags
            bottom = fn == sorted(flarenames)[-1]
            top = fn == sorted(flarenames)[0]
            left = gn == sorted(gweventnames)[0]
            right = gn == "Background"
            # print(
            #     f"{gn:20s} {fn} bottom {bottom}, top {top}, left {left}, right {right}"
            # )
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
                ax.set_facecolor((0.9, 0.9, 0.9))
                tps = ax.tick_params()
                ax.tick_params(
                    "both",
                    length=1,
                )
            # Label axes
            if top:
                ax.annotate(
                    [*gweventnames, "Background"][gi],
                    xy=(0.5, 1.1),
                    xycoords="axes fraction",
                    ha="left",
                    # va="center",
                    rotation=45,
                    rasterized=True,
                )
            if bottom:
                if gi != len(gweventnames):
                    ax.set_xlabel(r"$p^{\rm GW-AGN}_{ij}$")
                else:
                    ax.set_xlabel(r"$p^{\rm BG-AGN}_{j}$")
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
    s_arrs,
    b_arrs,
    axs=None,
):
    # Config; select gwevents with associations
    d = directories[0]
    config_file = pa.join(d, "config.yaml")
    config = yaml.safe_load(open(config_file))
    assoc_path = config["assoc_csv"]
    if assoc_path == "None":
        assoc_path = pa.join(pa.dirname(config_file), "assoc.csv")
    df_assoc = pd.read_csv(assoc_path, index_col="gweventname")
    assoc_rows = np.any(df_assoc, axis=1)
    assoc_cols = np.any(df_assoc, axis=0)
    # Get gweventnames, flarenames
    gweventnames = np.array(df_assoc.index[assoc_rows])
    flarenames = np.array(df_assoc.columns[assoc_cols])
    # Initialize figure if needed
    if axs is None:
        fig, axs = initialize_mosaic_axes(gweventnames, flarenames)
    # Plot
    for d, s, b in zip(directories, s_arrs, b_arrs):
        # Trim s_arr and b_arr to rows with associations
        s_arr_assoc = s[assoc_rows]
        b_arr_assoc = b[assoc_rows]
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
    # ax.legend(
    #     handles=legend_elements,
    #     title="Flares/AGN/day",
    #     facecolor="white",
    #     edgecolor="black",
    #     framealpha=1,
    #     loc="lower left",
    # )
    # Save
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.8,
        left=0.05,
        right=0.85,
    )
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
    _default_array_jobs = [11]
    print(f"Usage: python {pa.basename(__file__)} <path_to_directory>")
    print(f"Defaulting to array jobs {_default_array_jobs}.")
    paths = [
        pa.join(PROJDIR, f"Posterior_inference_lambda_O3/array/{i}")
        for i in _default_array_jobs
    ]
else:
    paths = sys.argv[1:]

# Calc the association probabilities
s_arrs = []
b_arrs = []
for p in paths:
    s_arr, b_arr, _ = calc_arrs_for_directory(p, force=False)
    s_arrs.append(s_arr)
    b_arrs.append(b_arr)

# Plot the association probabilities
plot_association_pdfs(paths, s_arrs, b_arrs)
