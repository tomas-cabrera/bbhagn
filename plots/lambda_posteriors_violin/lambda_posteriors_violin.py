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


def plot_lambda_posterior(path, offset=0, plot_kwargs={}, ax=None):
    # Load samples
    samples = np.loadtxt(pa.join(path, "O4_samples_graham23.dat"))
    samples_kde = np.concatenate([samples, -samples, 2 - samples])
    # Gaussian kde
    kernel = gaussian_kde(samples_kde, bw_method=0.005)
    x = np.linspace(0, 0.35, 1001)
    pdf = 3 * kernel(
        x
    )  # "3 *" because the KDE is normalized over [-samples_max, samples_max]
    # Quantiles
    quants = cl_around_mode(x, pdf)
    peak = quants[0]
    lo = peak - quants[1]
    hi = quants[2] - peak
    if peak == 0:
        quantstr = f"$\lambda < {hi:.3f}$"
    else:
        quantstr = f"$\lambda = {peak:.3f}_{{- {lo:.3f}}}^{{+ {hi:.3f}}}$"
    # Scale to figure
    y_lo = offset - 0.45 * pdf / np.nanmax(pdf)
    y_hi = offset + 0.85 * pdf / 60
    # Plot
    # plot_kwargs["label"] += f": {quantstr}"
    lines = ax.plot(x, [offset] * len(x), rasterized=True, lw=0.5, **plot_kwargs)
    if "color" in plot_kwargs:
        color = plot_kwargs.pop("color")
    # ax.plot(x, y_lo, rasterized=True, color=color, lw=0.5, **plot_kwargs)
    ax.plot(x, y_hi, rasterized=True, color=color, lw=0.5, **plot_kwargs)
    ax.fill_between(
        x,
        offset,
        y_hi,
        where=(x >= quants[1]) & (x <= quants[2]),
        color=color,
        alpha=0.6,
        lw=0,
        rasterized=True,
    )
    ax.fill_between(
        x,
        offset,
        y_hi,
        where=(x >= 0) & (x <= np.quantile(samples, 0.9)),
        color=color,
        alpha=0.5,
        lw=0,
        rasterized=True,
    )
    # # Plot line for median
    # ax.vlines(
    #     peak,
    #     y_lo,
    #     pdf[np.digitize(peak, x) - 1] / np.nanmax(pdf) + y_lo,
    #     color=lines[0].get_color(),
    #     rasterized=True,
    # )
    ax.text(
        0.975 * x.max(),
        0.95 * ax.get_ylim()[1],
        f"{plot_kwargs['label']}\n$\lambda_{{1 \sigma}} = {hi:.3f}, \lambda_{{90\%}} = {np.quantile(samples, 0.9):.3f}$",
        ha="right",
        va="top",
        fontsize=10,
        # bbox=dict(
        #     facecolor="none",
        #     edgecolor=color,
        #     lw=1,
        #     pad=2,
        # ),
        rasterized=True,
    )
    # Print quantiles
    for q in [0.1, 0.16, 0.5, 0.84, 0.9]:
        v = np.quantile(samples, q)
        print(f"Quantiles {q}: {v:6.3f}")
    print(f"Bayes factor [peak={peak:.3f}]/0: {(kernel(peak) / kernel(0))[0]}")
    if peak != 0:
        calc_zero_cl(x, pdf)


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
    fig, axd = plt.subplot_mosaic(
        np.transpose([list(range(len(paths)))]),
        figsize=(4, 6),
        sharex=True,
        # sharey=True,
        gridspec_kw={"hspace": 0},
    )
    # Plot
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for pi, (path, label) in enumerate(
        zip(
            paths,
            [
                r"GWTC-3.0 (76 BBHs$\to$8 flares)",
                r"BBHs with flares (8 BBHs)",
                r"$m_1 < 40~M_\odot$ (47 BBHs$\to$3 flares)",
                r"$m_1 \geq 40~M_\odot$ (29 BBHs$\to$6 flares)",
                r"$m_{\rm fin} < 40~M_\odot$ (24 BBHs$\to$1 flare)",
                r"$m_{\rm fin} \geq 40~M_\odot$ (52 BBHs$\to$7 flares)",
                r"$L_{\rm bol}^{\rm AGN} \geq$ 3e42 erg/s",
                # "^same, only coincidences",
                r"$L_{\rm bol}^{\rm AGN} \geq$ 5e41 erg/s",
                # "^same, only coincidences",
            ],
        )
    ):
        plot_lambda_posterior(
            # plot_lambda_posterior_hist(
            path,
            # offset=-pi,
            ax=axd[pi],
            plot_kwargs={
                "label": label,
                "color": color_cycle[pi % len(color_cycle)],
            },
        )
        # Format
        axd[pi].set_xlim(0, 0.35)
        if pi == len(paths) - 1:
            axd[pi].set_xlabel(r"$\lambda$")
        # axd[pi].set_ylabel("PDF")
        # axd[pi].tick_params(left=False, labelleft=False)
        # ax.grid()
        # ax.legend(
        #     title="Flares/AGN/day",
        #     loc="upper right",
        #     edgecolor="k",
        # )
    # Save
    fig.supylabel("Posterior PDF")
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
    _default_array_jobs = [
        11,
        18,
        12,
        13,
        14,
        15,
        16,
        # 19,
        17,
        # 20,
    ]
    print(f"Usage: python {pa.basename(__file__)} <path_to_directory>")
    print(f"Defaulting to array jobs {_default_array_jobs}.")
    paths = [
        pa.join(PROJDIR, f"Posterior_inference_lambda_O3/array/{i}")
        for i in _default_array_jobs
    ]
else:
    paths = sys.argv[1:]

# Plot the association probabilities
plot_lambda_posteriors(paths)
