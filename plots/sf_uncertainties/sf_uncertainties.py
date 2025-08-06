import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os.path as pa
from scipy.special import erfc

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
from myagn.flares import models as myflaremodels
import utils.graham23_tables as g23
from utils.paths import DATADIR, PROJDIR

# Initialize flare model
flaremodel = myflaremodels.Kimura20()

# Get fit params
df_fitparams = pd.read_csv(f"{PROJDIR}/fit_lightcurves/fitparams.csv")

# Initialize dt grid
dt_grid = np.linspace(0, 200, 1001)[1:]

# Iterate over flarenames
colors = {"g": "xkcd:green", "r": "xkcd:red"}
for fn, group in df_fitparams.groupby("flarename"):
    print(fn)
    print(group)

    # Get flare redshift
    z = g23.DF_ASSOC[g23.DF_ASSOC["flarename"] == fn]["Redshift"].values[0]

    # Initialize figure
    fig, ax = plt.subplots(1, 1)

    # Iterate over bands
    for f in ["g", "r"]:
        # Get flare params
        flareparams = group[group["filter"] == f]

        # Get sf params
        sfparams = flaremodel._sf_params.loc[f]
        print(sfparams)

        # Calculate base SF
        baseparams = {
            "SF0": sfparams["SF0"],
            "dt0": sfparams["dt0"],
            "bt": sfparams["bt"],
        }
        prob_base = (
            0.5
            * erfc(
                -flareparams["f_peak"].values[0]
                / flaremodel._calc_sf(**baseparams, dt=dt_grid)
                / (2**0.5)
            )
            / dt_grid
        )

        # Iterate over sources of error
        prob_errs = {}
        err_params = ["SF0", "bt"]
        for p in err_params:
            prob_errs_temp = []
            for i in [-1, 1]:
                params = baseparams.copy()
                params[p] = sfparams[p] + i * sfparams[f"{p}err"]
                prob_errs_temp.append(
                    0.5
                    * erfc(
                        -flareparams["f_peak"].values[0]
                        / flaremodel._calc_sf(**params, dt=dt_grid)
                        / (2**0.5)
                    )
                    / dt_grid
                )

            # Calculate minimum and maximium errors
            prob_min = np.minimum(*prob_errs_temp)
            prob_max = np.maximum(*prob_errs_temp)
            prob_errs[p] = {
                "min": prob_min,
                "max": prob_max,
            }

        # Calculate SF with both errors
        prob_errs_temp = []
        for SFi in [-1, 1]:
            for bti in [-1, 1]:
                params = baseparams.copy()
                params["SF0"] = sfparams["SF0"] + SFi * sfparams[f"SF0err"]
                params["bt"] = sfparams["bt"] + bti * sfparams[f"bterr"]
                prob_errs_temp.append(
                    0.5
                    * erfc(
                        -flareparams["f_peak"].values[0]
                        / flaremodel._calc_sf(**params, dt=dt_grid)
                        / (2**0.5)
                    )
                    / dt_grid
                )
        # Calculate minimum and maximium errors
        prob_min = np.min(prob_errs_temp, axis=0)
        prob_max = np.max(prob_errs_temp, axis=0)
        prob_errs["total"] = {
            "min": prob_min,
            "max": prob_max,
        }

        # Plot
        ax.plot(dt_grid, prob_base / prob_base, color=colors[f], label=f"{f}-band")
        lss = {
            "SF0": "-.",
            "bt": ":",
            "total": "--",
        }
        alphas = {
            "SF0": 0.5,
            "bt": 0.5,
            "total": 1,
        }
        for p in prob_errs:
            ax.fill_between(
                dt_grid,
                prob_errs[p]["min"] / prob_base,
                prob_errs[p]["max"] / prob_base,
                color=colors[f],
                alpha=0.1,
            )
            ax.plot(
                dt_grid,
                prob_errs[p]["min"] / prob_base,
                color=colors[f],
                linestyle=lss[p],
                alpha=alphas[p],
            )
            ax.plot(
                dt_grid,
                prob_errs[p]["max"] / prob_base,
                color=colors[f],
                linestyle=lss[p],
                alpha=alphas[p],
            )

        # Mark flare dt
        dt_flare = 3 * np.abs(flareparams["t_rise"].values[0]) / (1 + z)
        ax.axvline(
            dt_flare,
            color=colors[f],
            linestyle=(5, (10, 3)),
        )

    # Add for legend
    for p in prob_errs:
        plt.plot(
            [],
            [],
            color="k",
            label=f"{p} error",
            linestyle=lss[p],
            alpha=alphas[p],
        )
    plt.plot(
        [],
        [],
        color="k",
        label=f"flare dt",
        linestyle=(5, (10, 3)),
    )

    plt.title(fn)
    plt.ylim(1e-3, 1e3)
    plt.yscale("log")
    plt.xlabel(r"$\Delta t_{\rm rest}$ [days]")
    plt.ylabel(r"Flare rate / Base flare rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{pa.dirname(__file__)}/{fn}.png")
    plt.close()
