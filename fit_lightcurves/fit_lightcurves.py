import os
import os.path as pa
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

# Local imports
sys.path.append(pa.dirname(pa.dirname(__file__)))
from utils import graham23_tables as g23
from utils import paths
from utils.flaremorphology import graham23_flare_model
from utils.lightcurves import AlerceLightcurve

# Flare peak times (estimated from Graham+23 Fig. 3)
peak_times = {
    "J053408.41+085450.6": 58900,
    "J120437.98+500024.0": 58900,
    "J124942.30+344928.9": 58675,
    "J154342.46+461233.4": 58950,
    "J181719.94+541910.0": 58800,
    "J183412.42+365655.3": 58700,
    "J224333.95+760619.2": 58775,
}

# Iterate over flarenames
fitparams = []
for flarename in tqdm(np.unique(g23.DF_ASSOC["flarename"])):

    ## Load in data

    # Get lightcurve
    lightcurve = AlerceLightcurve(flarename)

    # Convert magnitudes to fluxes
    (
        lightcurve.data["mflux"],
        lightcurve.data["mflux_err_hi"],
        lightcurve.data["mflux_err_lo"],
    ) = lightcurve.calculate_fluxes()

    # Apply sigma cut
    lightcurve.apply_sigma_cut("mag", 5)

    ## Diagnostic plot

    # Initialize figure
    fig = plt.figure(figsize=(10, 6))

    # Iterate over filters
    filter2color = {"g": "xkcd:green", "r": "xkcd:red", "i": "xkcd:orange"}
    plot_data = "mag_binned"
    train_data = "mag_binned"
    ylim = []
    for f in np.unique(lightcurve.data["filter"]):
        # Mask for filter
        mask = lightcurve.data["filter"] == f
        fdata = lightcurve.data[mask]

        # Get data
        x = fdata["mjd"]
        if plot_data.startswith("mag"):
            y = fdata["mag"]
            y_err = fdata["mag_err"]
        elif plot_data.startswith("mflux"):
            y = fdata["mflux"]
            y_err_lo = fdata["mflux_err_lo"]
            y_err_hi = fdata["mflux_err_hi"]
            y_err = (y_err_lo + y_err_hi) / 2

        # Bin data
        if plot_data.endswith("binned") or train_data.endswith("binned"):
            # Bin data
            bins = np.arange(x.min(), x.max(), 8)
            i_bins = np.digitize(x, bins)

            # Calculate binned statistics
            x_binned = bins[:-1] + np.diff(bins) / 2
            y_binned = []
            y_err_binned = []
            for i in np.arange(1, len(bins)):
                # Select data in bin
                binmask = i_bins == i
                x_bin = x[binmask]
                y_bin = y[binmask]
                y_err_bin = y_err[binmask]
                try:
                    y_err_lo_bin = y_err_lo[binmask]
                    y_err_hi_bin = y_err_hi[binmask]
                except NameError:
                    pass

                # Skip empty bins
                if binmask.sum() == 0:
                    y_binned.append(np.nan)
                    try:
                        y_err_lo_bin
                        y_err_hi_bin
                        y_err_binned.append([np.nan, np.nan])
                    except NameError:
                        y_err_binned.append(np.nan)
                    continue

                # Define weights
                try:
                    weights = 1 / ((y_err_lo_bin + y_err_hi_bin) / 2) ** 2
                except NameError:
                    weights = 1 / y_err_bin**2

                # Calculate mean
                y_binned.append(
                    np.average(
                        y_bin,
                        weights=weights,
                    )
                )

                # Calculate error
                try:
                    y_err_binned.append(
                        [
                            1 / np.sum(1 / y_err_lo_bin**2) ** 0.5,
                            1 / np.sum(1 / y_err_hi_bin**2) ** 0.5,
                        ]
                    )
                except NameError:
                    y_err_binned.append(1 / np.sum(1 / y_err_bin**2) ** 0.5)

            # Cast as numpy arrays
            x_binned = np.array(x_binned)
            y_binned = np.array(y_binned)
            y_err_binned = np.array(y_err_binned)

        # If data to fit are specified
        if train_data:
            # If fitting to binned data
            if train_data.endswith("binned"):
                x_train = x_binned
                y_train = y_binned
                y_err_train = y_err_binned
            # Otherwise
            else:
                x_train = x
                y_train = y
                y_err_train = y_err

            # Get peak time
            t_peak = peak_times[flarename]

            # Get data around peak; remove NaNs
            tcrop = 200
            mask = (x_train >= t_peak - tcrop) & (x_train <= t_peak + tcrop)
            mask = mask & ~np.isnan(y_train)
            x_train = x_train[mask]
            y_train = y_train[mask]
            y_err_train = y_err_train[mask]

            # Fit lightcurve
            p0 = [t_peak, -1, 10, 20, 20]
            try:
                popt, pcov = curve_fit(
                    graham23_flare_model,
                    x_train,
                    y_train,
                    sigma=np.mean(y_err_train, axis=1),
                    p0=p0,
                )
            except np.AxisError:
                try:
                    popt, pcov = curve_fit(
                        graham23_flare_model,
                        x_train,
                        y_train,
                        sigma=y_err_train,
                        p0=p0,
                    )
                except RuntimeError:
                    popt = [np.nan] * len(p0)
            except RuntimeError:
                popt = [np.nan] * len(p0)

            # Save fit parameters
            fitparams.append(
                {
                    "flarename": flarename,
                    "filter": f,
                    **dict(
                        zip(
                            [
                                "t_peak",
                                "f_peak",
                                "t_rise",
                                "t_decay",
                                "f_base",
                            ],
                            popt,
                        )
                    ),
                }
            )

        # Plot data
        if plot_data.endswith("binned"):
            x_plot = x_binned
            y_plot = y_binned
            y_err_plot = y_err_binned.T
        else:
            x_plot = x
            y_plot = y
            y_err_plot = y_err.T
        plt.errorbar(
            x_plot,
            y_plot,
            yerr=y_err_plot,
            fmt=".",
            label=f,
            color=filter2color[f],
        )

        # Save plot limits before plotting fit data
        sigma_lim = 3
        ylim.append(
            [
                np.nanmean(y_plot) - sigma_lim * np.nanstd(y_plot),
                np.nanmean(y_plot) + sigma_lim * np.nanstd(y_plot),
            ]
        )

        # Plot fit
        if train_data:
            t_peak = peak_times[flarename]
            x_fit = np.linspace(t_peak - tcrop, t_peak + tcrop, 1000)
            y_fit = graham23_flare_model(x_fit, *popt)
            plt.plot(x_fit, y_fit, color=filter2color[f])

        # Plot peak time
        plt.axvline(t_peak, color="k", linestyle="--", alpha=0.5)

    # Format plot
    ylim = np.array(ylim)
    ylim = [np.nanmin(ylim[:, 0]), np.nanmax(ylim[:, 1])]
    plt.ylim(ylim)
    if plot_data.startswith("mag"):
        plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"{pa.dirname(__file__)}/{flarename}.png")
    plt.close()

# Save fit parameters
fitparams = pd.DataFrame(fitparams)
fitparams.to_csv(f"{pa.dirname(__file__)}/fitparams.csv", index=False)
