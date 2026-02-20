import os
import os.path as pa
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from tqdm import tqdm

# Local imports
sys.path.append(pa.dirname(pa.dirname(__file__)))
from utils.paths import DATADIR
from utils import graham23_tables as g23
from utils.flaremorphology import graham23_flare_model
from utils.lightcurves import IRSAZTF, AlerceLightcurve

# Initilize query
ztfquery = IRSAZTF()

# Load flare info
flare_path = f"{DATADIR}/graham23_tables/flare20.csv"
df_flares = pd.read_csv(flare_path)

# Iterate over flarenames
fitparams = []
mjds = []
mjds_peak = []
tgs = []
ras = []
decs = []
force = False
for i, row in df_flares.iterrows():

    # SkyCoord
    rastr = row["flarename"][1:10]
    decstr = row["flarename"][10:19]
    sc = SkyCoord(
        float(rastr[0:2]) + float(rastr[2:4]) / 60 + float(rastr[4:]) / 3600,
        {"+": 1, "-": -1}[decstr[0]]
        * (float(decstr[1:3]) + float(decstr[3:5]) / 60 + float(decstr[5:]) / 3600),
        unit=(u.hourangle, u.deg),
    )
    ras.append(sc.ra.deg)
    decs.append(sc.dec.deg)

    ## Load in data

    # Get lightcurve
    lc_path = f"{DATADIR}/xml_lightcurves/{row['flarename']}.xml"
    if not pa.exists(lc_path) or force:
        # Query
        lc_result = ztfquery.conesearch(sc.ra.deg, sc.dec.deg, 3 / 3600)
        # Check for error
        if lc_result.status_code != 200:
            print(f"Error [{lc_result.status_code}] for query {lc_result.url}")
        # Save to file
        with open(lc_path, "wb") as f:
            for chunk in lc_result.iter_content(chunk_size=128):
                f.write(chunk)
    lc = Table.read(lc_path)

    # Convert magnitudes to fluxes
    lc["mflux"] = 10 ** (-0.4 * (lc["mag"] - lc["magzp"]))
    lc["mflux_err_hi"] = (1 - 10 ** (-0.4 * lc["magerr"])) * lc["mflux"]
    lc["mflux_err_lo"] = -(1 - 10 ** (0.4 * lc["magerr"])) * lc["mflux"]

    ## Diagnostic plot

    # Initialize figure
    fig = plt.figure(figsize=(10, 6))

    # Iterate over filters
    filter2color = {"zg": "xkcd:green", "zr": "xkcd:red", "zi": "xkcd:orange"}
    plot_data = "mag_binned"
    train_data = "mag_binned"
    ylim = []
    for f in np.unique(lc["filtercode"]):
        # Mask for filter
        mask = lc["filtercode"] == f
        fdata = lc[mask].copy()

        # Apply sigma cut
        mean = np.mean(fdata["mflux"])
        std = np.std(fdata["mflux"])
        mask = np.abs(fdata["mflux"] - mean) < 5 * std
        fdata = fdata[mask]

        # Get data
        x = fdata["mjd"]
        if plot_data.startswith("mag"):
            y = fdata["mag"]
            y_err = fdata["magerr"]
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
            t_peak = row["MJDpeak"]

            # Get data around peak; remove NaNs
            tcrop = 200
            mask = (x_train >= t_peak - tcrop) & (x_train <= t_peak + tcrop)
            mask = mask & ~np.isnan(y_train)
            x_train = x_train[mask]
            y_train = y_train[mask]
            y_err_train = y_err_train[mask]

            # Fit lightcurve
            p0 = [t_peak, -1, 10, 20, 20]
            bounds = (
                [t_peak - 10, -np.inf, 0, 0, -np.inf],
                [t_peak + 10, 0, np.inf, np.inf, np.inf],
            )
            try:
                popt, pcov = curve_fit(
                    graham23_flare_model,
                    x_train,
                    y_train,
                    sigma=np.mean(y_err_train, axis=1),
                    p0=p0,
                    bounds=bounds,
                )
            except np.exceptions.AxisError:
                try:
                    popt, pcov = curve_fit(
                        graham23_flare_model,
                        x_train,
                        y_train,
                        sigma=y_err_train,
                        p0=p0,
                        bounds=bounds,
                    )
                except RuntimeError:
                    popt = [np.nan] * len(p0)
            except RuntimeError:
                popt = [np.nan] * len(p0)

            # Save fit parameters
            fitparams.append(
                {
                    "flarename": row["flarename"],
                    "filter": f[-1],
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

            # Save g-band mjds
            if f == "zg":
                mjds_peak.append(popt[0])
                tgs.append(popt[2])
                mjds.append(
                    popt[0] - 3 * popt[2]
                )  # Cabrera+ cutoffs (3 gaussrise sigma before peak)
                # mjds.append(row["MJDpeak"] - row["tg"])  # Veronesi+23 cutoffs

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
            x_fit = np.linspace(t_peak - tcrop, t_peak + tcrop, 1000)
            y_fit = graham23_flare_model(x_fit, *popt)
            plt.plot(x_fit, y_fit, color=filter2color[f])

        # Plot peak time
        plt.axvline(t_peak, color="k", linestyle="--", alpha=0.5)

    # Format plot
    if ylim:
        ylim = np.array(ylim)
        if len(ylim.shape) > 1:
            ylim = [np.nanmin(ylim[:, 0]), np.nanmax(ylim[:, 1])]
        plt.ylim(ylim)
    if plot_data.startswith("mag"):
        plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(MultipleLocator(250))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
    plt.legend()
    plt.savefig(f"{pa.dirname(__file__)}/{row['flarename']}.png")
    plt.close()

# Save fit parameters
fitparams = pd.DataFrame(fitparams)
fitparams.to_csv(f"{pa.dirname(__file__)}/fitparams.csv", index=False)

# Save fit params to flare20
df_flares["mjd"] = mjds
df_flares["mjd_peak_fit"] = mjds_peak
df_flares["tg_rise_fit"] = tgs
df_flares["ra"] = ras
df_flares["dec"] = decs
df_flares.to_csv(flare_path, index=False)
