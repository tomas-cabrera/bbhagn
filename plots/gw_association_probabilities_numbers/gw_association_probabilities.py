import argparse
import glob
import os.path as pa
import sys
from copy import copy
from math import pi

import astropy.units as u
import astropy_healpix as ah
import ligo.skymap.moc as lsm_moc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch
from scipy.stats import gaussian_kde, norm

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.graham23_tables as g23
import utils.structurefunction as sf
from utils.paths import DATADIR, PROJDIR

# Style file
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")


def calc_arrs(
    theta,
    cand_hpixs_arr,
    cand_zs_arr,
    pb,
    distnorm,
    distmu,
    distsigma,
    frac_det,
    N_GW_followups,
    B_expected_n,
    n_idx_sort_cut,
):
    """! Calculates the log posterior probability, given some signal and background.

    @param  theta               Tuple of hypothesis lambda and H0
    @param  cand_hpixs_arr      Array of healpix index arrays for each followup
                                (shape=(N_GW_followup)(# of candidates in followup))
    @param  cand_zs_arr         Array of redshift arrays for each followup
                                (shape=(N_GW_followup)(# of candidates in followup))
    @param  pb                  Array of probability arrays for each candidate's healpixes
                                (shape=(N_GW_followup)(# of healpix in idx_sort_cut region))
    @param  distnorm            Normalization constant on distance estimate (takes into account d_L^2 weighting)
    @param  distmu              Mean of distance estimate for event, in Mpc
    @param  distsigma           Uncertainty on distance estimate, in Mpc
    @param  f                   Fraction of detected AGN flares to total AGN flares
    @param  N_GW_followups      Number of GW followups

    @return lp+lnlikesum        (log) posterior probability

    """
    # Extract lambda and H0 values
    lam_po, H0_po = theta

    # Cosmo. params
    omegam = 0.3
    thiscosmo = FlatLambdaCDM(H0=H0_po, Om0=omegam)

    # Iterate through followups
    z_min_b = 0.01  # Minimum z for background events
    z_max_b = 1.0  # Maximum z for background events
    zs_arr = np.linspace(z_min_b, z_max_b, num=1000)
    lnlike_arr = np.zeros(N_GW_followups)
    s_arrs = []
    b_arrs = []
    for i in range(N_GW_followups):
        ####################
        ###    Signal    ###
        ####################
        # Get/calculate candidate coordinates, number of candidates for this followup
        followup_hpixs = cand_hpixs_arr[i]
        followup_zs = cand_zs_arr[i]
        followup_ds = thiscosmo.luminosity_distance(followup_zs)
        ncands = followup_zs.shape[0]

        # Skip if no candidates
        if ncands == 0:
            s_arrs.append(np.array([]))
            b_arrs.append(np.array([]))
            continue

        # ddL/dz, to transform dprob/dd_L to dprob/dz
        jacobian = (
            thiscosmo.comoving_distance(followup_zs).value
            + (1.0 + followup_zs) / thiscosmo.H(followup_zs).value
        )
        # Renormalize candidate posteriors along los to same z range as the background
        # (the original skymaps normalize the posteriors over z=[0,inf))
        new_normaliz = np.ones(ncands)
        ds_arr = thiscosmo.luminosity_distance(zs_arr).value
        jacobian_arr = (
            thiscosmo.comoving_distance(zs_arr).value
            + (1.0 + zs_arr) / thiscosmo.H(zs_arr).value
        )
        for l in range(ncands):
            this_post = (
                gauss(
                    distmu[i][l],
                    distsigma[i][l],
                    ds_arr,
                )
                * ds_arr**2
                * jacobian_arr
            )
            # Calculate normalization constant by ~integrating posterior
            new_normaliz[l] = distnorm[i][l] * np.trapz(this_post, zs_arr)
        # Calculate signal probabilities, multiplying HEALPix and redshift probabilities
        s_arr = (
            pb[i]  # HEALPix probability
            / pb_frac  # This isn't in the function's namespace; needed to normalize over credible region
            * distnorm[i]  # redshift probability
            * norm(distmu[i], distsigma[i]).pdf(followup_ds)
            * followup_ds**2
            / new_normaliz
            * jacobian
        )

        # Append to list
        s_arrs.append(s_arr)

        ####################
        ###  Background  ###
        ####################
        # normaliz = np.trapz(ds_arr_norm**2,ds_arr_norm)
        # b_arr = followup_ds**2/normaliz/len(idx_sort_cut)*B_expected_n

        pb_Vunif = (
            thiscosmo.comoving_distance(zs_arr).value ** 2 / thiscosmo.H(zs_arr).value
        )
        normaliz = np.trapz(pb_Vunif, zs_arr)
        b_arr = (
            thiscosmo.comoving_distance(followup_zs).value ** 2
            / thiscosmo.H(followup_zs).value
            / normaliz
            / n_idx_sort_cut[i]
            * B_expected_n[i]
        )

        # Append to list
        b_arrs.append(b_arr)

    return s_arrs, b_arrs


def get_gwtc_skymap(
    mapdir,
    gweventname,
    catdirs={
        "GWTC2": "all_skymaps",
        "GWTC2.1": "IGWN-GWTC2p1-v2-PESkyMaps",
        "GWTC3": "skymaps",
    },
):
    # Choose waveform
    ## If in GWTC2/GWTC2.1:
    if gweventname in [
        "GW190408_181802*",
        "GW190412*",
        "GW190413_052954",
        "GW190413_134308",
        "GW190421_213856",
        "GW190424_180648",
        "GW190425",
        "GW190426_152155*",
        "GW190503_185404",
        "GW190512_180714*",
        "GW190513_205428",
        "GW190514_065416",
        "GW190517_055101",
        "GW190519_153544",
        "GW190521",
        "GW190521_074359",
        "GW190527_092055",
        "GW190602_175927",
        "GW190620_030421",
        "GW190630_185205",
        "GW190701_203306",
        "GW190706_222641",
        "GW190707_093326*",
        "GW190708_232457*",
        "GW190719_215514",
        "GW190720_000836*",
        "GW190727_060333",
        "GW190728_064510*",
        "GW190731_140936",
        "GW190803_022701",
        "GW190814*",
        "GW190828_063405",
        "GW190828_065509*",
        "GW190909_114149*",
        "GW190910_112807*",
        "GW190915_235702*",
        "GW190924_021846*",
        "GW190929_012149*",
        "GW190930_133541*",
    ]:
        gwtc = "GWTC2"

        ### If ends in asterisk
        if gweventname.endswith("*"):
            waveform = "SEOBNRv4PHM"
            gweventname = gweventname[:-1]
        else:
            waveform = "NRSur7dq4"
    ## elif in GWTC2.1
    elif gweventname in [
        "GW190403_051519",
        "GW190426_190642",
        "GW190725_174728",
        "GW190805_211137",
        "GW190916_200658",
        "GW190917_114630",
        "GW190925_232845",
        "GW190926_050336",
    ]:
        gwtc = "GWTC2.1"
        waveform = "IMRPhenomXPHM"

    ## elif in GWTC3:
    elif gweventname in [
        "GW191103_012549",
        "GW191105_143521",
        "GW191109_010717",
        "GW191113_071753",
        "GW191126_115259",
        "GW191127_050227",
        "GW191129_134029",
        "GW191204_110529",
        "GW191204_171526",
        "GW191215_223052",
        "GW191216_213338",
        "GW191219_163120",
        "GW191222_033537",
        "GW191230_180458",
        "GW200105_162426",
        "GW200112_155838",
        "GW200115_042309",
        "GW200128_022011",
        "GW200129_065458",
        "GW200202_154313",
        "GW200208_130117",
        "GW200208_222617",
        "GW200209_085452",
        "GW200210_092254",
        "GW200216_220804",
        "GW200219_094415",
        "GW200220_061928",
        "GW200220_124850",
        "GW200224_222234",
        "GW200225_060421",
        "GW200302_015811",
        "GW200306_093714",
        "GW200308_173609",
        "GW200311_115853",
        "GW200316_215756",
        "GW200322_091133",
    ]:
        waveform = "IMRPhenomXPHM"
        gwtc = "GWTC3"
    else:
        raise ValueError(f"Event {gweventname} not found in GWTC2 or GWTC3")

    # Search catalogs from newest to oldest
    catdir = catdirs[gwtc]
    # glob for the skymap path
    ## Special case: GW190521 (GW190521_074359 also exists)
    if gweventname == "GW190521":
        globstr = f"{mapdir}/{catdir}/{gweventname}_C01:{waveform}.fits"
    ## Special case: GW190425 (BNS --> special waveform)
    elif gweventname == "GW190425":
        globstr = f"{mapdir}/{catdir}/{gweventname}_C01:SEOBNRv4T_surrogate_HS.fits"
    ## Special case: GW191219_163120, GW200115_042309 (NSBH --> high/low spin waveforms)
    elif gweventname in ["GW191219_163120", "GW200115_042309"]:
        globstr = f"{mapdir}/{catdir}/*{gweventname}*{waveform}:HighSpin.fits"
    else:
        globstr = f"{mapdir}/{catdir}/*{gweventname}*{waveform}*.fits"
    mappaths = glob.glob(globstr)

    # If multiple skymaps are found
    if len(mappaths) > 1:
        raise ValueError(f"Multiple skymaps found for {gweventname}")
    elif len(mappaths) == 0:
        raise ValueError(f"No skymaps found for {gweventname}")
    else:
        # Load skymap
        hs = read_sky_map(mappaths[0], moc=True)

    return hs


def gauss(mu_gauss, std_gauss, x_value):
    """! Calculate value of gaussian(mu_gauss, std_gauss) at x_value."""
    return np.exp(-((x_value - mu_gauss) ** 2.0) / (2 * std_gauss**2.0)) / (
        std_gauss * (2.0 * pi) ** 0.5
    )


def get_flattened_skymap(mapdir, gweventname):

    # Get skymap
    hs = get_gwtc_skymap(mapdir, gweventname)

    # Flatten skymap
    hs_flat = Table(lsm_moc.rasterize(hs))
    hs_flat.meta = hs.meta

    # Calculate prob
    hs_flat["PROB"] = hs_flat["PROBDENSITY"] * ah.nside_to_pixel_area(
        ah.npix_to_nside(len(hs_flat))
    )

    return hs_flat


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--force", action="store_true")
args = argparser.parse_args()

# Make s_grid and b_grid paths
s_grid_path = f"{pa.dirname(__file__)}/s_grid.csv"
b_grid_path = f"{pa.dirname(__file__)}/b_grid.csv"

frac_det = 1.0  # frac_det=1 in the limit where we detect all AGNs flares
mapdir = "/hildafs/projects/phy220048p/share/skymaps"
dt_followup = 200
N_GW_followups = g23.DF_GW.shape[0]  # [50,10]

# Get posterior samples for lambda and H0
samples_path = f"{PROJDIR}/Posterior_sims_H0_O4/O4_samples_graham23.dat"
samples = np.loadtxt(samples_path)

# Calculate mode of lambda and H0
kernel = gaussian_kde(samples.T)
lmin, lmax = 0.0, 1.0
hmin, hmax = 20, 120
xgrid, ygrid = np.mgrid[lmin:lmax:101j, hmin:hmax:101j]
grid = np.vstack([xgrid.ravel(), ygrid.ravel()])
lambda_po, H0_po = grid[:, np.argmax(kernel(grid))]
print(f"lambda_po, H0_po:", lambda_po, H0_po)

# Calculate s_grid and b_grid if they don't exist, or if forced
if args.force or not pa.exists(s_grid_path) or not pa.exists(b_grid_path):

    # Iterate over followups
    cand_hpixs = []
    cand_zs = []
    pbs, distnorms, distmus, distsigmas = [], [], [], []
    B_expected_n = []
    n_idx_sort_cut = []
    for i in g23.DF_GW.index:
        ##############################
        ###  Signals (BBH flares)  ###
        ##############################

        # Load skymap
        hs_flat = get_flattened_skymap(mapdir, g23.DF_GW["gweventname"][i])

        # Get eventname, strip asterisk if needed
        gweventname = g23.DF_GW["gweventname"][i]
        if gweventname.endswith("*"):
            gweventname = gweventname[:-1]

        # Get data from skymap
        pb = np.array(hs_flat["PROB"])
        distnorm = np.array(hs_flat["DISTNORM"])
        distmu = np.array(hs_flat["DISTMU"])
        distsigma = np.array(hs_flat["DISTSIGMA"])
        NSIDE = ah.npix_to_nside(len(hs_flat))

        # Calculate <pb_frac> credible region; outputs healpix indices that compose region
        pb_frac = 0.90
        # Make idx_sort_up (array of pb indices, from highest to lowest pb)
        idx_sort = np.argsort(pb)
        idx_sort_up = list(reversed(idx_sort))
        # Add pbs until pb_frac is reached
        sum = 0.0
        id = 0
        while sum < pb_frac:
            this_idx = idx_sort_up[id]
            sum = sum + pb[this_idx]
            id = id + 1
        # Cut indices to <pb_frac> credible region
        idx_sort_cut = idx_sort_up[:id]
        n_idx_sort_cut.append(len(idx_sort_cut))

        # Get flares for this followup
        assoc_mask = g23.DF_ASSOC["gweventname"] == gweventname
        df_assoc_event = g23.DF_ASSOC[assoc_mask]

        # Get hpxs for the flares
        hpixs = np.array(
            ah.lonlat_to_healpix(
                df_assoc_event["flare_ra"].values * u.deg,
                df_assoc_event["flare_dec"].values * u.deg,
                NSIDE,
                order="nested",
            )
        )

        # Get redshifts for the flares
        zs = df_assoc_event["Redshift"].values

        # Compile positions and z for all candidates in this follow up
        cand_hpixs.append(np.array(hpixs))
        cand_zs.append(np.array(zs))
        if len(hpixs) == 0:
            pbs.append(np.array([]))
            distnorms.append(np.array([]))
            distmus.append(np.array([]))
            distsigmas.append(np.array([]))
        else:
            pbs.append(pb[hpixs])
            distnorms.append(distnorm[hpixs])
            distmus.append(distmu[hpixs])
            distsigmas.append(distsigma[hpixs])
        # cand_ds=np.concatenate((s_ds,b_ds))
        # ncands=cand_hpixs.shape[0]
        # cand_ds=np.zeros(ncands)
        # for l in range(ncands):
        #    cand_ds[l]=cosmo.luminosity_distance(cand_zs[l]).value

        # Skip volume calculation if no candidates
        if len(hpixs) == 0:
            B_expected_n.append([np.nan] * df_assoc_event.shape[0])
            continue

        ## Set AGNi/Mpc^3
        # Constant physical density
        agn_per_mpc3 = 10**-4.75

        ### Set flares/AGN rate
        ## Constant
        flares_per_agn = np.array([1e-4] * df_assoc_event.shape[0])
        ## Kimura+20 structure function
        # Load fit parameters
        df_fitparams = pd.read_csv(f"{PROJDIR}/fit_lightcurves/fitparams.csv")
        # Iterate over flares
        flares_per_agn = []
        for fn in df_assoc_event["flarename"]:
            # Get fitparams
            df_fitparams_flare = df_fitparams[df_fitparams["flarename"] == fn]

            # Iterate over filters
            rates = []
            print(fn)
            for f in df_fitparams_flare["filter"]:
                # Get row
                params = df_fitparams_flare[df_fitparams_flare["filter"] == f]

                # Calculate structure function prob
                # 3σ = 3 * t_rise is a conservative estimate for the Δt of the Δm
                delta_t = 3 * params["t_rise"].values[0]
                prob = sf.calc_kimura20_sf_prob(f, delta_t, -params["f_peak"])

                # Scale to follow-up window
                rate = prob * dt_followup / delta_t

                # Append to list
                # If g-band:
                if f == "g":
                    rates.append(rate)

                print(f, rate)

            # Save maximum rate
            flares_per_agn.append(max(rates))

        # Cast as array
        flares_per_agn = np.array(flares_per_agn)

        # Load skymap, calculate expected number of background flares
        hs = get_gwtc_skymap(mapdir, g23.DF_GW["gweventname"][i])
        vol90 = crossmatch(hs, contours=[0.9]).contour_vols[0]
        n_bg = vol90 * agn_per_mpc3 * flares_per_agn
        B_expected_n.append(n_bg)
        del hs

    # Calculate signal and background arrays
    s_arrs, b_arrs = calc_arrs(
        (lambda_po, H0_po),
        cand_hpixs,
        cand_zs,
        pbs,
        distnorms,
        distmus,
        distsigmas,
        frac_det,
        N_GW_followups,
        B_expected_n,
        n_idx_sort_cut,
    )

    # Iterate over followups
    gweventnames = g23.DF_GW["gweventname"]
    flarenames = np.unique(g23.DF_ASSOC["flarename"])
    s_grid = pd.DataFrame(
        np.zeros((gweventnames.shape[0], flarenames.shape[0])),
        index=gweventnames,
        columns=flarenames,
    )
    b_grid = copy(s_grid)
    print(s_grid)
    print(b_grid)
    for gii, gi in enumerate(g23.DF_GW.index):
        # Get eventname, strip asterisk if needed
        gweventname = g23.DF_GW["gweventname"][gi]
        if gweventname.endswith("*"):
            gweventname = gweventname[:-1]

        # Get flares for this followup
        assoc_mask = g23.DF_ASSOC["gweventname"] == gweventname
        df_assoc_event = g23.DF_ASSOC[assoc_mask]

        # Iterate over flares for event
        for fi, fn in enumerate(df_assoc_event["flarename"]):
            s_grid.loc[gweventname, fn] = s_arrs[gii][fi].value
            b_grid.loc[gweventname, fn] = b_arrs[gii][fi]

    s_grid.to_csv(s_grid_path)
    b_grid.to_csv(b_grid_path)

# Load s_grid and b_grid
s_grid = pd.read_csv(s_grid_path, index_col=0)
b_grid = pd.read_csv(b_grid_path, index_col=0)

# Keep only events with at least one flare
mask = np.nansum(s_grid, axis=1) > 0
s_grid = s_grid[mask]
b_grid = b_grid[mask]

# Convert nans to zeros
s_grid = s_grid.fillna(0)
b_grid = b_grid.fillna(0)

# Reorder df
ordered_gweventnames = [
    "GW190403_051519",
    "GW190514_065416",
    "GW190521",
    "GW190424_180648",
    "GW190731_140936",
    "GW190803_022701",
    "GW190909_114149",
    "GW200216_220804",
    "GW200220_124850",
]
ordered_flarenames = [
    "J124942.30+344928.9",
    "J183412.42+365655.3",
    "J224333.95+760619.2",
    "J181719.94+541910.0",
    "J053408.41+085450.6",
    "J120437.98+500024.0",
    "J154342.46+461233.4",
]
s_grid = s_grid.loc[ordered_gweventnames]
b_grid = b_grid.loc[ordered_gweventnames]
s_grid = s_grid.loc[:, ordered_flarenames]
b_grid = b_grid.loc[:, ordered_flarenames]

# Calculate probabilities
prob_grid = lambda_po * s_grid / (lambda_po * s_grid + b_grid)

### Plot

# Initialize figure, axes
fig, axd = plt.subplot_mosaic(
    """
    AC
    DB
    """,
    figsize=(6, 6),
    gridspec_kw={
        "width_ratios": [3, 2],
        "height_ratios": [3, 1],
    },
)

# Assign gws to axes
ax_to_gws = {
    "A": [
        "GW190403_051519",
        "GW190514_065416",
        "GW190521",
    ],
    "B": [
        "GW190424_180648",
    ],
    "C": [
        "GW190731_140936",
        "GW190803_022701",
        "GW190909_114149",
    ],
    "D": [
        "GW200216_220804",
        "GW200220_124850",
    ],
}

# Get min and max probabilities
min_prob = np.nanmin(prob_grid)
max_prob = np.nanmax(prob_grid)

# Iterate over axes
for axk, gwl in ax_to_gws.items():
    # Set axes
    ax = axd[axk]

    # Get flare info
    df_temp = prob_grid.loc[gwl]
    df_temp = df_temp.loc[:, np.any(~np.isnan(df_temp), axis=0)]

    # Trim colormap
    cmap = plt.get_cmap("Blues")
    cmap_floor = 0.2
    cmap = colors.LinearSegmentedColormap.from_list(
        "myBlues", cmap(np.linspace(cmap_floor, 1, 256))
    )

    # Plot
    im = ax.imshow(
        df_temp,
        cmap=cmap,
        aspect="auto",
        vmin=min_prob,
        vmax=max_prob,
    )

    # Annotations
    threshold = 0.6 * im.norm(np.nanmax(prob_grid))
    kw = dict(horizontalalignment="center", verticalalignment="center")
    for i in range(df_temp.shape[0]):
        for j in range(df_temp.shape[1]):
            val = df_temp.iloc[i, j]
            if ~np.isnan(val):
                kw.update(color=["black", "white"][int(im.norm(val) > threshold)])
                if val >= 2e-2:
                    label = f"{val:.2f}"
                else:
                    label = f"{val:.1e}"
                im.axes.text(j, i, label, **kw)

    # x tick labels
    ax.set_xticks(
        np.arange(len(df_temp.columns)),
        labels=[fn[:7] for fn in df_temp.columns],
    )
    ax.xaxis.tick_top()

    # y tick labels
    ax.set_yticks(
        np.arange(len(df_temp.index)),
        labels=[fn[:8] for fn in df_temp.index],
    )

    # Aspect ratio
    ax.set_aspect("equal")

    # Background color + grid
    bg_color = (0.95, 0.95, 0.95)
    ax.set_facecolor(bg_color)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(df_temp.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(df_temp.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=bg_color, linestyle="-", linewidth=3)
    ax.tick_params(which="minor", top=False, left=False)

# Save figure
plt.tight_layout()
plt.savefig(__file__.replace(".py", ".png"))
