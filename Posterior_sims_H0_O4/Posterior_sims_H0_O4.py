"""! Script for the calculations described in https://arxiv.org/abs/2103.16069 """

###############################################################################

import glob
import os.path as pa
import random
import sys
from math import pi
from multiprocessing import Pool

import astropy.units as u
import astropy_healpix as ah
import corner
import emcee
import healpy as hp
import ligo.skymap.moc as lsm_moc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.io import fits
from astropy.table import Table
from joblib import Parallel, delayed
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch
from scipy import interpolate, optimize
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm
from tqdm import tqdm

# Local imports
sys.path.append(pa.dirname(pa.dirname(__file__)))
from utils.paths import DATADIR, PROJDIR
import utils.structurefunction as sf
import utils.graham23_tables as g23

###############################################################################

##############################
###         Colors         ###
##############################

# These are the "Tableau 20" colors as RGB.
tableau20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]

# Tableau Color Blind 10
tableau20blind = [
    (0, 107, 164),
    (255, 128, 14),
    (171, 171, 171),
    (89, 89, 89),
    (95, 158, 209),
    (200, 82, 0),
    (137, 137, 137),
    (163, 200, 236),
    (255, 188, 121),
    (207, 207, 207),
]

# Rescale to values between 0 and 1
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)
for i in range(len(tableau20blind)):
    r, g, b = tableau20blind[i]
    tableau20blind[i] = (r / 255.0, g / 255.0, b / 255.0)

###############################################################################

##############################
###        Functions       ###
##############################


def plt_style():
    plt.rcParams.update(
        {
            "lines.linewidth": 1.0,
            "lines.linestyle": "-",
            "lines.color": "black",
            "font.family": "serif",
            "font.weight": "normal",
            "font.size": 16.0,
            "text.color": "black",
            "text.usetex": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "axes.titlesize": "large",
            "axes.labelsize": "large",
            "axes.labelweight": "normal",
            "axes.labelcolor": "black",
            "axes.formatter.limits": [-4, 4],
            "xtick.major.size": 7,
            "xtick.minor.size": 4,
            "xtick.major.pad": 8,
            "xtick.minor.pad": 8,
            "xtick.labelsize": "large",
            "xtick.minor.width": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.size": 7,
            "ytick.minor.size": 4,
            "ytick.major.pad": 8,
            "ytick.minor.pad": 8,
            "ytick.labelsize": "large",
            "ytick.minor.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.numpoints": 1,
            "legend.fontsize": "large",
            "legend.shadow": False,
            "legend.frameon": False,
        }
    )


def lnprior(theta):
    """! Prior function for the lambda and H0 parameters.
    Currently defined as a uniform distibution in the initial domain.
    Returns 0.0 if theta is in the initial domain; -np.inf otherwise.

    @param  theta       The argument of the function.
                        Should be a tuple[-like]: (lambda, H0)

    """
    # Extract values
    lam_pr, H0_pr = theta
    # If the values are in the uniform domain
    if 0.0 < lam_pr < 1.0 and 20.0 < H0_pr < 120.0:
        return 0.0
    # Otherwise
    return -np.inf


def lnlike(lam_li, s_arr, b_arr, frac_det):
    """! Calculates the log likelihood on the given lambda.
    c.f. eq. 7 in the paper.

    @param  lam_li      lambda-like; the parameter to evaluate the likelihood of.
                        lambda = fraction of BBH mergers that produce AGN flares
    @param  s_arr       Probabilities for each of len(s_arr) BBH AGN flares
    @param  b_arr       Probabilities for each of len(b_arr) background AGN flares
    @param  f           Fraction of detected to real flares

    """
    return np.sum(np.log(lam_li * s_arr + b_arr)) - frac_det * lam_li


def lnprob(
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
    # Calculate prior; if the prior is not finite, return -np.inf (event outside of prescribed domain)
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    # Extract lambda and H0 values
    lam_po, H0_po = theta

    # Cosmo. params
    omegam = 0.3
    thiscosmo = FlatLambdaCDM(H0=H0_po, Om0=omegam)

    # Iterate through followups
    zs_arr = np.linspace(z_min_b, z_max_b, num=1000)
    lnlike_arr = np.zeros(N_GW_followups)
    for i in range(N_GW_followups):
        ####################
        ###    Signal    ###
        ####################
        # Get/calculate candidate coordinates, number of candidates for this followup
        followup_hpixs = cand_hpixs_arr[i]
        followup_zs = cand_zs_arr[i]
        followup_ds = thiscosmo.luminosity_distance(followup_zs)
        ncands = followup_zs.shape[0]
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
                    distmu[i][followup_hpixs[l]],
                    distsigma[i][followup_hpixs[l]],
                    ds_arr,
                )
                * ds_arr**2
                * jacobian_arr
            )
            # Calculate normalization constant by ~integrating posterior
            new_normaliz[l] = distnorm[i][followup_hpixs[l]] * np.trapz(
                this_post, zs_arr
            )
        # Calculate signal probabilities, multiplying HEALPix and redshift probabilities
        s_arr = (
            pb[i][followup_hpixs]  # HEALPix probability
            / pb_frac  # This isn't in the function's namespace; needed to normalize over credible region
            * distnorm[i][followup_hpixs]  # redshift probability
            * norm(distmu[i][followup_hpixs], distsigma[i][followup_hpixs]).pdf(
                followup_ds
            )
            * followup_ds**2
            / new_normaliz
            * jacobian
        )

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

        ####################
        ###  Likelihood  ###
        ####################
        # If there are any candidates, calculate the log likelihood
        if (s_arr.shape[0]) > 0:
            # lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i] = lnlike(lam_po, s_arr.value, b_arr, frac_det)

        else:
            # I think we need to do something different here, maybe s_arr needs to be set to 0
            lnlike_arr[i] = -lam_po

    # Sum log likelihoods and return with prior
    lnlikesum = np.sum(lnlike_arr, axis=0)
    # fixer=lnlikesum[0]
    # like=np.exp(lnlikesum-fixer)
    # normalization=np.trapz(like,lam_arr)
    return lp + lnlikesum


def gauss(mu_gauss, std_gauss, x_value):
    """! Calculate value of gaussian(mu_gauss, std_gauss) at x_value."""
    return np.exp(-((x_value - mu_gauss) ** 2.0) / (2 * std_gauss**2.0)) / (
        std_gauss * (2.0 * pi) ** 0.5
    )


def cl_around_mode(edg, myprob):

    peak = edg[np.argmax(myprob)]
    idx_sort_up = np.argsort(myprob)[::-1]

    i = 0
    bins = []
    integr = 0.0
    bmax = idx_sort_up[0]
    bmin = bmax
    bmaxbound = edg.shape[0] - 1
    bminbound = 0

    while integr < 0.68:
        if bmax == bmaxbound:
            bmin = bmin - 1
        elif bmin == bminbound:
            bmax = bmax + 1
        elif myprob[bmax + 1] > myprob[bmin - 1]:
            # print("Adding ",bmax_lo+1)
            bmax = bmax + 1
            bmin = bmin
            # bins_now_good = np.append(bins_now_good,
            bins.append(bmax + 1)
        else:
            # print("Adding ",bmin-1)
            bmin = bmin - 1
            bmax = bmax
            bins.append(bmin - 1)
        integr = simps(myprob[bmin:bmax], edg[bmin:bmax])
    print(integr, edg[bmin], edg[bmax])

    return peak, edg[bmin], edg[bmax]


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


###############################################################################

##############################
###       Constants        ###
##############################

frac_det = 1.0  # frac_det=1 in the limit where we detect all AGNs flares
lamb = 0.2  # da_array=[0.2,0.9]
N_GW_followups = 50  # [50,10]
# lamba_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
# N_GW_followups=[500,200,100,100,100,100,100,50,50,50,50]
H0 = 70.0
omegam = 0.3
mapdir = "/hildafs/projects/phy220048p/share/skymaps"
steps = 5000
z_min_b = 0.01  # Minimum z for background events
z_max_b = 1.0  # Maximum z for background events
nproc = 1  # Compute in parallel or not
lam_arr = np.linspace(0, 1.0, num=100)
dt_followup = 200

cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

##############################
###         Science        ###
##############################

maxdist = cosmo.luminosity_distance(z_max_b)
ds_arr_norm = np.linspace(0, maxdist.value, num=1000)

# Iterate over followups
cand_hpixs = []
cand_zs = []
pbs, distnorms, distmus, distsigmas = [], [], [], []
B_expected_n = []
n_idx_sort_cut = []
for i in tqdm(g23.DF_GW.index):
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
    pbs.append(pb)
    distnorms.append(distnorm)
    distmus.append(distmu)
    distsigmas.append(distsigma)
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


##############################
###          MCMC          ###
##############################

# Initialize some random lambdas and H0s for MCMC (w/ gaussian perturbations)
# np.random.randn(nwalkers, ndim=nparam)
pos = [lamb, H0] + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape
# Run the MCMC, in parallel or serial
if nproc != 1:
    with Pool(nproc) as pool:
        # Initialize sampler + run MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(
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
            ),
            pool=pool,
        )
        sampler.run_mcmc(pos, steps, progress=True)
else:
    # Initialize sampler + run MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lnprob,
        args=(
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
        ),
    )
    sampler.run_mcmc(pos, steps, progress=True)

# Reduce and retreive MCMC samples
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

# Save data
np.savetxt("O4_samples_graham23.dat", flat_samples)

# Make+save corner plot
labels = [r"$\lambda$", r"$H_0$ [km/s/Mpc]"]
fig = corner.corner(
    flat_samples,
    labels=labels,
    truths=[lamb, H0],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
fig.savefig("corner_O4_graham23.png", dpi=200)
