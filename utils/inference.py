import os.path as pa
import sys
from math import pi

import astropy.units as u
import astropy_healpix as ah
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.postprocess.crossmatch import crossmatch
import ligo.skymap.moc as lsm_moc
from myagn import distributions as myagndistributions
from myagn.flares import models as myflaremodels
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import norm

# Local imports
PROJDIR = pa.dirname(pa.dirname(__file__))
sys.path.append(PROJDIR)
import utils.graham23_tables as g23
import utils.io as io
from utils.gwselection import selection_beta

######################################################################


def gauss(mu_gauss, std_gauss, x_value):
    """! Calculate value of gaussian(mu_gauss, std_gauss) at x_value."""
    return np.exp(-((x_value - mu_gauss) ** 2.0) / (2 * std_gauss**2.0)) / (
        std_gauss * (2.0 * pi) ** 0.5
    )


def lnlike(theta, s_arr, b_arr, frac_det, n_flares_bg):
    """! Calculates the log likelihood on the given lambda for a single GW event.
    c.f. eq. 7 in the paper.

    @param  lam_li      lambda-like; the parameter to evaluate the likelihood of.
                        lambda = fraction of BBH mergers that produce AGN flares
    @param  s_arr       Probabilities for each of len(s_arr) BBH AGN flares
    @param  b_arr       Probabilities for each of len(b_arr) background AGN flares
    @param  f           Fraction of detected to real flares

    """
    # Extract values
    lam_li, H0_li, omegam_li = theta

    return (
        np.sum(np.log(lam_li * s_arr + b_arr))
        - np.log(selection_beta(H0_li))
        - frac_det * (lam_li + n_flares_bg)
    )


def lnlike_all(theta, s_arrs, b_arrs, frac_det, n_flares_bgs):
    # Extract values
    lam_po, H0_po, omega_po = theta

    # Calculate maximum in n_flares_bgs
    max_n_flares_bg = np.nanmax([np.nanmax(f) for f in n_flares_bgs if len(f) > 0])

    # Iterate over GW followups
    lnlike_arr = np.zeros(len(s_arrs))
    for i in range(len(s_arrs)):
        # Get arrays
        s_arr_i = s_arrs[i]
        b_arr_i = b_arrs[i]
        n_flares_bgs_i = n_flares_bgs[i]

        # Get non-filled values
        s_arr_i = s_arr_i[s_arr_i != 0]
        b_arr_i = b_arr_i[b_arr_i != 1]
        n_flares_bgs_i = n_flares_bgs_i[~np.isnan(n_flares_bgs_i)]

        ####################
        ###  Likelihood  ###
        ####################
        # If there are any candidates, calculate the log likelihood
        if (s_arr_i.shape[0]) > 0:
            # lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i] = lnlike(
                theta, s_arr_i, b_arr_i, frac_det, np.nanmax(n_flares_bgs_i)
            )

        else:
            # I think we need to do something different here, maybe s_arr needs to be set to 0
            # lnlike_arr[i] = -lam_po
            lnlike_arr[i] = -np.log(selection_beta(H0_po)) - frac_det * (
                lam_po + max_n_flares_bg
            )

    # Sum log likelihoods and return with prior
    lnlikesum = np.sum(lnlike_arr, axis=0)
    # fixer=lnlikesum[0]
    # like=np.exp(lnlikesum-fixer)
    # normalization=np.trapezoid(like,lam_arr)

    return lnlikesum


def calc_s_arr(
    pb,
    pb_frac,
    distnorm,
    distmu,
    distsigma,
    z_flares,
    cosmo,
    z_grid,
):
    ncands = z_flares.shape[0]
    dL_flares = cosmo.luminosity_distance(z_flares)
    # ddL/dz, to transform dprob/dd_L to dprob/dz
    jacobian = (
        cosmo.comoving_distance(z_flares).value
        + (1.0 + z_flares) / cosmo.H(z_flares).value
    )
    # Renormalize candidate posteriors along los to same z range as the background
    # (the original skymaps normalize the posteriors over z=[0,inf))
    new_normaliz = np.ones(ncands)
    ds_arr = cosmo.luminosity_distance(z_grid).value
    jacobian_arr = (
        cosmo.comoving_distance(z_grid).value + (1.0 + z_grid) / cosmo.H(z_grid).value
    )
    for l in range(ncands):
        this_post = (
            gauss(
                distmu[l],
                distsigma[l],
                ds_arr,
            )
            * ds_arr**2
            * jacobian_arr
        )
        # Calculate normalization constant by ~integrating posterior
        new_normaliz[l] = distnorm[l] * np.trapezoid(this_post, z_grid)
    # Calculate signal probabilities, multiplying HEALPix and redshift probabilities
    s_arr = (
        pb  # HEALPix probability
        / pb_frac  # This isn't in the function's namespace; needed to normalize over credible region
        * distnorm  # redshift probability
        * norm(distmu, distsigma).pdf(dL_flares)
        * dL_flares**2
        / new_normaliz
        * jacobian
    )
    return s_arr


def calc_b_arr(
    z_flares,
    f_cover,
    n_idx_sort_cut,
    B_expected_n,
    cosmo,
    z_grid,
    agndist_config,
):
    # Get AGN distribution
    agndist = getattr(
        myagndistributions,
        agndist_config["model"],
    )(
        *agndist_config["args"],
        **agndist_config["kwargs"],
    )

    # If Milliquas, add Graham+23 cuts
    # z <= 1.2 cut not included, because Graham+23 already does the 3D crossmatch and the z_grid we use here extends to 2.0
    if agndist_config["model"] == "Milliquas":
        mask = np.array(
            ["q" not in t for t in agndist._catalog["Type"]]
        )  # & (agndist._catalog["z"] <= 1.2)
        agndist._catalog = agndist._catalog[mask]

    # Calculate normalization for background probabilities
    pb_Vunif = agndist.dn_dOmega_dz(
        z_grid,
        *agndist_config["density_args"],
        cosmo=cosmo,
        **agndist_config["density_kwargs"],
    )
    normaliz = np.trapezoid(pb_Vunif, z_grid)

    # Calculate background probabilities
    b_arr = (
        agndist.dn_dOmega_dz(
            z_flares,
            *agndist_config["density_args"],
            cosmo=cosmo,
            **agndist_config["density_kwargs"],
        )
        / normaliz
        / n_idx_sort_cut
        * B_expected_n
        * f_cover
    )

    return b_arr


def calc_arrs(
    H0,
    omegam,
    pb_frac,
    f_covers,
    distnorm,
    distmu,
    distsigma,
    cand_hpxprob_arr,
    cand_zs_arr,
    n_idx_sort_cut,
    n_agn_coeffs,
    flares_per_agn,
    agndist_config,
    z_min_b,
    z_max_b,
):
    # Initialize cosmology
    thiscosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

    # Initialize redshift grid
    zs_arr = np.linspace(z_min_b, z_max_b, num=1000)

    # Iterate through true associations
    s_arrs = np.zeros_like(cand_hpxprob_arr)
    b_arrs = np.ones_like(cand_hpxprob_arr)
    B_expected_ns = np.ones_like(cand_hpxprob_arr) * np.nan
    # Iterate over GW followups
    for gwi in range(cand_hpxprob_arr.shape[0]):
        # Get flare indices
        fis = list(np.where(~np.isnan(cand_hpxprob_arr[gwi]))[0])

        # Skip remainder if no candidates
        if len(fis) == 0:
            continue

        # Get/calculate candidate coordinates, number of candidates for this followup
        followup_zs = cand_zs_arr[fis]
        ncands = followup_zs.shape[0]

        ####################
        ###Expected flares##
        ####################

        # Use coeffs to interpolate number of AGN in volume
        n_agn = np.polynomial.Polynomial(n_agn_coeffs[gwi])(H0)

        # Multiply by flare rate to get expected number of background flares
        B_expected_n = n_agn * flares_per_agn[fis]

        # Add to array
        B_expected_ns[gwi, fis] = B_expected_n

        ####################
        ###    Signal    ###
        ####################

        # Calculate GW counterpart probabilities
        s_arr = calc_s_arr(
            cand_hpxprob_arr[gwi][fis],
            pb_frac,
            distnorm[gwi][fis],
            distmu[gwi][fis],
            distsigma[gwi][fis],
            followup_zs,
            thiscosmo,
            zs_arr,
        )

        ####################
        ###  Background  ###
        ####################

        # Calculate background probabilities
        b_arr = calc_b_arr(
            followup_zs,
            f_covers[gwi],
            n_idx_sort_cut[gwi],
            B_expected_n,
            thiscosmo,
            zs_arr,
            agndist_config,
        )

        ####################
        ###    Add to arrs   ###
        ####################

        # Add to arrays
        s_arrs[gwi, fis] = s_arr
        b_arrs[gwi, fis] = b_arr

    return s_arrs, b_arrs, B_expected_ns


def _setup_task(i, config, df_fitparams):
    print("*" * 30)
    print(f"Index {i}")
    return_dict = {}

    ##############################
    ###  Signals (BBH flares)  ###
    ##############################

    # Get eventname, strip asterisk if needed
    gweventname = g23.DF_GW["gweventname"][i]
    if gweventname.endswith("*"):
        gweventname = gweventname[:-1]
    print(gweventname)

    ##############################
    ###       GW skymap        ###
    ##############################
    print("Loading skymap...")

    # Load skymap
    sm = io.get_gwtc_skymap(config["gwmapdir"], g23.DF_GW["gweventname"][i])

    # Get data from skymap
    sm["PROB"] = sm["PROBDENSITY"] * lsm_moc.uniq2pixarea(sm["UNIQ"])
    pb = np.array(sm["PROB"])
    distnorm = np.array(sm["DISTNORM"])
    distmu = np.array(sm["DISTMU"])
    distsigma = np.array(sm["DISTSIGMA"])

    # Make idx_sort_up (array of pb indices, from highest to lowest pb)
    idx_sort = np.argsort(pb)
    idx_sort_up = list(reversed(idx_sort))
    # Add pbs until pb_frac is reached
    sum = 0.0
    id = 0
    while sum < config["followup_prob"]:
        this_idx = idx_sort_up[id]
        sum = sum + pb[this_idx]
        id = id + 1
    # Cut indices to <pb_frac> credible region
    idx_sort_cut = idx_sort_up[:id]
    return_dict["n_idx_sort_cut"] = len(idx_sort_cut)

    ##############################
    ###         Flares         ###
    ##############################
    print("Loading flares...")

    # Get flares for this followup
    assoc_mask = g23.DF_ASSOC["gweventname"] == gweventname
    followup_flares = g23.DF_ASSOC["flarename"][assoc_mask]

    # Iterate over flares
    pbs = []
    distnorms = []
    distmus = []
    distsigmas = []
    for _, fr in g23.DF_FLARE.iterrows():
        # Check if flare in followup
        if fr["flarename"] not in followup_flares.values:
            pbs.append(np.nan)
            distnorms.append(np.nan)
            distmus.append(np.nan)
            distsigmas.append(np.nan)
            continue

        # Get uniqs for the flare
        lon = fr["flare_ra"] * u.deg
        lat = fr["flare_dec"] * u.deg
        uniq = io.lonlat_to_uniq(
            lon,
            lat,
            sm["UNIQ"],
        )
        ind_sm = np.where(sm["UNIQ"] == uniq)[0][0]

        # Get data for flare
        pbs.append(pb[ind_sm])
        distnorms.append(distnorm[ind_sm])
        distmus.append(distmu[ind_sm])
        distsigmas.append(distsigma[ind_sm])

    # Add to return_dict
    return_dict["pbs"] = pbs
    return_dict["distnorms"] = distnorms
    return_dict["distmus"] = distmus
    return_dict["distsigmas"] = distsigmas

    ##############################
    ###    Background rate     ###
    ##############################

    print("Calculating background rate...")

    ### Set AGN count model (polynomial as function of H0)
    # Get AGN distribution
    agndist = getattr(
        myagndistributions,
        config["agn_distribution"]["model"],
    )(
        *config["agn_distribution"]["args"],
        **config["agn_distribution"]["kwargs"],
    )

    # Set density_kwargs as attributes
    for k, v in config["agn_distribution"]["density_kwargs"].items():
        setattr(agndist, k, v)

    # Model number of AGNs
    # Define volume integrand
    def volume_integrand(
        dL,
        cosmo=FlatLambdaCDM(H0=config["H00"], Om0=config["Om0"]),
    ):
        # Includes unit conversions to cohere with myagn convention
        return (
            agndist._dn_d3Mpc_at_dL(
                dL * u.Mpc,
                cosmo=cosmo,
                **config["agn_distribution"]["density_kwargs"],
            )
            .to(u.Mpc**-3)
            .value
        )

    # Load skymap
    sm = io.get_gwtc_skymap(config["gwmapdir"], g23.DF_GW["gweventname"][i])
    # Iterate over a sample of H0 values
    n_agns = []
    hs = np.linspace(20, 120, num=10)
    for h in hs:
        tempcosmo = FlatLambdaCDM(H0=h, Om0=config["Om0"])
        n_agn = crossmatch(
            sm,
            contours=[0.9],
            cosmology=True,
            cosmo=tempcosmo,
            volume_integrand=volume_integrand,
        ).contour_vols[0]
        n_agns.append(n_agn)
    # Calculate polynomial fit to model n_agn as a function of H0
    fit = Polynomial.fit(hs, n_agns, 4)
    coeffs = fit.convert().coef
    return_dict["n_agn_coeffs"] = coeffs
    del sm

    # Append f_cover for GW event
    return_dict["f_covers"] = g23.DF_GW["f_cover"][i]

    return return_dict


def setup(config, df_fitparams, nproc=1):
    ##############################
    ###    Flares  ###
    ##############################

    # Set cand_zs
    cand_zs = g23.DF_FLARE["Redshift"].values

    # flares_per_agn rates
    # Set flare model
    flaremodel = getattr(
        myflaremodels,
        config["flare_rate"]["model"],
    )(
        *config["flare_rate"]["args"],
        **config["flare_rate"]["kwargs"],
    )

    # Iterate over flares
    flares_per_agn = []
    for _, fr in g23.DF_FLARE.iterrows():
        # Get fitparams
        df_fitparams_flare = df_fitparams[df_fitparams["flarename"] == fr["flarename"]]

        # Iterate over filters
        rates = []
        print("\n", fr["flarename"])
        for f in df_fitparams_flare["filter"]:
            # Use only g and r band data
            if f not in ["g", "r"]:
                continue

            # Get row
            params = df_fitparams_flare[df_fitparams_flare["filter"] == f]

            # Calculate structure function prob
            # 3σ = 3 * t_rise is a conservative estimate for the Δt of the Δm
            # / (1+z) converts to rest frame timescale
            delta_t = 3 * params["t_rise"].values[0] / (1 + fr["Redshift"])
            prob = flaremodel.flare_rate(
                f,
                delta_t,
                -params["f_peak"].values[0],
                fr["Redshift"],
            )

            # Scale to follow-up window
            rate = prob * config["dt_followup"]
            print(f, rate)

            # Append to list
            rates.append(rate)

        # Save lowest rate
        flares_per_agn.append(np.nanmin(rates))

    ##############################
    ###    GWs + flares        ###
    ##############################

    # Iterate over followups
    if nproc > 1:
        from multiprocessing import Pool

        with Pool(nproc) as p:
            results = p.starmap(
                _setup_task,
                [(i, config, df_fitparams) for i in g23.DF_GW.index],
            )
    else:
        results = [_setup_task(i, config, df_fitparams) for i in g23.DF_GW.index]

    # Unpack results
    f_covers = np.array([r["f_covers"] for r in results])
    distnorms = np.array([r["distnorms"] for r in results])
    distmus = np.array([r["distmus"] for r in results])
    distsigmas = np.array([r["distsigmas"] for r in results])
    pbs = np.array([r["pbs"] for r in results])
    n_idx_sort_cut = [r["n_idx_sort_cut"] for r in results]
    n_agn_coeffs = [r["n_agn_coeffs"] for r in results]
    flares_per_agn = np.array(flares_per_agn)

    return (
        f_covers,
        distnorms,
        distmus,
        distsigmas,
        pbs,
        cand_zs,
        n_idx_sort_cut,
        n_agn_coeffs,
        flares_per_agn,
    )
